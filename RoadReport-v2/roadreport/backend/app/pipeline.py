"""Explicit, bounded end-to-end orchestration. No demo fallback."""
import json,logging,os,tempfile,time,uuid
from pathlib import Path
import cv2
from app.db import get_job,update_job,create_finding,add_agent_step,refresh_summary
from app.storage import download_file,upload_file
from app.cv.video import decode_video
from app.cv.pothole import detect_potholes
from app.cv.crack import detect_cracks
from app.cv.track import build_tracks
from app.cv.severity import compute_severity
from app.agent import run_agent
from app.evidence import _draw_finding_box,_generate_reinspection_zoom
from app.ml.infer import FusionModel
logger=logging.getLogger(__name__)


def run_pipeline(job_id):
    start=time.monotonic()
    try:
        job=get_job(job_id)
        if not job:raise ValueError('Job not found')
        options=json.loads(job.get('options_json') or '{}')
        update_job(job_id,status='processing')
        with tempfile.TemporaryDirectory(prefix='roadreport_') as directory:
            d=Path(directory);video=d/'input.mp4';download_file(job['video_key'],str(video))
            frames,motion,meta=decode_video(video,lambda done,total:update_job(job_id,frames_done=done,frames_total=total))
            detections=[]
            for f in frames:
                candidates=detect_potholes(f['gray'],f['combined_mask'],f['roi_mask'],f['frame'])+detect_cracks(f['gray'],f['hsv'],f['combined_mask'],f['roi_mask'])
                detections.append(sorted(candidates,key=lambda x:x['confidence'],reverse=True)[:30])
            tracks=build_tracks(detections,frames)
            model_path=os.getenv('ROADREPORT_MODEL')
            model=FusionModel(model_path) if model_path else None
            model_info=model.info() if model else {'state':'baseline_untrained','note':'No trained fusion model configured. Scores are heuristic, not probabilities.'}
            results=run_agent(tracks,str(video),motion,options,meta,model)
            for result in results:
                track=result['track'];det=track.best_detection;f=track.best_frame_data
                fid=uuid.uuid4().hex
                score,label=compute_severity(det)
                coords=[]
                for dd,ff in zip(track.detections,track.frame_data_list):
                    hh,ww=ff['gray'].shape
                    x,y,w,h=dd['bbox'];coords.append({'t':ff['timestamp_s'],'box':[x/ww,y/hh,w/ww,h/hh]})
                measurements=dict(det.get('measurements',{}))
                measurements.update({k:result[k] for k in ['evidence_class','reason','initial_score','reinspection_score','reinspection_performed','contact','impact','predictions','features']})
                measurements.update(track_observations=track.n,track_seconds=coords[-1]['t']-coords[0]['t'],track=coords,coordinate_space={'width':f['frame'].shape[1],'height':f['frame'].shape[0]},model=model_info,severity_kind='visual_proxy')
                record={'bbox_json':json.dumps(list(det['bbox'])),'type':track.type,'confidence':result['visual_score'],'severity_label':label}
                annotated,crop=_draw_finding_box(f['frame'],record)
                keys={}
                for name,image in [('annotated',annotated),('crop',crop),('zoom',_generate_reinspection_zoom(crop,track.type))]:
                    path=d/f'{fid}_{name}.jpg'
                    if not cv2.imwrite(str(path),image):raise RuntimeError('Evidence encoding failed')
                    key=f'jobs/{job_id}/findings/{fid}_{name}.jpg';upload_file(str(path),key);keys[name]=key
                measurements['zoom_key']=keys['zoom']
                create_finding(job_id=job_id,finding_id=fid,type=track.type,severity_label=label,severity_score=score,confidence=result['visual_score'],status=result['status'],frame_index=f['index'],timestamp_s=f['timestamp_s'],bbox_json=record['bbox_json'],image_key=keys['annotated'],crop_key=keys['crop'],measurements_json=json.dumps(measurements,allow_nan=False))
                for number,step in enumerate(result['trace'],1):
                    add_agent_step(job_id=job_id,finding_ref=fid,step_no=number,action=step['action'],reason=step['reason'],confidence_before=step['before'],confidence_after=step['after'])
            evidence={'schema_version':1,'video':meta,'motion':motion,'telemetry':options.get('telemetry'),'calibration':options.get('calibration'),'source_kind':options.get('source_kind','real'),'source_note':options.get('source_note',''),'model':model_info,'opencv_version':cv2.__version__,'processing_seconds':round(time.monotonic()-start,3),'candidate_tracks':len(tracks),'track_limit_reached':len(tracks)>100,'coverage_note':'No retained sharp visual frames; inspection required' if not frames else 'Visual sampling and fixed ROI may miss defects. Motion is a camera proxy.'}
            refresh_summary(job_id)
            update_job(job_id,status='done',frames_done=meta['decoded_frames'],frames_total=meta['decoded_frames'],frames_used=len(frames),evidence_json=json.dumps(evidence,allow_nan=False))
            logger.info('job_completed job=%s elapsed=%.2f findings=%d',job_id,time.monotonic()-start,len(results))
    except Exception:
        logger.exception('job_failed job=%s',job_id)
        update_job(job_id,status='failed',error_message='Analysis failed. Check server logs using this job ID; no sample results were substituted.')
