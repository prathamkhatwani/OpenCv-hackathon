"""Real-code regression tests. Generated fixtures are not field-validation data."""
import copy,json,time
import cv2,numpy as np,pytest
from fastapi.testclient import TestClient
from app import db
from app.main import app
from app.schemas import Options,Telemetry,Calibration
from app.storage import _get_local_dest_path
from app.cv.track import Track,build_tracks
from app.cv.contact import estimate_contact
from app.cv.motion import event_evidence
from app.cv.video import decode_video
from app.ml.decision import decide
from app.ml.features import FEATURES
from app.ml.train import validate,train
from app.ml.infer import FusionModel

@pytest.fixture(autouse=True)
def isolated(tmp_path,monkeypatch):
    monkeypatch.setenv('ROADREPORT_DB',str(tmp_path/'test.db'))
    monkeypatch.setenv('LOCAL_STORAGE_DIR',str(tmp_path/'storage'))
    monkeypatch.setenv('USE_LOCAL_STORAGE','true')
    monkeypatch.delenv('ROADREPORT_MODEL',raising=False)
    db.init_db()

@pytest.fixture
def video(tmp_path):
    path=tmp_path/'synthetic.avi'
    out=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),20,(640,360));assert out.isOpened()
    rng=np.random.default_rng(42);bg=np.clip(rng.normal(150,16,(360,640,3)),0,255).astype('uint8')
    for i in range(40):
        frame=bg.copy();cv2.ellipse(frame,(320,245+i//5),(40,24),0,0,360,(25,25,25),-1);out.write(frame)
    out.release();return path

def track_at(x):
    track=None
    for i in range(3):
        frame=np.zeros((360,640,3),dtype='uint8');f={'frame':frame,'gray':frame[:,:,0],'timestamp_s':i*.25,'index':i}
        d={'type':'pothole','confidence':.9,'bbox':(int(x*640)-3,180+i*9,6,6),'center_x':x*640,'center_y':183+i*9}
        if track is None:track=Track(d,f)
        else:track.add(d,f)
    return track

def calibration():
    return Calibration(homography=[[10,0,-5],[0,-20,20],[0,0,1]],straight_path_valid=True,camera_id='synthetic',calibration_note='Synthetic geometry unit fixture',lateral_uncertainty_m=.01).model_dump()

@pytest.mark.parametrize('key',['../escape','/etc/passwd','a/../../escape','a\\escape',''])
def test_storage_containment(key):
    with pytest.raises(ValueError):_get_local_dest_path(key)

def test_sidecars():
    with pytest.raises(ValueError):Telemetry(samples=[{'t':1},{'t':1}])
    with pytest.raises(ValueError):Telemetry(samples=[{'t':0,'lat':12},{'t':1}])
    with pytest.raises(ValueError):Options.model_validate({'made_up':3})
    with pytest.raises(ValueError):Calibration(homography=[[0]*3]*3,camera_id='a',calibration_note='invalid')

def test_contact_geometry_and_no_shock():
    missed=estimate_contact(track_at(.5),calibration(),None,10)
    assert missed['state']=='no_contact_expected'
    hit=estimate_contact(track_at(.425),calibration(),None,10)
    assert hit['state']=='contact_expected' and len(hit['windows'])==2
    assert estimate_contact(track_at(.425),None,None,10)['state']=='unknown'
    quiet=event_evidence(missed,[],None)
    assert decide(.9,3,missed,quiet)[0]=='confirmed'
    assert decide(.5,3,missed,quiet)[0]=='needs_review'
    assert decide(.1,3,hit,dict(quiet,shock=True))[0]=='dismissed'

def test_impact_time_and_axle_coverage():
    contact={'windows':[[1,1.4],[2,2.4]]}
    telemetry=Telemetry(samples=[{'t':i*.05,'az':4 if i==44 else 0} for i in range(61)]).model_dump()
    result=event_evidence(contact,[],telemetry)
    assert result['imu_available'] and result['shock'] and result['event_t']==2.2
    assert not event_evidence({'windows':[[.1,.5]]},[],telemetry)['shock']
    telemetry['synchronization_uncertainty_s']=1
    assert not event_evidence(contact,[],telemetry)['imu_available']
    assert decide(.9,3,{'state':'contact_expected'},dict(result,ambiguous=True))[0]=='needs_review'

def test_tracker_contract_gap_and_best_frame():
    t=track_at(.5);t.detections[1]['confidence']=.95
    assert t.best_frame_data['index']==1
    assert len(build_tracks([[d] for d in t.detections],t.frame_data_list))==1
    t.frame_data_list[-1]['timestamp_s']=3
    assert len(build_tracks([[d] for d in t.detections],t.frame_data_list))==2
    with pytest.raises(ValueError):build_tracks(t.detections,t.frame_data_list)

def test_native_motion_independent_of_visual_sampling(video):
    frames,motion,meta=decode_video(video)
    assert len(motion)==40 and 2<=len(frames)<=8
    assert meta['decoded_frames']==40

def test_full_api_pipeline_review_and_report(video):
    with TestClient(app) as client:
        with video.open('rb') as f:
            r=client.post('/api/jobs',files={'video':('synthetic.avi',f,'video/x-msvideo')},data={'options_json':json.dumps({'source_kind':'synthetic'})})
        assert r.status_code==201,r.text
        jid=r.json()['job_id'];deadline=time.monotonic()+40
        while time.monotonic()<deadline:
            j=client.get('/api/jobs/'+jid).json()
            if j['status'] in ('done','failed'):break
            time.sleep(.1)
        assert j['status']=='done',j
        assert j['evidence']['model']['state']=='baseline_untrained'
        assert j['evidence']['video']['decoded_frames']==40
        rows=client.get(f'/api/jobs/{jid}/findings').json()['findings'];assert rows
        for row in rows:
            assert row['agent_steps'] and all(s['finding_ref']==row['id'] for s in row['agent_steps'])
            assert client.get(row['image_url']).status_code==200
            assert client.post('/api/findings/'+row['id']+'/review',json={'decision':'reject'}).status_code==200
        j=client.get('/api/jobs/'+jid).json();assert j['summary']['by_status']['rejected']==len(rows)
        report=client.post(f'/api/jobs/{jid}/report');assert report.status_code==200,report.text
        assert client.get(report.json()['report_url']).content.startswith(b'%PDF')
        client.post('/api/findings/'+rows[0]['id']+'/review',json={'decision':'approve'})
        assert client.post(f'/api/jobs/{jid}/report').status_code==200
        assert client.get(f'/api/jobs/{jid}/export').status_code==200
        assert client.get('/api/storage/nope?expires=1&signature=invalid').status_code==403

def test_auth_and_invalid_upload(monkeypatch):
    import app.main as main
    monkeypatch.setattr(main,'API_KEY','test-key')
    with TestClient(app) as client:
        assert client.get('/api/jobs').status_code==401
        assert client.get('/api/health').status_code==200
        assert client.get('/api/jobs',headers={'X-API-Key':'test-key'}).status_code==200
        assert client.post('/api/jobs',headers={'X-API-Key':'test-key'},files={'video':('bad.mp4',b'not video')}).status_code==422

def training_records():
    rng=np.random.default_rng(7);records=[]
    for split in ['train','validation','test']:
        for i in range(40):
            exists=int(i%4!=0);contact='yes' if i%4 in (1,2) else 'no';impact=int(i%4==1)
            f=dict(zip(FEATURES,[float(rng.uniform(.6,.95) if exists else rng.uniform(.1,.5)),.8 if exists else .2,1,4,1 if contact=='yes' else 0,1,5 if impact else 0,.8,4 if impact else 0,1,8]))
            records.append(dict(event_id=f'{split}-{i}',drive_id=split,site_id=split,split=split,pothole_exists=exists,impact_associated=impact,contact_label=contact,source_kind='synthetic',features=f))
    return records

def test_training_export_and_no_contact_recall(tmp_path):
    records=training_records()
    with pytest.raises(ValueError):validate(records)
    validate(records,True);artifact=train(records)
    assert artifact['source_kind']=='synthetic' and artifact['test_metrics']['no_contact_pothole_recall']['n']==10
    p=tmp_path/'model.json';p.write_text(json.dumps(artifact));model=FusionModel(p);assert model.experimental
    assert set(model.predict(records[-1]['features']))=={'pothole_exists','impact_associated'}
    altered=copy.deepcopy(records);altered[-1]['drive_id']='train'
    with pytest.raises(ValueError,match='leaks'):validate(altered,True)

def test_benchmark_counts_misses_and_duplicates():
    from eval.eval import evaluate
    labels=[{'event_id':'a','pothole_exists':1,'contact_label':'no'},{'event_id':'b','pothole_exists':1,'contact_label':'yes'}]
    p={'prediction_id':'1','event_id':'a','vision_positive':True,'shock_positive':False,'fusion_positive':True}
    result=evaluate(labels,[p,dict(p,prediction_id='2')])
    assert result['fusion_positive']['tp']==1
    assert result['fusion_positive']['fp']==1
    assert result['fusion_positive']['fn']==1
    assert result['fusion_positive']['no_contact_recall']==1

def test_portrait_frame_memory_bound():
    from app.cv.video import frame_record
    record=frame_record(np.zeros((2000,100,3),dtype='uint8'),0,0)
    assert max(record['gray'].shape)<=640

def test_empty_visual_path_is_honest(tmp_path):
    from app.pipeline import run_pipeline
    from app.storage import upload_file
    p=tmp_path/'blank.avi';writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,180))
    for _ in range(10):writer.write(np.full((180,320,3),100,dtype='uint8'))
    writer.release();upload_file(str(p),'jobs/blank/video.avi')
    j=db.create_job(video_key='jobs/blank/video.avi')
    run_pipeline(j['id']);done=db.get_job(j['id'])
    assert done['status']=='done' and done['frames_used']==0
    assert db.get_findings(j['id'])==[]
    assert 'No retained sharp' in json.loads(done['evidence_json'])['coverage_note']
