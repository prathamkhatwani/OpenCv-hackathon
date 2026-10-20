"""Single-team API. Public deployments require an API key and HTTPS."""
import hashlib,hmac,json,logging,os,secrets,tempfile,threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
import cv2
from fastapi import FastAPI,File,Form,HTTPException,Request,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse,JSONResponse
from app.db import init_db,get_db,create_job,get_job,update_job,get_findings,get_finding,get_agent_steps,review_finding,list_jobs
from app.storage import upload_file,download_file,presign_get,is_local_storage,_get_local_dest_path
from app.pipeline import run_pipeline
from app.report import generate_report
from app.schemas import Options,Review
from app.cv.video import inspect_video
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)
API_KEY=os.getenv('ROADREPORT_API_KEY','')
SIGNING_SECRET=os.getenv('ROADREPORT_SIGNING_SECRET') or secrets.token_hex(32)
MAX_BYTES=100*1024*1024
pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='roadreport')
slots=threading.BoundedSemaphore(2)

@asynccontextmanager
async def lifespan(app):
    if os.getenv('ROADREPORT_ENV')=='production' and (len(API_KEY)<24 or len(os.getenv('ROADREPORT_SIGNING_SECRET',''))<32):
        raise RuntimeError('Production requires a 24+ character API key and 32+ character signing secret')
    if cv2.__version__.split('.')[0]!='5':raise RuntimeError('OpenCV 5 runtime required')
    init_db()
    with get_db() as conn:
        conn.execute("UPDATE jobs SET status='failed',error_message='Processing interrupted by server restart; upload again.' WHERE status IN ('uploaded','processing')")
    yield

app=FastAPI(title='RoadReport evidence API',version='2.0.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('ROADREPORT_ORIGINS','http://localhost:5173,http://127.0.0.1:5173').split(','),allow_methods=['GET','POST'],allow_headers=['Content-Type','X-API-Key'])

@app.middleware('http')
async def authorize(request:Request,call_next):
    path=request.url.path
    if request.method!='OPTIONS' and API_KEY and path.startswith('/api/') and path!='/api/health' and not path.startswith('/api/storage/'):
        if not hmac.compare_digest(request.headers.get('X-API-Key',''),API_KEY):
            return JSONResponse({'detail':'API key required'},status_code=401)
    return await call_next(request)

@app.exception_handler(Exception)
async def unexpected(request,exc):
    logger.exception('request_failed path=%s',request.url.path)
    return JSONResponse({'detail':'Internal error; consult server logs'},status_code=500)


def signed_url(key):
    if not is_local_storage():return presign_get(key,expires=3600)
    expires=int(time.time())+3600
    sig=hmac.new(SIGNING_SECRET.encode(),f'{key}:{expires}'.encode(),hashlib.sha256).hexdigest()
    return f'/api/storage/{key}?expires={expires}&signature={sig}'

@app.get('/api/health')
def health():
    return {'status':'ok','version':'2.0.0','opencv':cv2.__version__,'storage':'local' if is_local_storage() else 's3','authentication_required':bool(API_KEY)}

@app.get('/api/storage/{key:path}')
def local_file(key:str,expires:int,signature:str):
    if not is_local_storage():raise HTTPException(404,'Local storage disabled')
    expected=hmac.new(SIGNING_SECRET.encode(),f'{key}:{expires}'.encode(),hashlib.sha256).hexdigest()
    if expires<int(time.time()) or not hmac.compare_digest(expected,signature):raise HTTPException(403,'Expired or invalid evidence link')
    try:path=_get_local_dest_path(key)
    except ValueError:raise HTTPException(400,'Invalid evidence path')
    if not path.is_file():raise HTTPException(404,'Evidence not found')
    return FileResponse(path)


def execute(job_id):
    try:run_pipeline(job_id)
    finally:slots.release()

@app.post('/api/jobs',status_code=201)
async def upload(request:Request,video:UploadFile=File(...),location_text:str=Form(''),options_json:str=Form('{}')):
    if len(location_text)>300:raise HTTPException(422,'Location too long')
    if len(options_json)>2_000_000:raise HTTPException(413,'Sidecars exceed 2 MB')
    try:options=Options.model_validate_json(options_json).model_dump()
    except ValueError as e:raise HTTPException(422,str(e)[:600])
    extension=Path(video.filename or '').suffix.lower()
    if extension not in ('.mp4','.mov','.avi','.mkv','.webm'):raise HTTPException(400,'Supported formats: MP4, MOV, AVI, MKV, WebM')
    if not slots.acquire(blocking=False):raise HTTPException(429,'Processing queue full; retry after current job finishes')
    submitted=False
    try:
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/('video'+extension);total=0
            with path.open('wb') as output:
                while chunk:=await video.read(1024*1024):
                    total+=len(chunk)
                    if total>MAX_BYTES:raise HTTPException(413,'Video exceeds 100 MB')
                    output.write(chunk)
            try:meta=inspect_video(path)
            except ValueError as e:raise HTTPException(422,str(e))
            telemetry=options.get('telemetry')
            if telemetry:
                if any(not 0<=s['t']+telemetry['time_offset_s']<=meta['duration_s']+.1 for s in telemetry['samples']):
                    raise HTTPException(422,'Aligned telemetry timestamps must fall within video duration')
            jid=uuid.uuid4().hex;key=f'jobs/{jid}/video{extension}'
            upload_file(str(path),key)
            create_job(video_key=key,job_id=jid,location_text=location_text or None)
            update_job(jid,options_json=json.dumps(options,allow_nan=False),frames_total=meta['frame_count'])
            pool.submit(execute,jid);submitted=True
            return {'job_id':jid,'status':'uploaded'}
    finally:
        if not submitted:slots.release()
        await video.close()


def require_job(jid):
    job=get_job(jid)
    if not job:raise HTTPException(404,'Job not found')
    return job

def job_response(job):
    result={k:v for k,v in job.items() if k not in ('options_json','evidence_json','summary_json')}
    result['summary']=json.loads(job.get('summary_json') or '{}')
    result['evidence']=json.loads(job.get('evidence_json') or '{}')
    result['video_url']=signed_url(job['video_key'])
    result['source_kind']=json.loads(job.get('options_json') or '{}').get('source_kind','real')
    return result

@app.get('/api/jobs')
def jobs():
    return {'jobs':[{'id':j['id'],'status':j['status'],'location_text':j['location_text'],'created_at':j['created_at']} for j in list_jobs()]}

@app.get('/api/jobs/{jid}')
def job(jid:str):return job_response(require_job(jid))

@app.get('/api/jobs/{jid}/findings')
def findings(jid:str):
    require_job(jid);output=[]
    for finding in get_findings(jid):
        f=dict(finding);f['measurements']=json.loads(f.pop('measurements_json') or '{}');f['bbox']=json.loads(f.pop('bbox_json') or 'null')
        f['agent_steps']=get_agent_steps(jid,f['id'])
        for name,key in [('image_url',f.get('image_key')),('crop_url',f.get('crop_key')),('zoom_url',f['measurements'].get('zoom_key'))]:f[name]=signed_url(key) if key else None
        with get_db() as conn:f['reviews']=[dict(r) for r in conn.execute('SELECT decision,reviewer,created_at FROM reviews WHERE finding_id=? ORDER BY id',(f['id'],))]
        output.append(f)
    return {'findings':output,'total':len(output)}

@app.post('/api/findings/{fid}/review')
def review(fid:str,payload:Review):
    f=get_finding(fid)
    if not f:raise HTTPException(404,'Finding not found')
    if require_job(f['job_id'])['status']!='done':raise HTTPException(409,'Wait for job completion')
    updated=review_finding(fid,payload.decision,payload.reviewer)
    add_review_step=len(get_agent_steps(f['job_id'],fid))+1
    from app.db import add_agent_step
    add_agent_step(job_id=f['job_id'],finding_ref=fid,step_no=add_review_step,action='human_'+payload.decision,reason='Reviewed by '+payload.reviewer)
    return updated

@app.post('/api/jobs/{jid}/report')
def report(jid:str):
    j=require_job(jid)
    if j['status']!='done':raise HTTPException(409,'Wait for completed analysis')
    ff=get_findings(jid)
    if any(f['status']=='needs_review' for f in ff):raise HTTPException(409,'Resolve pending reviews before report export')
    selected=[f for f in ff if f['status'] in ('confirmed','approved')]
    with tempfile.TemporaryDirectory() as directory:
        for f in selected:
            if f.get('image_key'):download_file(f['image_key'],str(Path(directory)/Path(f['image_key']).name))
        path=str(Path(directory)/'report.pdf');generate_report(j,selected,directory,path)
        key=f'jobs/{jid}/reports/{uuid.uuid4().hex}.pdf';upload_file(path,key)
    return {'report_url':signed_url(key)}

@app.get('/api/jobs/{jid}/export')
def export(jid:str):
    j=require_job(jid)
    if j['status']!='done':raise HTTPException(409,'Analysis is not complete')
    # No signed URLs in archival evidence exports.
    return {'schema_version':1,'job':j,'findings':get_findings(jid),'trace':get_agent_steps(jid),'note':'Use measurements.features for labeled training events; labels must be annotated independently.'}

# Container serves the built dashboard on the same origin as the API.
if os.getenv('FRONTEND_DIST'):
    from fastapi.staticfiles import StaticFiles
    app.mount('/',StaticFiles(directory=os.environ['FRONTEND_DIST'],html=True),name='dashboard')
