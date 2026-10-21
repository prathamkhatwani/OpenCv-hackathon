import {useState,useEffect,useRef} from 'react';
import api,{errorText} from './api';
import RouteMap from './components/RouteMap';
import EvidenceChart from './components/EvidenceChart';
import Recorder from './components/Recorder';
import DemoMode from './components/DemoMode';
import './App.css';
const format=t=>`${Math.floor(t/60)}:${(t%60).toFixed(1).padStart(4,'0')}`;
const label=s=>(s||'unknown').replaceAll('_',' ');
const pct=v=>v==null?'unavailable':`${(v*100).toFixed(0)}%`;
async function readJson(file){if(!file)return null;if(file.size>2e6)throw Error('Sidecar exceeds 2 MB');return JSON.parse(await file.text());}
export default function App(){
 const [tab,setTab]=useState(()=>new URLSearchParams(window.location.search).has('demo')?'demo':'inspect'),[file,setFile]=useState(null),[telemetryFile,setTelemetryFile]=useState(null),[calibrationFile,setCalibrationFile]=useState(null),[location,setLocation]=useState(''),[source,setSource]=useState('real'),[key,setKey]=useState(sessionStorage.getItem('roadreport_key')||'');
 const [job,setJob]=useState(null),[findings,setFindings]=useState([]),[selected,setSelected]=useState(null),[time,setTime]=useState(0),[error,setError]=useState(''),[busy,setBusy]=useState(false),[history,setHistory]=useState([]),[filter,setFilter]=useState('all'),[health,setHealth]=useState(null),[notice,setNotice]=useState(''),[reportUrl,setReportUrl]=useState(null);
 const video=useRef(),generation=useRef(0),mounted=useRef(true);
 const selectedFinding=findings.find(f=>f.id===selected),m=selectedFinding?.measurements||{},e=job?.evidence||{},duration=e.video?.duration_s||1;
 useEffect(()=>{mounted.current=true;return()=>{mounted.current=false;};},[]);
 useEffect(()=>{if(tab==='demo')return;const abort=new AbortController();api.get('/health',{signal:abort.signal}).then(r=>setHealth(r.data)).catch(()=>{});return()=>abort.abort();},[tab]);
 useEffect(()=>{sessionStorage.setItem('roadreport_key',key);},[key]);
 const refreshHistory=()=>api.get('/jobs').then(r=>setHistory(r.data.jobs)).catch(err=>setError(errorText(err)));
 const poll=async(id,token)=>{
  try{
   const r=await api.get(`/jobs/${id}`);if(!mounted.current||token!==generation.current)return;
   setJob(r.data);
   if(r.data.status==='done'){
    const f=await api.get(`/jobs/${id}/findings`);if(token!==generation.current)return;setFindings(f.data.findings);setSelected(f.data.findings[0]?.id||null);setBusy(false);refreshHistory();
   }else if(r.data.status==='failed'){setError(r.data.error_message);setBusy(false);}else{setTimeout(()=>{if(mounted.current&&token===generation.current)poll(id,token);},1200);}
  }catch(err){if(token===generation.current){setError(errorText(err));setBusy(false);}}
 };
 const openJob=id=>{setReportUrl(null);setError('');setFindings([]);setBusy(true);setTab('inspect');poll(id,++generation.current);};
 const analyze=async ev=>{
  ev.preventDefault();setError('');setNotice('');setReportUrl(null);if(!file)return;
  if(file.size>100*1024*1024){setError('Maximum video size is 100 MB');return;}
  setBusy(true);setFindings([]);setJob(null);const token=++generation.current;
  try{
   const options={source_kind:source,telemetry:await readJson(telemetryFile),calibration:await readJson(calibrationFile)};
   const data=new FormData();data.append('video',file);data.append('location_text',location);data.append('options_json',JSON.stringify(options));
   const r=await api.post('/jobs',data);poll(r.data.job_id,token);
  }catch(err){setError(errorText(err));setBusy(false);}
 };
 const seek=t=>{setTime(t);if(video.current)video.current.currentTime=t;};
 const select=f=>{setSelected(f.id);seek(f.timestamp_s);};
 const review=async decision=>{
  setReportUrl(null);setError('');try{await api.post(`/findings/${selected}/review`,{decision,reviewer:'dashboard reviewer'});const [j,f]=await Promise.all([api.get(`/jobs/${job.id}`),api.get(`/jobs/${job.id}/findings`)]);setJob(j.data);setFindings(f.data.findings);setNotice('Review saved.');}catch(err){setError(errorText(err));}
 };
 const report=async()=>{setError('');try{const r=await api.post(`/jobs/${job.id}/report`);setReportUrl(r.data.report_url);}catch(err){setError(errorText(err));}};
 const downloadEvidence=async()=>{try{const r=await api.get(`/jobs/${job.id}/export`);const u=URL.createObjectURL(new Blob([JSON.stringify(r.data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download=`evidence-${job.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}catch(err){setError(errorText(err));}};
 const track=m.track||[];const nearby=track.length?track.reduce((a,b)=>Math.abs(b.t-time)<Math.abs(a.t-time)?b:a):null;
 const box=nearby&&Math.abs(nearby.t-time)<.16?nearby.box:null;
 return <div className="shell">
 <header><a className="brand" href="#" onClick={ev=>{ev.preventDefault();setTab('inspect');}}>◈ <span>RoadReport<small>EVIDENCE BEFORE ACTION</small></span></a><nav><button className={tab==='demo'?'nav active':'nav'} onClick={()=>setTab('demo')}>Demo</button><button className={tab==='inspect'?'nav active':'nav'} onClick={()=>setTab('inspect')}>Inspect</button><button className={tab==='capture'?'nav active':'nav'} onClick={()=>setTab('capture')}>Capture</button><button className={tab==='history'?'nav active':'nav'} onClick={()=>{setTab('history');refreshHistory();}}>History</button></nav><span className="runtime">{tab==='demo'?'SIMULATION':`OpenCV ${health?.opencv||'—'} · ${health?.detector||'YOLOv8n ONNX'}`}</span></header>
 {tab!=='demo'&&<div className="connection"><label>Team API key <input type="password" value={key} onChange={ev=>setKey(ev.target.value)} placeholder={health?.authentication_required?'Required by server':'Optional on localhost'}/></label><small>Stored in this tab only. {health?`${health.storage.toUpperCase()} storage`:'API unavailable — start the backend'}</small></div>}
 {tab==='demo'&&<DemoMode/>}
 {error&&<div className="error" role="alert">{error}<button className="nav" onClick={()=>setError('')}>Dismiss</button></div>}{notice&&<div className="notice" role="status">{notice}</div>}
 {tab==='capture'&&<Recorder/>}
 {tab==='history'&&<section className="panel"><h2>Inspection history</h2>{history.length?history.map(j=><button className="history" key={j.id} onClick={()=>openJob(j.id)}><span>{j.location_text||'Unnamed drive'}<small>{new Date(j.created_at).toLocaleString()}</small></span><span className="pill">{j.status}</span></button>):<p>No saved jobs.</p>}</section>}
 {tab==='inspect'&&<>
 <section className="hero"><div><span className="eyebrow">ROAD INSPECTION / 02</span><h1>See the defect.<br/><em>Understand the impact.</em></h1><p>Follow the visual evidence, tyre path and motion together. A pothole remains a pothole when the wheels miss it.</p><button className="secondary" onClick={()=>setTab('demo')}>▶ Try the sample simulation</button><div className="row"><span className="pill">Vision</span><span className="pill">Contact geometry</span><span className="pill">Motion evidence</span></div></div>
 <form className="panel upload" onSubmit={analyze}><label className="file">Dashcam video<input required type="file" accept=".mp4,.mov,.avi,.mkv,.webm" onChange={ev=>setFile(ev.target.files[0])}/><small>Up to 60 seconds · 100 MB</small></label><label>Location label<input value={location} onChange={ev=>setLocation(ev.target.value)} placeholder="Optional — not a GPS measurement" maxLength={300}/></label><div className="two"><label>Telemetry JSON (optional)<input type="file" accept=".json" onChange={ev=>setTelemetryFile(ev.target.files[0])}/></label><label>Calibration JSON (optional)<input type="file" accept=".json" onChange={ev=>setCalibrationFile(ev.target.files[0])}/></label></div><label>Evidence source<select value={source} onChange={ev=>setSource(ev.target.value)}><option value="real">Real recording</option><option value="synthetic">Synthetic / demonstration</option></select></label><button disabled={busy||!file}>{busy?'Processing…':'Analyze evidence →'}</button><small>Without calibration: contact unknown. Without trained weights: explicit heuristic baseline. No sample substitution.</small></form></section>
 {busy&&<section className="panel"><h3>{job?'Analyzing video':'Uploading and validating'}</h3><progress value={job?.frames_done||0} max={job?.frames_total||100}/><p>{job?`${job.frames_done||0} / ${job.frames_total||0} decoded frames — vision and agent inspection follow`:'Please wait…'}</p></section>}
 {job?.status==='done'&&!busy&&<>
 <div className="row spread section-title"><div><span className="eyebrow">INSPECTION WORKBENCH</span><h2>{job.location_text||'Road observation'}</h2></div><div className="row"><button className="secondary" onClick={downloadEvidence}>Evidence JSON</button><button onClick={report} disabled={findings.some(f=>f.status==='needs_review')}>Export reviewed PDF</button></div></div>
 <div>{reportUrl&&<a href={reportUrl} target="_blank" rel="noreferrer">Download generated PDF</a>}</div>
 <div className="notice">{e.detector?.backend==='opencv_dnn'&&<span className="pill" style={{marginRight:'8px',background:'rgba(50,213,176,0.18)',color:'#32d5b0',border:'1px solid #32d5b0',fontWeight:600}}>✓ YOLOv8n RDD2022 · OpenCV DNN</span>}{job.source_kind==='synthetic'?'SYNTHETIC RECORDING — not field-validation evidence. ':''}{e.detector?.backend==='opencv_dnn'?'Trained neural vision model active (RDD2022 dataset). ':e.model?.note} {e.rdd_visual_model?.state==='trained_visual_pilot'?'RDD2022 visual crop model is active. ':''} {e.coverage_note} {e.track_limit_reached?'Candidate limit reached; results are incomplete.':''}</div>
 <div className="stats"><div><b>{findings.length}</b><span>candidate findings</span></div><div><b>{findings.filter(f=>['confirmed','approved'].includes(f.status)).length}</b><span>supported / approved</span></div><div><b>{findings.filter(f=>f.measurements.contact.state==='no_contact_expected').length}</b><span>no contact expected</span></div><div><b>{findings.filter(f=>f.status==='needs_review').length}</b><span>need human review</span></div></div>
 <div className="workbench"><main><div className="panel player"><div className="row spread"><b>Recorded drive</b><small>{e.video?.width} × {e.video?.height} · {e.video?.fps?.toFixed(1)} fps · {format(duration)}</small></div><div className="video-stage" style={{aspectRatio:`${e.video?.width||16}/${e.video?.height||9}`}}><video ref={video} src={job.video_url} controls playsInline onTimeUpdate={ev=>setTime(ev.target.currentTime)} onError={()=>setError('Video unavailable or browser codec unsupported. Reload this job to renew links.')} />
 <svg className="overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none">{(m.contact?.corridors||[]).map((p,i)=><polygon key={i} points={p.map(([x,y])=>`${x*1000},${y*1000}`).join(' ')} fill="#32d5b0" fillOpacity=".14" stroke="#32d5b0" strokeWidth="2"/>)}{box&&<rect x={box[0]*1000} y={box[1]*1000} width={box[2]*1000} height={box[3]*1000} fill="none" stroke="#ffb45e" strokeWidth="4"/>}</svg></div><small>Boxes show observed track frames only. Green corridors require supplied calibration.</small></div>
 <EvidenceChart motion={e.motion} telemetry={e.telemetry} time={time} duration={duration} onSeek={seek} contact={m.contact}/>
 <details className="panel"><summary>Measured GPS track</summary><RouteMap samples={e.telemetry?.samples||[]} offset={e.telemetry?.time_offset_s||0} time={time}/></details>
 </main><aside className="panel"><div className="row spread"><h3>Findings</h3><select aria-label="Filter findings" value={filter} onChange={ev=>setFilter(ev.target.value)}><option value="all">All</option><option value="needs_review">Needs review</option><option value="confirmed">Supported</option><option value="dismissed">Dismissed</option></select></div>
 {!findings.length&&<div className="empty"><h3>No candidates returned</h3><p>This is not proof of a defect-free road. Check visibility and coverage; no sample findings have been inserted.</p></div>}
 <div className="findings">{findings.filter(f=>filter==='all'||f.status===filter).map(f=><button key={f.id} className={`finding ${selected===f.id?'selected':''}`} onClick={()=>select(f)}><div><b>{label(f.type)}</b><span>{format(f.timestamp_s)}</span></div><small>{label(f.measurements.evidence_class)}</small><span className={`pill ${f.status}`}>{label(f.status)}</span></button>)}</div></aside></div>
 {selectedFinding&&<section className="panel detail"><div className="row spread"><div><span className="eyebrow">EVIDENCE RECORD</span><h2>{label(selectedFinding.type)} · {format(selectedFinding.timestamp_s)}</h2><p>{m.reason}</p></div><span className={`pill ${selectedFinding.status}`}>{label(selectedFinding.status)}</span></div><div className="detail-grid"><div><img className="evidence-image" src={selectedFinding.image_url} alt="Observed frame with candidate bounding box"/><div className="two crops"><figure><img src={selectedFinding.crop_url} alt="Original crop"/><figcaption>Observed crop</figcaption></figure><figure><img src={selectedFinding.zoom_url} alt="Interpolated inspection crop"/><figcaption>Interpolated crop — no new depth information</figcaption></figure></div></div><div><dl>{[['Vision detector',m.detector?.detector?`${m.detector.detector.toUpperCase()} (${m.detector.backend||'OpenCV DNN'})`:'YOLOv8n (OpenCV DNN)'],['Confidence score',pct(selectedFinding.confidence)],['Reinspection score',m.reinspection_performed?m.reinspection_score?.toFixed(3):'Not requested'],['Observed frames',m.track_observations],['Track duration',`${m.track_seconds?.toFixed(2)}s`],['Contact',label(m.contact?.state)],['Motion source',label(m.impact?.source)],['Visual severity proxy',`${selectedFinding.severity_label} (${selectedFinding.severity_score})`],['RDD2022 visual crop probability',pct(m.rdd_visual?.pothole_probability)],['Trained event defect probability',pct(m.predictions?.pothole_exists)],['Trained association probability',pct(m.predictions?.impact_associated)]].map(([k,v])=><div key={k}><dt>{k}</dt><dd>{v??'unavailable'}</dd></div>)}</dl><p>{m.contact?.reason}</p><div className="row"><button onClick={()=>review('approve')}>Approve finding</button><button className="secondary" onClick={()=>review('reject')}>Reject finding</button></div><small>Review is saved before the status changes. Reports exclude pending and rejected findings.</small></div></div><h3>What the inspection agent did</h3><ol className="trace">{selectedFinding.agent_steps.map(step=><li key={step.id}><b>{label(step.action)}</b><p>{step.reason}</p>{step.confidence_before!=null&&<small>{step.confidence_before.toFixed(3)} → {step.confidence_after?.toFixed(3)} score</small>}</li>)}</ol></section>}
 </>}
 </>}
 <footer>RoadReport v2 · Advisory road inspection · No physical depth or autonomous driving claims</footer></div>;
}
