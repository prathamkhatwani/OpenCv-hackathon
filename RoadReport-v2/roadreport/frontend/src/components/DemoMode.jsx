import {useEffect,useRef,useState} from 'react';
import EvidenceChart from './EvidenceChart';
import './DemoMode.css';
const clock=t=>`${Math.floor(t/60)}:${(t%60).toFixed(1).padStart(4,'0')}`;
export default function DemoMode(){
 const [demo,setDemo]=useState(null),[error,setError]=useState(''),[time,setTime]=useState(0),[playing,setPlaying]=useState(false),[overlay,setOverlay]=useState(true);
 const video=useRef(null);
 useEffect(()=>{const abort=new AbortController();fetch('/demo/manifest.json',{signal:abort.signal}).then(r=>{if(!r.ok)throw Error('Demo manifest could not be loaded.');return r.json();}).then(setDemo).catch(e=>{if(e.name!=='AbortError')setError(e.message);});return()=>abort.abort();},[]);
 useEffect(()=>{
  if(!demo)return;const player=video.current;let handle;
  const tick=()=>{setTime(player.currentTime);handle=requestAnimationFrame(tick);};handle=requestAnimationFrame(tick);
  return()=>cancelAnimationFrame(handle);
 },[demo]);
 const seek=t=>{const value=Math.min(demo?.duration_s||18,Math.max(0,t));if(video.current)video.current.currentTime=value;setTime(value);};
 const toggle=async()=>{if(!video.current)return;if(video.current.paused){try{await video.current.play();setError('');}catch{setError('Playback could not start. Try the video controls or download the MP4.');}}else video.current.pause();};
 if(!demo)return <section className="panel" role="status">{error||'Loading the simulation…'}</section>;
 const index=Math.min(2,Math.floor(time/6)),scene=demo.cases[index],local=time-scene.start;
 const frame=demo.frames[Math.min(demo.frames.length-1,Math.floor(time*demo.fps))];
 const pulse=demo.motion[Math.min(demo.motion.length-1,Math.floor(time*demo.fps))]?.motion_score||0;
 const steps=[['Observe',scene.pothole?'A scripted pothole enters view.':'A scripted shadow enters view.',.5],['Compare tyre paths',scene.contact?'Left tyre corridor overlaps the pothole.':'The object lies outside both tyre corridors.',2],['Check the motion timing',scene.pulse==null?'No simulated pulse; none is expected.':scene.contact?'Pulse coincides with expected crossing.':'A scripted camera shake has an unrelated cause.',scene.pulse??4],['Explain the outcome',scene.reason,5.2]];
 return <section className="demo-mode">
  <div className="demo-intro"><div><span className="eyebrow">INTERACTIVE SIMULATION / 18 SECONDS</span><h1>One road.<br/><em>Three different stories.</em></h1><p>Watch what happens when vision, tyre paths and camera movement agree—or don’t.</p></div><div className="demo-disclosure"><span className="demo-dot"/> SYNTHETIC DEMO<p>Original animated video with scripted evidence. These are illustrated scenarios, not live detector results or measured sensor data.</p></div></div>
  {error&&<p className="error" role="alert">{error}</p>}
  <div className="demo-cases" role="group" aria-label="Choose a simulation scenario">{demo.cases.map((c,i)=><button key={c.id} aria-pressed={index===i} className={`demo-case ${index===i?'current':''}`} onClick={()=>seek(c.start+.05)}><span className="case-number">0{i+1}</span><span><strong>{c.title}</strong><small>{i===0?'Visual + contact + timed pulse':i===1?'Visual + no contact + no pulse':'Shadow + unrelated pulse'}</small></span><span className="case-time">{c.start}–{c.end}s</span></button>)}</div>
  <div className="demo-layout"><div>
   <div className="panel demo-player"><div className="row spread"><span className="pill">SIMULATION REPLAY</span><span className="demo-time">{clock(time)} <span>/ {clock(demo.duration_s)}</span></span></div>
    <div className="video-stage" style={{aspectRatio:'16/9'}}><video ref={video} src={demo.video} preload="auto" playsInline controls onPlay={()=>setPlaying(true)} onPause={()=>setPlaying(false)} onEnded={()=>setPlaying(false)} onTimeUpdate={e=>setTime(e.currentTarget.currentTime)} onError={()=>setError('The sample video could not load. Check that public/demo/road-simulation.mp4 exists.')} aria-label="Synthetic road simulation with three scenarios"/>
    {overlay&&<svg className="overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">{frame.corridors.map((p,i)=><polygon key={i} points={p.map(([x,y])=>`${x*1000},${y*1000}`).join(' ')} fill="#32d5b0" fillOpacity=".16" stroke="#32d5b0" strokeWidth="2"/>)}{frame.box&&<rect x={frame.box[0]*1000} y={frame.box[1]*1000} width={frame.box[2]*1000} height={frame.box[3]*1000} stroke={scene.pothole?'#ffb45e':'#b4a4fa'} strokeWidth="3" fill="none"/>}</svg>}
    </div><div className="row spread demo-controls"><div className="row"><button onClick={toggle}>{playing?'Pause simulation':'Play simulation'}</button><button className="secondary" onClick={()=>{seek(0);video.current?.pause();}}>Restart</button></div><label className="overlay-switch"><input type="checkbox" checked={overlay} onChange={e=>setOverlay(e.target.checked)}/> Show scripted overlays</label></div>
    <label className="demo-scrub">Seek video<input aria-label="Simulation time" type="range" min="0" max="18" step="0.04" value={time} onChange={e=>seek(Number(e.target.value))}/></label>
    <small>Teal: illustrated tyre corridors · orange: known pothole · purple: known shadow</small>
   </div>
   <EvidenceChart motion={demo.motion} time={time} duration={demo.duration_s} onSeek={seek} contact={{windows:scene.contact?[[4.8,5.2]]:[]}} synthetic/>
  </div><aside className="demo-evidence panel"><span className="eyebrow">CASE 0{index+1} · SCRIPTED EVIDENCE</span><h2>{scene.title}</h2><div className="demo-readings"><div><span>Object in the scene</span><b>{scene.pothole?'Pothole':'Shadow'}</b></div><div><span>Tyre path</span><b>{scene.contact?'Overlapping':'No contact'}</b></div><div><span>Current simulated pulse</span><b>{pulse.toFixed(1)} <small>/ 8</small></b></div><div><span>Expected pulse time</span><b>{scene.contact?'5.0 seconds':'None from this object'}</b></div></div>
   <div className={`demo-verdict ${local>=5.2?'revealed':''}`}><small>{local>=5.2?'ILLUSTRATED OUTCOME':'SCENARIO PREVIEW'}</small><h3>{scene.decision}</h3><p>{scene.reason}</p></div>
   <p className="demo-note">{index===1?'Key idea: absence of shock cannot reject a pothole that the tyres never touched.':index===2?'Key idea: movement is supporting evidence, not proof of a pothole.':'Key idea: timing and tyre overlap matter more than a shake alone.'}</p>
  </aside></div>
  <section className="panel demo-reasoning"><div className="row spread"><h2>Follow the reasoning</h2><small>Scripted walkthrough · progresses with the video</small></div><ol>{steps.map(([title,detail,at],i)=><li className={local>=at?'reached':''} key={title}><span>{local>=at?'✓':i+1}</span><div><h3>{title}</h3><p>{detail}</p></div></li>)}</ol></section>
  <div className="demo-downloads"><div><b>Take the sample with you</b><p>No account, upload or backend is required for this simulation.</p></div><div className="row"><a href={demo.video} download="roadreport-synthetic-demo.mp4">Download MP4 ↗</a><a href="/demo/manifest.json" download="roadreport-demo-manifest.json">Scripted evidence JSON ↗</a></div></div>
  <small className="demo-provenance">{demo.provenance} Demo playback creates no inspection jobs, reviews or reports.</small>
 </section>;
}
