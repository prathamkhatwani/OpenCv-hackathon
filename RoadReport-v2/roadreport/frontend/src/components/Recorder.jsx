import {useState,useRef,useEffect} from 'react';
export default function Recorder(){
 const [status,setStatus]=useState('idle'),[message,setMessage]=useState(''),[downloads,setDownloads]=useState(null);
 const video=useRef(),cleanup=useRef(()=>{}),urls=useRef([]),timer=useRef();
 useEffect(()=>()=>{cleanup.current();clearTimeout(timer.current);urls.current.forEach(URL.revokeObjectURL);},[]);
 const start=async()=>{
  setMessage('');setStatus('starting');
  try{
   if(!window.isSecureContext||!navigator.mediaDevices||!window.MediaRecorder)throw Error('Recording needs HTTPS (or localhost) and a browser with MediaRecorder.');
   if(typeof DeviceMotionEvent!=='undefined'&&typeof DeviceMotionEvent.requestPermission==='function')await DeviceMotionEvent.requestPermission();
   const stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1280},height:{ideal:720}},audio:false});
   cleanup.current=()=>stream.getTracks().forEach(t=>t.stop());
   video.current.srcObject=stream;await video.current.play();
   const format=['video/mp4','video/webm;codecs=vp8','video/webm'].find(t=>MediaRecorder.isTypeSupported(t));
   const recorder=new MediaRecorder(stream,format?{mimeType:format}:{});const chunks=[],samples=[];let startTime=0,gps=null,watch=null;
   const motion=e=>{
    if(!startTime)return;const a=e.acceleration,g=e.accelerationIncludingGravity;let az=null;
    if(a&&g&&[a.x,a.y,a.z,g.x,g.y,g.z].every(v=>typeof v==='number'&&Number.isFinite(v))){
     const gravity=[g.x-a.x,g.y-a.y,g.z-a.z],norm=Math.hypot(...gravity);
     if(norm>7&&norm<12)az=(a.x*gravity[0]+a.y*gravity[1]+a.z*gravity[2])/norm;
    }
    const t=(performance.now()-startTime)/1000;if(t<0||t>59)return;
    const row={t,az};const rr=e.rotationRate;
    if(rr){for(const [key,val] of [['gx',rr.beta],['gy',rr.gamma],['gz',rr.alpha]])if(typeof val==='number')row[key]=val*Math.PI/180;}
    if(gps&&performance.now()-gps.received<2000){row.lat=gps.coords.latitude;row.lon=gps.coords.longitude;if(gps.coords.speed!=null&&gps.coords.speed>=0)row.speed_mps=gps.coords.speed;}
    if(!samples.length||t>samples[samples.length-1].t)samples.push(row);
   };
   window.addEventListener('devicemotion',motion);
   if(navigator.geolocation)watch=navigator.geolocation.watchPosition(p=>{gps={coords:p.coords,received:performance.now()};},()=>{}, {enableHighAccuracy:true});
   recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
   const finish=()=>{stream.getTracks().forEach(t=>t.stop());window.removeEventListener('devicemotion',motion);if(watch!=null)navigator.geolocation.clearWatch(watch);};
   recorder.onstart=()=>{startTime=performance.now();setStatus('recording');};
   recorder.onstop=()=>{finish();setStatus('idle');urls.current.forEach(URL.revokeObjectURL);
    const blob=new Blob(chunks,{type:recorder.mimeType});const v=URL.createObjectURL(blob);
    const side={schema_version:1,units:'SI',acceleration_frame:'vehicle_vertical_gravity_removed',time_offset_s:0,synchronization_uncertainty_s:.15,samples};
    const s=URL.createObjectURL(new Blob([JSON.stringify(side,null,2)],{type:'application/json'}));urls.current=[v,s];
    setDownloads({video:v,telemetry:s,ext:recorder.mimeType.includes('mp4')?'mp4':'webm',count:samples.length});
    setMessage('Recorder timestamps are approximate. Verify alignment before making impact claims.');
   };
   cleanup.current=()=>{if(recorder.state!=='inactive')recorder.stop();finish();};recorder.start(200);timer.current=setTimeout(()=>cleanup.current(),30000);
  }catch(e){cleanup.current();setStatus('idle');setMessage(e.message);}
 };
 return <section className="panel"><h2>Capture paired evidence</h2><p>Mount the phone securely; start recording while parked. Use ordinary safe journeys. Never deliberately hit potholes. Recording stops after 30 seconds.</p>
 <video className="recorder" ref={video} muted playsInline/>
 <div className="row"><button onClick={start} disabled={status!=='idle'}>Start camera + sensors</button><button className="secondary" disabled={status!=='recording'} onClick={()=>{clearTimeout(timer.current);cleanup.current();}}>Stop</button></div>
 {message&&<p role="status">{message}</p>}
 {downloads&&<div className="notice"><a href={downloads.video} download={`drive.${downloads.ext}`}>Download video</a> · {downloads.count>=2?<a href={downloads.telemetry} download="telemetry.json">Download telemetry ({downloads.count} samples)</a>:'No usable sensor samples; use video-only analysis.'}<p>Upload the saved files in Inspect. Some browsers produce files with missing duration metadata; see the FFmpeg remux command in the README.</p></div>}
 </section>;
}
