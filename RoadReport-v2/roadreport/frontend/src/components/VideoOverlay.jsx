import {useEffect,useState} from 'react';
import {boxAtTime} from './trackOverlay.mjs';

export default function VideoOverlay({videoRef,findings,jobId,selectedId}) {
 const [time,setTime]=useState(0);
 useEffect(()=>{
  const player=videoRef.current;
  if(!player)return;
  let handle,disposed=false;
  const update=()=>setTime(player.currentTime);
  const tick=(_,metadata)=>{
   if(disposed)return;
   setTime(metadata?.mediaTime ?? player.currentTime);
   handle=player.requestVideoFrameCallback ? player.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
  };
  handle=player.requestVideoFrameCallback ? player.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
  player.addEventListener('seeked',update);
  player.addEventListener('loadedmetadata',update);
  player.addEventListener('pause',update);
  return()=>{
   disposed=true;
   if(player.cancelVideoFrameCallback)player.cancelVideoFrameCallback(handle);else cancelAnimationFrame(handle);
   player.removeEventListener('seeked',update);
   player.removeEventListener('loadedmetadata',update);
   player.removeEventListener('pause',update);
  };
 },[videoRef,jobId]);

 const isPaused = !videoRef.current || videoRef.current.paused;
 const targetFindings = (selectedId && isPaused)
   ? findings.filter(f => f.id === selectedId)
   : findings;

 const visible=targetFindings.filter(f=>!['rejected','dismissed'].includes(f.status)).map(f=>({finding:f,box:boxAtTime(f.measurements?.track,time)})).filter(x=>x.box);
 return <svg className="overlay" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-label={`${visible.length} visible tracked road defects`}>
  {visible.map(({finding:f,box:b})=>{
   const color=f.status==='needs_review'?'#ffca65':'#56efbc';
   const name=`${f.type} #${f.measurements?.track_id ?? f.id} · ${f.status==='needs_review'?'review':Math.round(f.confidence*100)+'%'}`;
   return <g key={f.id}><rect x={b[0]*1000} y={b[1]*1000} width={b[2]*1000} height={b[3]*1000} fill="none" stroke={color} strokeWidth="3" vectorEffect="non-scaling-stroke"/><text x={Math.min(b[0]*1000,720)} y={Math.max(25,b[1]*1000-10)} fill={color} stroke="#061016" strokeWidth="4" paintOrder="stroke" fontSize="22" fontWeight="700">{name}</text></g>;
  })}
 </svg>;
}
