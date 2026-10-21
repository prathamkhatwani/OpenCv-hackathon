export default function EvidenceChart({motion=[],telemetry,time,duration,onSeek,contact,synthetic=false}){
 const width=900,height=190,pad=36,d=Math.max(duration,1),offset=telemetry?.time_offset_s||0;
 const camera=motion.filter(s=>s.quality>=.55),imu=(telemetry?.samples||[]).filter(s=>s.az!=null);
 const x=t=>pad+t/d*(width-pad*2);
 const points=(rows,fn)=>rows.map(s=>`${x(s.t+(fn==='az'?offset:0))},${fn==='az'?145-Math.max(-10,Math.min(10,s.az))*3:80-Math.min(10,s.motion_score)*5}`).join(' ');
 return <div className="chart"><div className="row spread"><b>{synthetic?'Scripted camera motion':'Motion evidence'}</b><small>{synthetic?'Click to seek • orange: simulated pulse score':'Click to seek • orange: camera score • teal: vertical IMU (m/s²)'}</small></div>
 <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Camera-motion and optional IMU timeline" onClick={e=>{const r=e.currentTarget.getBoundingClientRect();onSeek(Math.max(0,Math.min(d,((e.clientX-r.left)/r.width*width-pad)/(width-pad*2)*d)));}}>
 {(contact?.windows||[]).map(([a,b],i)=><rect key={i} x={x(a)} y="20" width={Math.max(1,x(b)-x(a))} height="150" fill="#a998ff" opacity=".16"/>)}
 {[80,145].map(y=><line key={y} x1={pad} x2={width-pad} y1={y} y2={y} stroke="#364c55"/>)}
 <text x="4" y="45">10</text><text x="4" y="82">0</text><text x="4" y="119">+10</text><text x="4" y="177">−10</text>
 <polyline fill="none" stroke="#ffb45e" strokeWidth="1.6" points={points(camera,'motion_score')}/>
 <polyline fill="none" stroke="#34d9bb" strokeWidth="1.5" points={points(imu,'az')}/>
 {!imu.length&&<text x={width/2} y="145" textAnchor="middle">{synthetic?'No IMU measurements — illustration only':'IMU unavailable — camera motion is a proxy'}</text>}
 <line x1={x(time)} x2={x(time)} y1="15" y2="175" stroke="#fff" strokeWidth="1.5"/>
 {[0,.25,.5,.75,1].map(f=><text key={f} x={x(d*f)} y="188" textAnchor="middle">{(d*f).toFixed(1)}s</text>)}
 </svg><small>{synthetic?'Purple: scripted tyre-crossing interval. This waveform comes from the animation, not sensor measurements.':'Purple windows: predicted wheel-contact intervals. Scores and thresholds are experimental; a pulse alone does not establish a pothole.'}</small></div>;
}
