import {useEffect,useRef,useMemo} from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
export default function RouteMap({samples,offset=0,time=0}){
 const box=useRef(null),map=useRef(null),marker=useRef(null);
 const gps=useMemo(()=>samples.filter(s=>s.lat!=null&&s.lon!=null),[samples]);
 useEffect(()=>{
  if(!box.current||gps.length<2)return;
  const m=L.map(box.current);map.current=m;
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap contributors',maxZoom:19}).addTo(m);
  const line=L.polyline(gps.map(s=>[s.lat,s.lon]),{color:'#10bfa5'}).addTo(m);m.fitBounds(line.getBounds(),{padding:[20,20],maxZoom:17});
  marker.current=L.circleMarker([gps[0].lat,gps[0].lon],{radius:7,color:'#f7af54'}).addTo(m);
  const observer=new ResizeObserver(()=>m.invalidateSize());observer.observe(box.current);
  return()=>{observer.disconnect();m.remove();map.current=null;marker.current=null;};
 // GPS input is stable for the selected job.
 },[gps]);
 useEffect(()=>{if(marker.current&&gps.length){const s=gps.reduce((a,b)=>Math.abs(b.t+offset-time)<Math.abs(a.t+offset-time)?b:a);marker.current.setLatLng([s.lat,s.lon]);}},[time,offset,gps]);
 return gps.length<2?<div className="empty">No GPS track supplied. No route or defect coordinates are invented.</div>:<><div className="map" ref={box}/><small>Measured vehicle GPS positions, not precise pothole coordinates. External map tiles require internet.</small></>;
}
