"""Conservative straight-road contact estimates; unknown unless calibrated."""
import numpy as np


def project(h,x,y):
    p=np.asarray(h)@np.array([x,y,1.])
    if abs(p[2])<1e-6:return None
    q=p[:2]/p[2]
    return q if np.isfinite(q).all() else None


def estimate_contact(track,calibration,telemetry,duration):
    result={'state':'unknown','overlap':None,'windows':[],'reason':'No validated camera/vehicle calibration','corridors':[]}
    if not calibration or not calibration['straight_path_valid']:return result
    c=calibration;h=c['homography'];u=c['lateral_uncertainty_m']
    # Project the observed box width at its bottom edge. This is a conservative box proxy, not a tyre sensor.
    positions=[]
    for det,f in zip(track.detections,track.frame_data_list):
        ih,iw=f['gray'].shape;x,y,w,bh=det['bbox']
        left=project(h,x/iw,(y+bh)/ih);right=project(h,(x+w)/iw,(y+bh)/ih)
        if left is not None and right is not None:
            lateral=sorted([float(left[0]),float(right[0])]);distance=float((left[1]+right[1])/2)
            if 0<distance<60:positions.append((f['timestamp_s'],lateral,distance))
    if len(positions)<2:
        result['reason']='Insufficient valid ground-plane observations';return result
    centres=[-c['track_width_m']/2,c['track_width_m']/2]
    overlaps=[]
    for _,(lo,hi),_ in positions:
        overlaps.append(max(max(0., min(hi+u,centre+c['tyre_width_m']/2)-max(lo-u,centre-c['tyre_width_m']/2))/c['tyre_width_m'] for centre in centres))
    result['overlap']=round(float(np.clip(max(overlaps),0,1)),3)
    # Render normalized corridors; only visible calibrated ground-plane region.
    inv=np.linalg.inv(np.asarray(h))
    for centre in centres:
        points=[]
        for xx,yy in [(centre-c['tyre_width_m']/2,2),(centre-c['tyre_width_m']/2,30),(centre+c['tyre_width_m']/2,30),(centre+c['tyre_width_m']/2,2)]:
            p=project(inv,xx,yy)
            if p is not None:points.append([float(np.clip(p[0],-1,2)),float(np.clip(p[1],-1,2))])
        if len(points)==4:result['corridors'].append(points)
    if max(overlaps)==0:
        result.update(state='no_contact_expected',reason='Observed box stays outside both tyre corridors, including lateral uncertainty')
        return result
    # Require consensus across observations; curves/occlusion stay uncertain.
    if min(overlaps)<.25:
        result['reason']='Tyre overlap changes across observations';return result
    speeds=[]
    if telemetry:
        for t,_,_ in positions:
            nearby=[s['speed_mps'] for s in telemetry['samples'] if s.get('speed_mps') is not None and abs(s['t']+telemetry['time_offset_s']-t)<.5]
            speeds+=nearby
    if speeds:speed=float(np.median(speeds))
    else:
        tt=np.array([p[0] for p in positions]);yy=np.array([p[2] for p in positions])
        speed=-float(np.polyfit(tt,yy,1)[0]) if np.ptp(tt)>.2 else 0
    if not .5<speed<55:
        result['reason']='Cannot estimate a reliable approach speed';return result
    predictions=np.array([t+y/speed for t,_,y in positions])
    spread=float(np.ptp(predictions))
    if spread>.8:
        result['reason']='Inconsistent time-to-contact across track';return result
    t=float(np.median(predictions));unc=c['timing_uncertainty_s']+spread/2
    if telemetry:unc+=telemetry['synchronization_uncertainty_s']
    windows=[]
    for arrival in [t,t+c['wheelbase_m']/speed]:
        if arrival-unc>=0 and arrival+unc<=duration:windows.append([round(arrival-unc,3),round(arrival+unc,3)])
    if not windows:
        result['reason']='Predicted impact interval is outside recorded footage';return result
    result.update(state='contact_expected',reason='Consistent calibrated straight-path overlap; contact remains an estimate',windows=windows,speed_mps=round(speed,2),expected_t=round(t,3))
    return result
