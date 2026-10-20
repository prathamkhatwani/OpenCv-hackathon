"""Native-frame camera motion proxy. No physical acceleration is inferred from pixels."""
import cv2
import numpy as np

class MotionStream:
    def __init__(self):
        self.previous=None; self.points=None; self.last_t=None; self.samples=[]

    def push(self, frame, t):
        image=cv2.resize(frame,(320,180))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        dy=rotation=quality=0.
        dt=t-self.last_t if self.last_t is not None else 0
        if self.previous is not None and self.points is not None and len(self.points)>=12 and dt>0:
            p,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None)
            if p is not None and status is not None:
                valid=status.reshape(-1)==1
                old=self.points[valid];new=p[valid]
                if len(old)>=12:
                    m,inliers=cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=2)
                    if m is not None and inliers is not None:
                        quality=float(inliers.mean())*min(1.,len(old)/40.)
                        # Affine displacement at image centre, normalized by height and time.
                        dy=float(m[1,0]*160+(m[1,1]-1)*90+m[1,2])/180/dt
                        rotation=float(np.arctan2(m[1,0],m[0,0]))/dt
        self.samples.append({'t':round(t,4),'vertical_velocity':round(dy,5),'rotation':round(rotation,5),'quality':round(quality,3)})
        # Upper/mid scene reduces near-road parallax. Moving objects remain a known limitation.
        mask=np.zeros_like(gray);mask[20:135,20:300]=255
        self.points=cv2.goodFeaturesToTrack(gray,120,.02,8,mask=mask)
        self.previous=gray;self.last_t=t

    def finish(self):
        if not self.samples:return []
        t=np.array([s['t'] for s in self.samples]);v=np.array([s['vertical_velocity'] for s in self.samples])
        residual=[]
        for i in range(len(v)):
            local=v[np.abs(t-t[i])<=.35]
            residual.append(abs(v[i]-np.median(local)))
        good=np.array([s['quality']>=.55 for s in self.samples])
        baseline=np.asarray(residual)[good]
        median=float(np.median(baseline)) if baseline.size else 0
        mad=float(np.median(abs(baseline-median))) if baseline.size else 0
        for s,x in zip(self.samples,residual):
            s['motion_score']=round(max(0.,(x-median)/max(.02,1.4826*mad)),3) if s['quality']>=.55 else 0.
        return self.samples


def event_evidence(contact, motion, telemetry):
    windows=contact.get('windows',[])
    empty={'camera_peak':0.,'camera_quality':0.,'imu_peak':0.,'imu_available':False,'event_t':None,'source':'unavailable','shock':False}
    if not windows:return empty
    candidates=[s for s in motion if any(a<=s['t']<=b for a,b in windows)]
    if candidates:
        peak=max(candidates,key=lambda s:s['motion_score'])
        empty.update(camera_peak=peak['motion_score'],camera_quality=peak['quality'],event_t=peak['t'],source='camera_proxy')
    if telemetry and telemetry['synchronization_uncertainty_s']<=.25:
        offset=telemetry['time_offset_s']
        covered=[]
        for a,b in windows:
            imu=[s for s in telemetry['samples'] if s.get('az') is not None and a<=s['t']+offset<=b]
            # Each axle interval needs independent coverage; the gap between axles is irrelevant.
            if (len(imu)>=4 and imu[0]['t']+offset-a<=.15 and b-imu[-1]['t']-offset<=.15
                    and max(y['t']-x['t'] for x,y in zip(imu,imu[1:]))<=.15):
                covered.extend(imu)
        if covered:
            peak=max(covered,key=lambda s:abs(s['az']))
            empty.update(imu_peak=abs(peak['az']),imu_available=True,event_t=peak['t']+offset,source='imu')
    empty['shock']=bool(empty['imu_peak']>=2. if empty['imu_available'] else empty['camera_peak']>=4 and empty['camera_quality']>=.65)
    return empty
