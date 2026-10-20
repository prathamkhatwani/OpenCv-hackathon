"""Bounded video decoding with native motion and separate visual samples."""
import cv2
from app.cv.frames import build_roi_mask,build_glare_mask,is_blurry
from app.cv.motion import MotionStream


def inspect_video(path):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():raise ValueError('Cannot decode this video')
        fps=cap.get(cv2.CAP_PROP_FPS);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 1<=fps<=120 or count<=0 or not 32<=w<=4096 or not 32<=h<=4096:raise ValueError('Unsupported video metadata (1–120 fps, 32–4096 pixels)')
        duration=count/fps
        if duration>60.1 or count>7200:raise ValueError('Videos must be at most 60 seconds / 7200 frames')
        return {'fps':fps,'frame_count':count,'width':w,'height':h,'duration_s':duration}
    finally:cap.release()


def frame_record(frame,index,t,width=640):
    h,w=frame.shape[:2];scale=width/max(w,h);image=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))))
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    roi=build_roi_mask(*gray.shape)
    return {'frame':image,'gray':gray,'hsv':hsv,'index':index,'timestamp_s':round(t,4),'roi_mask':roi,'combined_mask':cv2.bitwise_and(roi,build_glare_mask(hsv))}


def decode_video(path,progress=lambda *args:None):
    meta=inspect_video(path);cap=cv2.VideoCapture(str(path));motion=MotionStream();frames=[];skipped=0;next_visual=0.;last_t=-1.;decoded=0
    try:
        for index in range(min(meta['frame_count'],7200)):
            ret,frame=cap.read()
            if not ret:break
            decoded+=1
            pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            t=pts if pts>last_t else index/meta['fps']
            if t<=last_t:raise ValueError('Non-monotonic video timestamps')
            if t>60.1:raise ValueError('Decoded video exceeds duration limit')
            last_t=t;motion.push(frame,t)
            if t>=next_visual and len(frames)<240:
                next_visual=t+.25
                f=frame_record(frame,index,t)
                if is_blurry(f['gray']):skipped+=1
                else:frames.append(f)
            if index%30==0:progress(index+1,meta['frame_count'])
        if not decoded:raise ValueError('No video frames decoded')
        meta.update(decoded_frames=decoded,visual_frames=len(frames),blurred_samples=skipped)
        meta['timestamp_mode']='decoder PTS, nominal FPS fallback for non-advancing PTS'
        return frames,motion.finish(),meta
    finally:cap.release()


def neighboring_frames(path,t):
    """Bounded active inspection: three original-resolution frames around candidate."""
    cap=cv2.VideoCapture(str(path));result=[]
    try:
        fps=cap.get(cv2.CAP_PROP_FPS)
        for time in [max(0,t-.12),t,min(60,t+.12)]:
            cap.set(cv2.CAP_PROP_POS_MSEC,time*1000);ok,frame=cap.read()
            if ok:
                actual=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                result.append((frame,actual if actual>=0 else time))
        return result
    finally:cap.release()
