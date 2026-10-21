"""OpenCV preprocessing + HOG + portable logistic crop classifier. Visual evidence only."""
import hashlib,json,math
from pathlib import Path
import cv2,numpy as np
from skimage.feature import hog
FEATURE_SIZE=324

def crop_features(crop):
    if crop is None or crop.size==0:raise ValueError('Empty visual crop')
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    gray=cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA)
    return hog(gray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True)

def context_crop(image,xyxy):
    x1,y1,x2,y2=xyxy;pad=max(2,round(max(x2-x1,y2-y1)*.12));h,w=image.shape[:2]
    return image[max(0,int(y1)-pad):min(h,int(y2)+pad),max(0,int(x1)-pad):min(w,int(x2)+pad)]

class VisualModel:
    def __init__(self,path):
        raw=Path(path).read_bytes()
        if len(raw)>1_000_000:raise ValueError('Oversized visual model')
        self.data=json.loads(raw)
        if self.data.get('schema')!='rdd2022-hog-logistic-v1':raise ValueError('Unsupported visual model')
        self.coef=np.asarray(self.data['coef']);self.mean=np.asarray(self.data['mean']);self.scale=np.asarray(self.data['scale'])
        size=FEATURE_SIZE
        if any(a.shape!=(size,) or not np.isfinite(a).all() for a in [self.coef,self.mean,self.scale]) or np.any(self.scale<=0):raise ValueError('Invalid visual model arrays')
        cal=self.data['calibration']
        if not all(math.isfinite(float(v)) for v in [self.data['intercept'],cal['coef'],cal['intercept']]):raise ValueError('Invalid visual calibration')
        self.sha256=hashlib.sha256(raw).hexdigest()
    def predict_crop(self,crop):
        x=(crop_features(crop)-self.mean)/self.scale
        score=float(x@self.coef+self.data['intercept']);cal=self.data['calibration'];score=cal['coef']*score+cal['intercept']
        return 1/(1+math.exp(-max(-40,min(40,score))))
    def info(self):
        return {'state':'trained_visual_pilot','dataset':'RDD2022','sha256':self.sha256,'metrics':self.data['test_metrics'],'note':'Crop classifier only; not a full detector or shock-association model. Display-only until field validation.'}
