"""Fit normalized image -> ground homography from measured correspondences."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from app.schemas import Calibration
p=argparse.ArgumentParser();p.add_argument('--points',required=True);p.add_argument('--output',required=True);a=p.parse_args()
d=json.loads(Path(a.points).read_text());im=np.asarray(d['image_points_normalized'],dtype=float);ground=np.asarray(d['ground_points_m'],dtype=float)
if im.shape!=ground.shape or len(im)<6 or im.shape[1]!=2:raise ValueError('Provide at least six measured point pairs, spread across the road plane')
if not np.isfinite(im).all() or not np.isfinite(ground).all() or np.any((im<0)|(im>1)):raise ValueError('Invalid points')
h,mask=cv2.findHomography(im,ground,cv2.RANSAC,.15)
if h is None:raise ValueError('Degenerate point layout')
projection=cv2.perspectiveTransform(im.reshape(-1,1,2),h).reshape(-1,2)
error=np.linalg.norm(projection-ground,axis=1)
print(json.dumps({'fit_median_error_m':float(np.median(error)),'fit_max_error_m':float(max(error)),'inliers':int(mask.sum()),'note':'Fit residual is not held-out calibration validation.'},indent=2))
cal=Calibration(homography=h.tolist(),camera_id=d['camera_id'],calibration_note=d['calibration_note'],track_width_m=d['track_width_m'],tyre_width_m=d['tyre_width_m'],wheelbase_m=d['wheelbase_m'],straight_path_valid=False)
Path(a.output).write_text(cal.model_dump_json(indent=2))
print('Saved with straight_path_valid=false. Validate separate measured points before enabling contact estimation.')
