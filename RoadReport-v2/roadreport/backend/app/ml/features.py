import numpy as np
FEATURES=['visual_score','reinspection_score','track_seconds','track_observations','contact_overlap','contact_known','camera_peak','camera_quality','imu_peak','imu_available','speed_mps']
VISUAL_FEATURES=FEATURES[:4]

def extract(track,reinspection,contact,impact):
    times=[f['timestamp_s'] for f in track.frame_data_list]
    return dict(zip(FEATURES,[track.best_confidence,reinspection,max(times)-min(times),track.n,contact.get('overlap') or 0,float(contact['state']!='unknown'),impact['camera_peak'],impact['camera_quality'],impact['imu_peak'],float(impact['imu_available']),contact.get('speed_mps',0)]))

def vector(features,names=FEATURES):
    if any(k not in features for k in names):raise ValueError('Missing model features')
    x=np.array([features[k] for k in names],dtype=float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite model features')
    return x
