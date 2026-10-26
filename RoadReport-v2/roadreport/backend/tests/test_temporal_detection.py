import json
import numpy as np
from app.cv.track import build_tracks
from app.agent import run_agent


def frame(t):
    image = np.random.default_rng(1).integers(0, 255, (200, 300, 3), dtype=np.uint8)
    return {'frame': image, 'gray': image[:,:,0].copy(), 'index': int(t*30), 'timestamp_s': t}


def det(x=80, y=90, confidence=.8):
    return {'type':'pothole', 'bbox': (x,y,40,25), 'confidence':confidence, 'measurements':{}}


def test_stationary_and_upward_jitter_one_track():
    tracks = build_tracks([[det(y=y)] for y in [90,90,89,91,90]], [frame(i/4) for i in range(5)])
    assert len(tracks)==1 and tracks[0].n==5


def test_missed_detections_rejoin_same_id():
    tracks = build_tracks([[det(y=90)],[det(y=95)],[],[],[det(y=110)]], [frame(i/4) for i in range(5)])
    assert len(tracks)==1 and tracks[0].n==3


def test_two_nearby_potholes_stay_separate():
    tracks=build_tracks([[det(70,90+i*5), det(150,90+i*5)] for i in range(4)], [frame(i/4) for i in range(4)])
    assert len(tracks)==2
    assert all(t.n==4 for t in tracks)
    assert len({t.track_id for t in tracks})==2


def test_long_gap_does_not_merge_new_object():
    assert len(build_tracks([[det()],[det()]], [frame(0),frame(3)]))==2


def test_agent_serializes_track_without_inflating_confidence():
    frames=[frame(i/4) for i in range(3)]
    tracks=build_tracks([[det(confidence=.8)] for _ in frames],frames)
    stored=[]
    def save(f):
        stored.append(f)
        return 1
    run_agent(tracks,lambda *a,**k:None,save,'test',frames)
    assert len(stored)==1 and stored[0]['status']=='confirmed'
    m=json.loads(stored[0]['measurements_json'])
    assert m['track_id']==1 and len(m['track'])==3
    assert m['track'][0]['box']==[80/300,90/200,40/300,25/200]
    assert stored[0]['confidence']==.8


def test_transient_candidate_is_not_confirmed():
    frames=[frame(0)]
    saved=[]
    run_agent(build_tracks([[det(confidence=.99)]],frames),lambda *a,**k:None,lambda f:saved.append(f),'test',frames)
    assert saved==[]
