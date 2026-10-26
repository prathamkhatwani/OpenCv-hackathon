import numpy as np
from app.cv.detector import YOLODetector


def test_nms_keeps_different_classes_but_suppresses_same_class_duplicates():
    detector = YOLODetector.__new__(YOLODetector)
    detector.conf_threshold=.45
    detector.nms_threshold=.45
    # Same box: two pothole proposals and one crack proposal.
    output=np.array([[[100,100,100],[100,100,100],[50,50,50],[50,50,50],
                      [.1,.1,.85],[.1,.1,.1],[.1,.1,.1],[.9,.8,.1]]],dtype=np.float32)
    detections=detector._postprocess(output,1,0,0,200,200)
    assert len(detections)==2
    assert {d['type'] for d in detections}=={'pothole','crack'}
    pothole=next(d for d in detections if d['type']=='pothole')
    assert pothole['confidence']==.9
    assert 'contrast' not in pothole['measurements']
