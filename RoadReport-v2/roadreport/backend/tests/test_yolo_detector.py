"""Tests for YOLOv8 OpenCV DNN detector."""
import numpy as np
import pytest
from pathlib import Path
from app.cv.detector import YOLODetector, get_detector, detect_all, CLASS_NAMES


def test_yolo_detector_loads():
    """Verify YOLODetector loads the ONNX model successfully with OpenCV DNN."""
    detector = get_detector()
    assert detector is not None
    info = detector.info()
    assert info['detector'] == 'yolov8n'
    assert info['backend'] == 'opencv_dnn'
    assert 'classes' in info['metadata']


def test_yolo_detector_inference_on_synthetic_frame():
    """Verify inference runs cleanly on an image and returns formatted detections."""
    detector = get_detector()
    # Create synthetic test frame
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    # Draw a simulated dark patch in road area
    import cv2
    cv2.circle(frame, (320, 240), 20, (30, 30, 30), -1)
    
    dets = detector.detect(frame)
    assert isinstance(dets, list)
    for d in dets:
        assert 'type' in d
        assert 'confidence' in d
        assert 'bbox' in d
        assert len(d['bbox']) == 4
        x, y, w, h = d['bbox']
        assert 0 <= x < 640
        assert 0 <= y < 360
        assert w > 0 and h > 0


def test_yolo_detector_roi_filtering():
    """Verify that detections outside ROI mask are correctly filtered."""
    detector = get_detector()
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    
    # ROI that only includes the bottom half
    roi = np.zeros((360, 640), dtype=np.uint8)
    roi[180:, :] = 255
    
    dets = detector.detect(frame, roi_mask=roi)
    for d in dets:
        assert d['center_y'] >= 180


def test_yolo_detector_classes():
    """Verify class mapping matches the RDD2022 dataset standard."""
    assert 'pothole' in CLASS_NAMES
    assert 'alligator_crack' in CLASS_NAMES
    assert 'longitudinal_crack' in CLASS_NAMES
    assert 'transverse_crack' in CLASS_NAMES
