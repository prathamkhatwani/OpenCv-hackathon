import cv2
import numpy as np
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Track:
    """Represents a tracked detection across multiple frames."""
    
    def __init__(self, detection: Dict, frame_data: dict):
        self.detections: List[Dict] = [detection]
        self.frame_data_list: List[dict] = [frame_data]
        self.type: str = detection['type']
    
    @property
    def n(self) -> int:
        return len(self.detections)
    
    @property
    def best_detection(self) -> Dict:
        """Return detection with highest confidence."""
        return max(self.detections, key=lambda d: d['confidence'])
    
    @property
    def best_confidence(self) -> float:
        return self.best_detection['confidence']
    
    @property
    def best_frame_data(self) -> dict:
        """Return frame data corresponding to best detection."""
        best_idx = self.detections.index(self.best_detection)
        return self.frame_data_list[best_idx]
    
    @property
    def best_bbox(self) -> tuple:
        return self.best_detection['bbox']
    
    @property
    def best_frame_index(self) -> int:
        return self.best_frame_data['index']
    
    @property
    def best_timestamp(self) -> float:
        return self.best_frame_data['timestamp_s']
    
    def last_detection(self) -> Dict:
        return self.detections[-1]
    
    def last_frame_data(self) -> dict:
        return self.frame_data_list[-1]
    
    def add(self, detection: Dict, frame_data: dict):
        self.detections.append(detection)
        self.frame_data_list.append(frame_data)


def _iou(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-x)*max(0, bottom-y)
    return intersection/max(a[2]*a[3]+b[2]*b[3]-intersection, 1)


def _predict(track, timestamp):
    last = np.array(track.last_detection()['bbox'], dtype=float)
    if track.n < 2:
        return last
    dt = track.frame_data_list[-1]['timestamp_s']-track.frame_data_list[-2]['timestamp_s']
    if dt <= 0:
        return last
    velocity = (last-np.array(track.detections[-2]['bbox']))/dt
    prediction = last+velocity*min(timestamp-track.last_frame_data()['timestamp_s'], .75)
    prediction[2:] = np.maximum(prediction[2:], 5)
    return prediction


def build_tracks(frame_detections, frame_data_list, max_gap_s=1.0):
    """One-to-one motion association, tolerant of stopped cameras and missed frames.

    IDs are job-local. Re-identification beyond max_gap_s is deliberately not
    assumed: pixel coordinates alone cannot identify a road on a later visit.
    """
    from scipy.optimize import linear_sum_assignment
    if not frame_data_list:
        return []
    if frame_detections and isinstance(frame_detections[0], dict):
        by_index = {f['index']: [] for f in frame_data_list}
        for det in frame_detections:
            if det.get('frame_index') in by_index:
                by_index[det['frame_index']].append(det)
        frame_detections = [by_index[f['index']] for f in frame_data_list]
    if len(frame_detections) != len(frame_data_list):
        raise ValueError('Every sampled frame must have a detection list (possibly empty)')
    active, finished = [], []
    next_id = 1
    for detections, frame in zip(frame_detections, frame_data_list):
        timestamp = frame['timestamp_s']
        alive = []
        for track in active:
            if timestamp-track.last_frame_data()['timestamp_s'] <= max_gap_s:
                alive.append(track)
            else:
                finished.append(track)
        active = alive
        costs = np.full((len(active), len(detections)), 1e6)
        for i, track in enumerate(active):
            prediction = _predict(track, timestamp)
            for j, det in enumerate(detections):
                if det['type'] != track.type:
                    continue
                box = np.asarray(det['bbox'], dtype=float)
                ratio = box[2]*box[3]/max(prediction[2]*prediction[3], 1)
                distance = np.linalg.norm((box[:2]+box[2:]/2)-(prediction[:2]+prediction[2:]/2))
                scale = max(np.linalg.norm(prediction[2:]), np.linalg.norm(box[2:]), 20)
                overlap = _iou(prediction, box)
                # No one-way vertical-motion assumption: stops, turns and jitter are valid.
                if .3 <= ratio <= 3.3 and (overlap >= .08 or distance/scale <= .75):
                    costs[i,j] = 1-overlap + .4*distance/scale + .1*abs(np.log(ratio))
        used = set()
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for i,j in zip(rows, cols):
                if costs[i,j] < 1e6:
                    active[i].add(detections[j], frame)
                    used.add(j)
        for j, det in enumerate(detections):
            if j not in used:
                track = Track(det, frame)
                track.track_id = next_id
                next_id += 1
                active.append(track)
    return sorted(finished+active, key=lambda t: t.track_id)
