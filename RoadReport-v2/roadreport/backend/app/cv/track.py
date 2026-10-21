import cv2
import numpy as np
from typing import List, Dict, Optional
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


def _can_link(prev_det: Dict, curr_det: Dict, frame_h: int, frame_w: int) -> bool:
    """
    Check if two detections across consecutive sampled frames can be linked.
    On a moving camera, real defects drift downward and x stays roughly stable.
    """
    dx = abs(curr_det['center_x'] - prev_det['center_x'])
    dy = curr_det['center_y'] - prev_det['center_y']  # Should be positive (moving down)
    
    # x stability check
    if dx > 0.12 * frame_w:
        return False
    
    # Downward motion check
    if dy <= 0 or dy > 0.35 * frame_h:
        return False
    
    # Area ratio check (grows as it gets closer)
    prev_bbox = prev_det['bbox']
    curr_bbox = curr_det['bbox']
    prev_area = prev_bbox[2] * prev_bbox[3]
    curr_area = curr_bbox[2] * curr_bbox[3]
    ratio = curr_area / max(prev_area, 1)
    if ratio < 0.8 or ratio > 2.5:
        return False
    
    return True


def _template_match_score(prev_frame: np.ndarray, curr_frame: np.ndarray,
                           prev_bbox: tuple, curr_bbox: tuple) -> float:
    """
    Use template matching as tiebreaker between candidate links.
    Returns TM_CCOEFF_NORMED score.
    """
    px, py, pw, ph = prev_bbox
    # Extract crop from previous frame as template
    template = prev_frame[py:py+ph, px:px+pw]
    if template.size == 0 or min(template.shape[:2]) < 5:
        return 0.0
    
    # Search region in current frame (expanded area around current bbox)
    cx, cy, cw, ch = curr_bbox
    pad = max(cw, ch) // 2
    sy = max(0, cy - pad)
    sx = max(0, cx - pad)
    ey = min(curr_frame.shape[0], cy + ch + pad)
    ex = min(curr_frame.shape[1], cx + cw + pad)
    search_region = curr_frame[sy:ey, sx:ex]
    
    if (search_region.shape[0] < template.shape[0] or 
        search_region.shape[1] < template.shape[1]):
        # Resize template to fit
        template = cv2.resize(template, (min(template.shape[1], search_region.shape[1]),
                                          min(template.shape[0], search_region.shape[0])))
    
    if template.size == 0 or search_region.size == 0:
        return 0.0
    
    try:
        # Convert to grayscale for matching
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(search_region.shape) == 3:
            search_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)
    except cv2.error:
        return 0.0


def build_tracks(frame_detections: List[Dict], frame_data_list: List[dict]) -> List[Track]:
    """
    Build tracks from per-frame detections using greedy motion-aware linking.
    
    Args:
        frame_detections: list of lists, one per frame, each containing detection dicts
        frame_data_list: list of frame data dicts (with 'frame', 'index', etc.)
    
    Returns:
        List of Track objects. Tracks with >= 3 detections are 'persistent'.
    """
    if not frame_detections:
        return []
    
    h, w = frame_data_list[0]['frame'].shape[:2]
    
    active_tracks: List[Track] = []
    finished_tracks: List[Track] = []
    
    for frame_idx, (detections, fdata) in enumerate(zip(frame_detections, frame_data_list)):
        used_detections = set()
        
        # Try to extend existing tracks
        for track in active_tracks:
            best_match_idx = -1
            best_score = -1
            
            last_det = track.last_detection()
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_detections:
                    continue
                if det['type'] != track.type:
                    continue
                if not _can_link(last_det, det, h, w):
                    continue
                
                # Use template matching as tiebreaker
                score = _template_match_score(
                    track.last_frame_data()['frame'],
                    fdata['frame'],
                    last_det['bbox'],
                    det['bbox']
                )
                
                if score > best_score and score >= 0.5:
                    best_score = score
                    best_match_idx = det_idx
                elif score < 0.5 and best_match_idx == -1:
                    # Accept without template match if it's the only spatial candidate
                    best_match_idx = det_idx
                    best_score = score
            
            if best_match_idx >= 0:
                track.add(detections[best_match_idx], fdata)
                used_detections.add(best_match_idx)
        
        # Move tracks that weren't extended to finished
        still_active = []
        for track in active_tracks:
            if track.last_frame_data().get('index') == fdata.get('index'):
                still_active.append(track)
            else:
                finished_tracks.append(track)
        active_tracks = still_active
        
        # Start new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in used_detections:
                active_tracks.append(Track(det, fdata))
    
    # Finish remaining active tracks
    finished_tracks.extend(active_tracks)
    
    # Sort by best confidence descending
    finished_tracks.sort(key=lambda t: t.best_confidence, reverse=True)
    
    logger.info(f"Built {len(finished_tracks)} tracks, "
                f"{sum(1 for t in finished_tracks if t.n >= 3)} persistent (3+ frames)")
    
    return finished_tracks
