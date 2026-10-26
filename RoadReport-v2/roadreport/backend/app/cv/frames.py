import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def build_roi_mask(height: int, width: int) -> np.ndarray:
    """Build a road ROI mask covering the road surface across full width,
    preventing edge potholes from being discarded.
    Returns a binary mask (uint8, 0 or 255)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    top_y = int(height * 0.35)
    bottom_y = height
    top_left = int(width * 0.05)
    top_right = int(width * 0.95)
    pts = np.array([[0, bottom_y], [width, bottom_y], 
                     [top_right, top_y], [top_left, top_y]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def build_glare_mask(frame_hsv: np.ndarray, v_threshold: int = 230) -> np.ndarray:
    """Mask very bright pixels (V > threshold) to exclude sky glare and white markings.
    Returns binary mask where bright pixels are 0 (excluded)."""
    v_channel = frame_hsv[:, :, 2]
    glare = (v_channel > v_threshold).astype(np.uint8) * 255
    return cv2.bitwise_not(glare)

def is_blurry(gray: np.ndarray, threshold: float = 60.0) -> bool:
    """Check if frame is too blurry using Laplacian variance."""
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def extract_frames(video_path: str, target_fps: float = 4.0, max_frames: int = 240,
                    target_width: int = 960) -> List[dict]:
    """
    Extract frames from video at target_fps, resized to target_width.
    Skips blurry frames.
    
    Returns list of dicts:
      {'frame': np.ndarray, 'gray': np.ndarray, 'hsv': np.ndarray,
       'index': int, 'timestamp_s': float, 'roi_mask': np.ndarray,
       'combined_mask': np.ndarray}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    
    # Calculate sample interval
    sample_interval = max(1, int(fps / target_fps))
    
    results = []
    roi_mask = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_interval != 0:
            frame_idx += 1
            continue
        
        # Resize to target width maintaining aspect ratio
        h, w = frame.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        frame = cv2.resize(frame, (target_width, new_h))
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Skip blurry frames
        if is_blurry(gray):
            frame_idx += 1
            continue
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Build masks (once, since all frames same size after resize)
        if roi_mask is None:
            roi_mask = build_roi_mask(new_h, target_width)
        
        glare_mask = build_glare_mask(hsv)
        combined_mask = cv2.bitwise_and(roi_mask, glare_mask)
        
        timestamp_s = frame_idx / fps
        
        results.append({
            'frame': frame,
            'gray': gray,
            'hsv': hsv,
            'index': frame_idx,
            'timestamp_s': round(timestamp_s, 2),
            'roi_mask': roi_mask,
            'combined_mask': combined_mask
        })
        
        if len(results) >= max_frames:
            break
        
        frame_idx += 1
    
    cap.release()
    logger.info(f"Extracted {len(results)} frames from {total_frames} total (sampled at ~{target_fps} fps)")
    return results
