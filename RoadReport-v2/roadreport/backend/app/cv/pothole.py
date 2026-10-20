import cv2
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def _norm(value: float, low: float, high: float) -> float:
    """Normalize value to [0,1] range between low and high."""
    return max(0.0, min(1.0, (value - low) / max(high - low, 1e-6)))

def detect_potholes(gray: np.ndarray, combined_mask: np.ndarray, 
                     roi_mask: np.ndarray, frame: np.ndarray) -> List[Dict]:
    """
    Detect pothole candidates in a frame.
    
    Returns list of dicts with keys:
      bbox (x,y,w,h), area_frac, solidity, aspect, contrast, ring_edge,
      confidence, contour, center_x, center_y
    """
    h, w = gray.shape
    roi_area = cv2.countNonZero(roi_mask)
    if roi_area == 0:
        return []
    
    # CLAHE + bilateral filter
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=51, C=10
    )
    
    # Apply ROI mask
    thresh = cv2.bitwise_and(thresh, combined_mask)
    
    # Morphological operations
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    # Canny for edge density
    edges = cv2.Canny(smoothed, 50, 150)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / roi_area
        
        # Area fraction filter
        if area_frac < 0.003 or area_frac > 0.12:
            continue
        
        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1)
        if solidity < 0.80:
            continue
        
        # Aspect ratio
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / max(bh, 1)
        if aspect < 0.4 or aspect > 2.5:
            continue
        
        # Contrast: mean gray of dilated ring minus mean gray inside
        mask_inside = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_inside, [cnt], -1, 255, -1)
        
        dilated = cv2.dilate(mask_inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        ring = cv2.subtract(dilated, mask_inside)
        
        mean_inside = cv2.mean(gray, mask=mask_inside)[0]
        mean_ring = cv2.mean(gray, mask=ring)[0]
        contrast = mean_ring - mean_inside
        
        if contrast < 20:
            continue
        
        # Ring edge density
        ring_edge = cv2.mean(edges, mask=ring)[0] / 255.0
        
        # Reject if at very top of ROI (too far to judge)
        roi_top = int(h * 0.45)
        center_y = y + bh // 2
        if center_y < roi_top + int(h * 0.05):
            continue
        
        # Compute confidence
        contrast_norm = _norm(contrast, 20, 60)
        solidity_norm = _norm(solidity, 0.80, 1.0)
        ring_edge_norm = _norm(ring_edge, 0.05, 0.40)
        size_plausibility = _norm(area_frac, 0.003, 0.06)
        
        confidence = (
            0.35 * contrast_norm +
            0.25 * solidity_norm +
            0.20 * ring_edge_norm +
            0.20 * size_plausibility
        )
        confidence = max(0.0, min(1.0, confidence))
        
        center_x = x + bw // 2
        
        candidates.append({
            'type': 'pothole',
            'bbox': (x, y, bw, bh),
            'area_frac': round(area_frac, 5),
            'solidity': round(solidity, 3),
            'aspect': round(aspect, 2),
            'contrast': round(contrast, 1),
            'ring_edge': round(ring_edge, 3),
            'confidence': round(confidence, 3),
            'contour': cnt,
            'center_x': center_x,
            'center_y': center_y,
            'measurements': {
                'area_frac': round(area_frac, 5),
                'solidity': round(solidity, 3),
                'contrast': round(contrast, 1),
                'ring_edge': round(ring_edge, 3)
            }
        })
    
    logger.info(f"Found {len(candidates)} pothole candidates")
    return candidates
