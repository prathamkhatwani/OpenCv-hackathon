import cv2
import numpy as np
from typing import List, Dict
import logging

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

logger = logging.getLogger(__name__)

def _norm(value: float, low: float, high: float) -> float:
    """Normalize value to [0,1] range."""
    return max(0.0, min(1.0, (value - low) / max(high - low, 1e-6)))

def _remove_lane_markings(edges: np.ndarray, hsv: np.ndarray) -> np.ndarray:
    """Remove lane markings (white and yellow) from edge map."""
    h_ch, s_ch, v_ch = cv2.split(hsv)
    
    # White markings: low saturation, high value
    white_mask = ((s_ch < 40) & (v_ch > 180)).astype(np.uint8) * 255
    
    # Yellow markings: hue 15-35
    yellow_mask = ((h_ch >= 15) & (h_ch <= 35) & (s_ch > 50)).astype(np.uint8) * 255
    
    marking_mask = cv2.bitwise_or(white_mask, yellow_mask)
    marking_mask = cv2.dilate(marking_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    
    # Subtract markings from edges
    cleaned = cv2.subtract(edges, marking_mask)
    return cleaned

def detect_cracks(gray: np.ndarray, hsv: np.ndarray, combined_mask: np.ndarray,
                   roi_mask: np.ndarray) -> List[Dict]:
    """
    Detect crack candidates in a frame.
    
    Returns list of dicts with keys:
      bbox (x,y,w,h), length_px, elongation, mean_width_px, density,
      continuity, confidence, center_x, center_y
    """
    h, w = gray.shape
    roi_area = cv2.countNonZero(roi_mask)
    if roi_area == 0:
        return []
    
    # CLAHE + bilateral filter (same as pothole)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Canny edge detection
    edges = cv2.Canny(smoothed, 50, 150)
    
    # Remove lane markings
    edges = _remove_lane_markings(edges, hsv)
    
    # Apply ROI mask
    edges = cv2.bitwise_and(edges, combined_mask)
    
    # Morphological close to join fragments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, connectivity=8)
    
    candidates = []
    
    for i in range(1, num_labels):  # Skip background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        comp_area = stats[i, cv2.CC_STAT_AREA]
        
        # Skeletonize the component
        comp_mask = (labels == i).astype(np.uint8)
        
        if skeletonize is not None:
            skeleton = skeletonize(comp_mask > 0).astype(np.uint8)
            length_px = int(np.sum(skeleton))
        else:
            # Fallback: thin using OpenCV morphology
            thin = cv2.ximgproc.thinning(comp_mask * 255) if hasattr(cv2, 'ximgproc') else comp_mask * 255
            length_px = cv2.countNonZero(thin)
        
        # Filter by length
        if length_px < 60:
            continue
        
        # Elongation
        elongation = max(bw, bh) / max(1, min(bw, bh))
        if elongation < 3:
            continue
        
        # Mean width
        mean_width_px = comp_area / max(length_px, 1)
        
        # Density
        density = comp_area / max(roi_area, 1)
        
        # Continuity: skeleton length / bounding box diagonal
        diag = np.sqrt(bw**2 + bh**2)
        continuity = min(1.0, length_px / max(diag, 1))
        
        # Confidence
        confidence = (
            0.4 * _norm(length_px, 60, 400) +
            0.3 * _norm(elongation, 3, 15) +
            0.3 * continuity
        )
        confidence = max(0.0, min(1.0, confidence))
        
        center_x = int(centroids[i][0])
        center_y = int(centroids[i][1])
        
        candidates.append({
            'type': 'crack',
            'bbox': (x, y, bw, bh),
            'length_px': length_px,
            'elongation': round(elongation, 2),
            'mean_width_px': round(mean_width_px, 2),
            'density': round(density, 6),
            'continuity': round(continuity, 3),
            'confidence': round(confidence, 3),
            'center_x': center_x,
            'center_y': center_y,
            'measurements': {
                'length_px': length_px,
                'elongation': round(elongation, 2),
                'mean_width_px': round(mean_width_px, 2),
                'density': round(density, 6),
                'continuity': round(continuity, 3)
            }
        })
    
    logger.info(f"Found {len(candidates)} crack candidates")
    return candidates
