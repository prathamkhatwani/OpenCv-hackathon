import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def compute_severity(detection: Dict) -> Tuple[float, str]:
    """
    Compute severity score (0-100) and label for a detection.
    
    Pothole: 100 * (0.6 * min(area_frac/0.08, 1) + 0.4 * min(contrast/60, 1))
    Crack: 100 * (0.5 * min(length_px/400, 1) + 0.3 * min(mean_width_px/8, 1) + 0.2 * min(density/0.05, 1))
    
    Labels: <33 = Low, 33-66 = Medium, >66 = High
    """
    det_type = detection.get('type', '')
    measurements = detection.get('measurements', {})
    
    if det_type == 'pothole':
        area_frac = measurements.get('area_frac', 0)
        contrast = measurements.get('contrast', 0)
        score = 100 * (
            0.6 * min(area_frac / 0.08, 1.0) +
            0.4 * min(contrast / 60.0, 1.0)
        )
    elif det_type == 'crack':
        length_px = measurements.get('length_px', 0)
        mean_width = measurements.get('mean_width_px', 0)
        density = measurements.get('density', 0)
        score = 100 * (
            0.5 * min(length_px / 400.0, 1.0) +
            0.3 * min(mean_width / 8.0, 1.0) +
            0.2 * min(density / 0.05, 1.0)
        )
    else:
        score = 0.0
    
    score = max(0.0, min(100.0, score))
    
    if score > 66:
        label = 'High'
    elif score >= 33:
        label = 'Medium'
    else:
        label = 'Low'
    
    return round(score, 1), label
