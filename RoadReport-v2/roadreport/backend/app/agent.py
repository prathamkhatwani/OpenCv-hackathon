"""Agent decision loop with step logging for Pothole & Road Damage Reporter."""

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Confidence thresholds
HIGH: float = 0.75
MID: float = 0.40
LOW: float = 0.30


def _norm(val: float, low: float, high: float) -> float:
    """Normalize a value within [low, high] to [0.0, 1.0]."""
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (val - low) / (high - low)))


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.5) -> np.ndarray:
    """
    Apply unsharp masking to enhance image details and edges.

    Args:
        image: Source image array (BGR or grayscale).
        sigma: Standard deviation for Gaussian blur.
        strength: Sharpening boost factor.

    Returns:
        Sharpened image as a numpy array.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


def pad_crop(frame: np.ndarray, bbox: Union[tuple, list], pad: float = 0.25) -> np.ndarray:
    """
    Extract a padded crop from frame around the specified bounding box.

    Args:
        frame: Source image frame.
        bbox: Bounding box tuple or list (x, y, w, h).
        pad: Padding percentage relative to box dimensions.

    Returns:
        Cropped numpy array.
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = (int(v) for v in bbox[:4])
    pad_x = int(bw * pad)
    pad_y = int(bh * pad)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3 if frame.ndim == 3 else 1), dtype=frame.dtype)
    return frame[y1:y2, x1:x2].copy()


def run_detector_strict(crop: np.ndarray, det_type: str) -> float:
    """
    Re-run detection on a zoomed/enhanced crop with tighter thresholds (~20% stricter).

    Args:
        crop: Padded image crop around detection.
        det_type: Type of detection ('pothole' or 'crack').

    Returns:
        Confidence score in range [0.0, 1.0].
    """
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape
    if h < 5 or w < 5:
        return 0.0

    if det_type == 'pothole':
        # CLAHE + bilateral filtering
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # Stricter adaptive threshold (C increased to 12)
        min_dim = min(h, w)
        if min_dim > 51:
            block_size = 51
        else:
            block_size = max(3, (min_dim // 2) * 2 - 1)

        thresh = cv2.adaptiveThreshold(
            smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, blockSize=block_size, C=12
        )

        # Morphology cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(
            thresh, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )

        # Find the largest dark region
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        total_area = h * w
        area_frac = area / max(total_area, 1)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1.0)

        # Contrast check between interior and surrounding ring
        mask_inside = np.zeros_like(gray)
        cv2.drawContours(mask_inside, [cnt], -1, 255, -1)
        dilated = cv2.dilate(mask_inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        ring = cv2.subtract(dilated, mask_inside)

        mean_inside = cv2.mean(gray, mask=mask_inside)[0]
        mean_ring = cv2.mean(gray, mask=ring)[0] if cv2.countNonZero(ring) > 0 else mean_inside
        contrast = mean_ring - mean_inside

        # Stricter thresholds: area_frac >= 0.004, solidity >= 0.85, contrast >= 24
        if area_frac < 0.004 or solidity < 0.85 or contrast < 24:
            return 0.1

        try:
            from .cv.pothole import _norm as pothole_norm
            norm_func = pothole_norm
        except (ImportError, AttributeError):
            norm_func = _norm

        confidence = (
            0.35 * norm_func(contrast, 24, 72) +
            0.25 * norm_func(solidity, 0.85, 1.0) +
            0.20 * norm_func(area_frac, 0.004, 0.06) +
            0.20 * min(1.0, area_frac / 0.01)
        )
        return max(0.0, min(1.0, float(confidence)))

    elif det_type == 'crack':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        smoothed = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        edges = cv2.Canny(smoothed, 60, 180)  # Stricter Canny
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        edge_density = cv2.countNonZero(edges) / max(h * w, 1)

        # Stricter: need more edge density
        if edge_density < 0.02:
            return 0.1

        confidence = min(1.0, edge_density / 0.15) * 0.7 + 0.3 * min(1.0, max(h, w) / 100.0)
        return max(0.0, min(1.0, float(confidence)))

    return 0.0



def inspect_track(track,video_path):
    """Bounded adaptive action: inspect three native-resolution neighboring frames."""
    from app.cv.video import neighboring_frames
    best=track.best_detection;f=track.best_frame_data
    h,w=f['gray'].shape;x,y,bw,bh=best['bbox']
    scores=[];observations=[]
    for frame,t in neighboring_frames(video_path,f['timestamp_s']):
        hh,ww=frame.shape[:2]
        box=[x*ww/w,y*hh/h,bw*ww/w,bh*hh/h]
        crop=pad_crop(frame,box)
        if crop.size:
            score=run_detector_strict(crop,track.type)
            scores.append(score);observations.append({'t':round(t,4),'score':round(score,4),'native_size':[ww,hh]})
    return float(np.median(scores)) if scores else track.best_confidence,observations


def run_agent(tracks,video_path,motion,options,meta,model=None):
    from app.cv.contact import estimate_contact
    from app.cv.motion import event_evidence
    from app.ml.features import extract
    from app.ml.decision import decide
    candidates=[]
    for track in tracks[:100]:
        contact=estimate_contact(track,options.get('calibration'),options.get('telemetry'),meta['duration_s']) if track.type=='pothole' else {'state':'unknown','overlap':None,'windows':[],'corridors':[],'reason':'Tyre-contact fusion applies to potholes'}
        impact=event_evidence(contact,motion,options.get('telemetry'))
        candidates.append((track,contact,impact))
    for i,(_,contact,impact) in enumerate(candidates):
        if impact['shock'] and impact['event_t'] is not None:
            impact['ambiguous']=any(j!=i and other['state']=='contact_expected' and any(a<=impact['event_t']<=b for a,b in other['windows']) for j,(_,other,_) in enumerate(candidates))
    results=[]
    for track,contact,impact in candidates:
        initial=track.best_confidence;reinspect=initial;reinspection_performed=False
        trace=[{'action':'detect_and_track','reason':f'{track.n} observed frames; classical visual score {initial:.3f}','before':None,'after':initial}]
        trace.append({'action':'estimate_contact','reason':contact['reason'],'before':initial,'after':initial})
        trace.append({'action':'check_motion','reason':f"Source: {impact['source']}; pulse: {impact['shock']}; ambiguous: {impact.get('ambiguous',False)}",'before':initial,'after':initial})
        if .3<=initial<.75 or impact.get('ambiguous'):
            reinspection_performed=True
            reinspect,samples=inspect_track(track,video_path)
            visual=.6*initial+.4*reinspect
            trace.append({'action':'inspect_neighboring_frames','reason':'Native-frame inspection: '+json.dumps(samples),'before':initial,'after':visual})
        else:visual=initial
        features=extract(track,reinspect,contact,impact)
        predictions=model.predict(features) if model and track.type=='pothole' else None
        status,evidence_class,reason=decide(visual,track.n,contact,impact,predictions,model.experimental if model else False)
        if track.type=='crack':evidence_class='visual_only' if status=='confirmed' else 'uncertain'
        trace.append({'action':('request_human_review' if status=='needs_review' else status),'reason':reason,'before':visual,'after':visual})
        results.append({'track':track,'status':status,'evidence_class':evidence_class,'reason':reason,'visual_score':visual,'initial_score':initial,'reinspection_score':reinspect,'reinspection_performed':reinspection_performed,'contact':contact,'impact':impact,'predictions':predictions,'features':features,'trace':trace})
    return results
