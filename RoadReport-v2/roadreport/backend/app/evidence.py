import json
import cv2
import numpy as np
from typing import Dict, Any, Tuple

def _draw_finding_box(
    frame: np.ndarray,
    finding: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw bounding box and label on a copy of frame, returning (annotated_frame, crop)."""
    h_frame, w_frame = frame.shape[:2]

    # Parse bbox
    bbox_raw = finding.get("bbox_json")
    bbox = None
    if bbox_raw:
        try:
            bbox = json.loads(bbox_raw)
        except Exception:
            pass

    x, y, w, h = 0, 0, w_frame, h_frame
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x, y, w, h = bbox
    elif isinstance(bbox, dict):
        x = bbox.get("x", 0)
        y = bbox.get("y", 0)
        w = bbox.get("w", bbox.get("width", w_frame))
        h = bbox.get("h", bbox.get("height", h_frame))
    else:
        # Default box in lower center if bounding box was absent
        w = int(w_frame * 0.25)
        h = int(h_frame * 0.25)
        x = int((w_frame - w) / 2)
        y = int(h_frame * 0.6)

    # Clamping
    x = max(0, min(int(x), w_frame - 1))
    y = max(0, min(int(y), h_frame - 1))
    w = max(1, min(int(w), w_frame - x))
    h = max(1, min(int(h), h_frame - y))

    # Pick color by severity
    severity = str(finding.get("severity_label", "medium")).lower()
    if severity == "high":
        box_color = (0, 0, 255)  # Red (BGR)
    elif severity == "medium":
        box_color = (0, 140, 255)  # Orange
    else:
        box_color = (0, 215, 255)  # Amber / Yellow

    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

    # Label badge
    finding_type = str(finding.get("type", "damage")).capitalize()
    confidence = float(finding.get("confidence", 0.0))
    label = f"{finding_type} ({severity.capitalize()} {confidence:.2f})"
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    text_y = max(y - 6, label_h + 4)
    cv2.rectangle(
        annotated,
        (x, text_y - label_h - 2),
        (x + label_w + 4, text_y + baseline - 2),
        box_color,
        -1,
    )
    cv2.putText(
        annotated,
        label,
        (x + 2, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        crop = annotated

    return annotated, crop


def _generate_reinspection_zoom(raw_crop: np.ndarray, finding_type: str) -> np.ndarray:
    """Generate 3x bicubic upscaled, unsharp-masked crop with contour edges and contrast ring."""
    if raw_crop is None or raw_crop.size == 0 or min(raw_crop.shape[:2]) < 2:
        return raw_crop

    # 1. 3x bicubic upscale
    zoom = cv2.resize(raw_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # 2. Unsharp masking
    blurred = cv2.GaussianBlur(zoom, (0, 0), 1.0)
    sharpened = cv2.addWeighted(zoom, 1.5, blurred, -0.5, 0)

    # 3. Enhanced edge/contour overlay for OpenCV proof
    overlay = sharpened.copy()
    gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY) if sharpened.ndim == 3 else sharpened

    if str(finding_type).lower() == "pothole":
        zh, zw = gray.shape[:2]
        blk = max(3, (min(zh, zw) // 4) * 2 + 1)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blk, 8
        )
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            main_cnt = max(cnts, key=cv2.contourArea)
            # Inner cavity contour (Cyan BGR)
            cv2.drawContours(overlay, [main_cnt], -1, (255, 255, 0), 2)
            # Dilated ring boundary (Orange BGR)
            mask_in = np.zeros_like(gray)
            cv2.drawContours(mask_in, [main_cnt], -1, 255, -1)
            dilated = cv2.dilate(mask_in, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
            ring_cnts, _ = cv2.findContours(cv2.subtract(dilated, mask_in), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, ring_cnts, -1, (0, 165, 255), 1)
    else:
        # Crack fracture edges highlighted in bright yellow-orange
        edges = cv2.Canny(gray, 40, 120)
        overlay[edges > 0] = (0, 220, 255)

    return cv2.addWeighted(sharpened, 0.45, overlay, 0.55, 0)

