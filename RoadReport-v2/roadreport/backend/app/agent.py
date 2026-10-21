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


def run_agent(
    tracks: list,
    add_step_fn: Callable,
    create_finding_fn: Callable,
    job_id: str,
    frame_data_list: list
) -> List[Dict[str, Any]]:
    """
    Run the agent decision loop on all tracks.

    Args:
        tracks: List of Track objects (or dicts) from build_tracks.
        add_step_fn: Function to log agent steps (job_id, finding_ref, step_no,
                     action, reason, conf_before, conf_after).
        create_finding_fn: Function to persist a finding in the database.
        job_id: Current job ID string.
        frame_data_list: List of all extracted frame data for potential re-extraction.

    Returns:
        List of finding dictionaries with status decisions and metadata.
    """
    findings: List[Dict[str, Any]] = []

    for track_idx, track in enumerate(tracks):
        c = float(track['best_confidence'] if isinstance(track, dict) else track.best_confidence)
        n = int(track['n'] if isinstance(track, dict) else track.n)
        track_type = str(track['type'] if isinstance(track, dict) else track.type)
        best_det = track['best_detection'] if isinstance(track, dict) else track.best_detection
        best_fdata = track['best_frame_data'] if isinstance(track, dict) else track.best_frame_data
        detections = track['detections'] if isinstance(track, dict) else track.detections
        track_fdata_list = track['frame_data_list'] if isinstance(track, dict) else track.frame_data_list

        finding_ref = f"track_{track_idx}"
        step_no = 0

        # Step 1: Initial detection log
        step_no += 1
        add_step_fn(
            job_id=job_id,
            finding_ref=finding_ref,
            step_no=step_no,
            action="detected",
            reason=f"Pass-1 confidence {c:.2f}, seen in {n} frames",
            confidence_before=0.0,
            confidence_after=c
        )

        if c >= HIGH and n >= 2:
            # High confidence + persistent → confirmed
            status = "confirmed"
            reason = f"High confidence ({c:.2f}) and persistent ({n} frames)"
            step_no += 1
            add_step_fn(job_id, finding_ref, step_no, "confirmed", reason, c, c)

        elif c >= MID:
            # Medium confidence → re-inspect with 3x zoom
            crop = pad_crop(best_fdata['frame'], best_det['bbox'], pad=0.25)
            if crop.size > 0:
                crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                crop = unsharp_mask(crop)
                c2 = run_detector_strict(crop, track_type)
            else:
                c2 = 0.0

            step_no += 1
            add_step_fn(
                job_id, finding_ref, step_no, "re-inspect",
                f"3x zoom re-inspection: confidence {c:.2f} → {c2:.2f}",
                c, c2
            )

            if c2 >= 0.70 and n >= 3:
                status = "confirmed"
                reason = f"Confirmed after re-inspection (c2={c2:.2f}, {n} frames)"
            elif c2 < LOW and n < 3:
                status = "dismissed"
                reason = f"Likely shadow or stain (c2={c2:.2f}, {n} frames)"
            else:
                status = "needs_review"
                reason = f"Evidence still mixed (c2={c2:.2f}, {n} frames)"

            step_no += 1
            add_step_fn(job_id, finding_ref, step_no, status, reason, c, c2)
            c = max(c, c2)  # Use best confidence

        else:
            # Low confidence → dismissed
            status = "dismissed"
            reason = f"Below minimum confidence ({c:.2f})"
            step_no += 1
            add_step_fn(job_id, finding_ref, step_no, "dismissed", reason, c, c)

        # Check for blurry best frame - try neighboring frames in track
        if status != "dismissed":
            try:
                from .cv.frames import is_blurry
            except (ImportError, AttributeError):
                def is_blurry(gray_img: np.ndarray, threshold: float = 80.0) -> bool:
                    return float(cv2.Laplacian(gray_img, cv2.CV_64F).var()) < threshold

            best_gray = best_fdata['gray']
            if is_blurry(best_gray, threshold=80.0):
                step_no += 1
                sharpest_idx = 0
                sharpest_var = float(cv2.Laplacian(best_gray, cv2.CV_64F).var())
                for i, (det_item, fdata_item) in enumerate(zip(detections, track_fdata_list)):
                    var = float(cv2.Laplacian(fdata_item['gray'], cv2.CV_64F).var())
                    if var > sharpest_var:
                        sharpest_var = var
                        sharpest_idx = i

                try:
                    current_idx = detections.index(best_det)
                except ValueError:
                    current_idx = -1

                if sharpest_idx != current_idx and sharpest_idx < len(detections):
                    add_step_fn(
                        job_id, finding_ref, step_no, "re-extract",
                        f"Best frame blurry (var={cv2.Laplacian(best_gray, cv2.CV_64F).var():.0f}), "
                        f"using sharper frame (var={sharpest_var:.0f})",
                        c, c
                    )
                    best_det = detections[sharpest_idx]
                    best_fdata = track_fdata_list[sharpest_idx]

        # Create finding if not dismissed
        if status != "dismissed":
            try:
                from .cv.severity import compute_severity
            except (ImportError, AttributeError):
                def compute_severity(detection: dict) -> Tuple[float, str]:
                    meas = detection.get('measurements', {})
                    area = float(meas.get('area_px', 500))
                    score = min(100.0, max(10.0, area / 20.0))
                    label = "High" if score >= 70.0 else ("Medium" if score >= 40.0 else "Low")
                    return score, label

            severity_score, severity_label = compute_severity(best_det)

            finding: Dict[str, Any] = {
                'job_id': job_id,
                'type': track_type,
                'severity_label': severity_label,
                'severity_score': severity_score,
                'confidence': round(c, 3),
                'status': status,
                'frame_index': best_fdata['index'],
                'timestamp_s': best_fdata['timestamp_s'],
                'bbox_json': json.dumps(list(best_det['bbox'])),
                'measurements_json': json.dumps(best_det.get('measurements', {})),
                'finding_ref': finding_ref,
                'frame': best_fdata['frame'],  # For image saving later
                'bbox': best_det['bbox'],
            }

            finding_id = create_finding_fn(finding)
            finding['id'] = finding_id
            findings.append(finding)

            logger.info(
                f"Finding {finding_ref}: {track_type} - {status} "
                f"(confidence={c:.2f}, severity={severity_label})"
            )

    confirmed = sum(1 for f in findings if f['status'] == 'confirmed')
    review = sum(1 for f in findings if f['status'] == 'needs_review')
    logger.info(
        f"Agent complete: {confirmed} confirmed, {review} needs_review, "
        f"{len(tracks) - len(findings)} dismissed"
    )

    return findings
