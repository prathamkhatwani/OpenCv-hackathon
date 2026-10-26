"""Job execution orchestrator for Pothole & Road Damage Reporter."""

import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.db import (
    create_finding,
    get_findings,
    get_job,
    update_finding,
    update_job,
)
from app.storage import download_file, upload_file

logger = logging.getLogger(__name__)


def _safe_json_dumps(obj: Any) -> str:
    def _default(o):
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(obj, default=_default)


def _resolve_pipeline_components() -> Tuple[Callable, Callable, Callable, Callable, Callable]:
    """Dynamically resolve and import CV and agent modules.
    
    Supports both package styles (app.cv / app.agent and cv / agent).
    """
    try:
        from app.cv.frames import extract_frames
    except ImportError:
        from cv.frames import extract_frames

    try:
        from app.cv.pothole import detect_potholes
    except ImportError:
        try:
            from cv.pothole import detect_potholes
        except ImportError:
            raise ImportError("Could not import 'detect_potholes' from app.cv.pothole or cv.pothole")

    try:
        from app.cv.crack import detect_cracks
    except ImportError:
        try:
            from cv.crack import detect_cracks
        except ImportError:
            raise ImportError("Could not import 'detect_cracks' from app.cv.crack or cv.crack")

    try:
        from app.cv.track import build_tracks
    except ImportError:
        try:
            from cv.track import build_tracks
        except ImportError:
            raise ImportError("Could not import 'build_tracks' from app.cv.track or cv.track")

    try:
        from app.agent import run_agent
    except ImportError:
        try:
            from agent import run_agent
        except ImportError:
            raise ImportError("Could not import 'run_agent' from app.agent or agent")

    return extract_frames, detect_potholes, detect_cracks, build_tracks, run_agent


def _invoke_detector(detector_func: Callable, frame_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Invoke a detector function flexibly based on its declared parameters."""
    sig = inspect.signature(detector_func)
    params = list(sig.parameters.keys())

    kwargs = {}
    if "frame" in params:
        kwargs["frame"] = frame_item["frame"]
    if "gray" in params:
        kwargs["gray"] = frame_item["gray"]
    if "hsv" in params:
        kwargs["hsv"] = frame_item["hsv"]
    if "mask" in params:
        kwargs["mask"] = frame_item.get("combined_mask", frame_item.get("roi_mask"))
    if "roi_mask" in params:
        kwargs["roi_mask"] = frame_item.get("roi_mask")
    if "combined_mask" in params:
        kwargs["combined_mask"] = frame_item.get("combined_mask")
    if "timestamp_s" in params:
        kwargs["timestamp_s"] = frame_item.get("timestamp_s", 0.0)
    if "frame_index" in params:
        kwargs["frame_index"] = frame_item.get("index", 0)
    elif "frame_idx" in params:
        kwargs["frame_idx"] = frame_item.get("index", 0)

    if len(kwargs) == len(params):
        raw_res = detector_func(**kwargs)
    elif len(params) == 1:
        param_name = params[0]
        if param_name in ("frame_item", "frame_dict", "data", "item"):
            raw_res = detector_func(frame_item)
        elif param_name in ("frame", "image", "img"):
            raw_res = detector_func(frame_item["frame"])
        else:
            try:
                raw_res = detector_func(frame_item)
            except (TypeError, KeyError):
                raw_res = detector_func(frame_item["frame"])
    else:
        try:
            raw_res = detector_func(frame_item)
        except TypeError:
            raw_res = detector_func(frame_item["frame"])

    if raw_res is None:
        return []
    results = raw_res if isinstance(raw_res, list) else [raw_res]

    # Ensure frame_index and timestamp_s are populated
    for det in results:
        if isinstance(det, dict):
            if "frame_index" not in det:
                det["frame_index"] = frame_item.get("index", 0)
            if "timestamp_s" not in det:
                det["timestamp_s"] = frame_item.get("timestamp_s", 0.0)

    return results


def _invoke_tracker(
    tracker_func: Callable,
    frames: List[Dict[str, Any]],
    detections: List[Dict[str, Any]],
) -> List[Any]:
    """Invoke multi-frame tracking function with proper grouping."""
    # Build per-frame grouped detections
    per_frame: List[List[Dict[str, Any]]] = [[] for _ in range(len(frames))]
    idx_map = {fd.get('index', i): i for i, fd in enumerate(frames)}
    for det in detections:
        f_idx = det.get('frame_index')
        slot = idx_map.get(f_idx, 0 if f_idx is None else min(max(0, f_idx), len(per_frame) - 1))
        per_frame[slot].append(det)

    try:
        raw_tracks = tracker_func(per_frame, frames)
    except Exception:
        try:
            raw_tracks = tracker_func(frames, per_frame)
        except Exception:
            try:
                raw_tracks = tracker_func(detections, frames)
            except Exception:
                raw_tracks = tracker_func(detections)

    if raw_tracks is None:
        return []
    return raw_tracks if isinstance(raw_tracks, list) else [raw_tracks]


def _deduplicate_tracks(
    tracks: List[Any],
    max_gap_s: float = 2.5,
    min_iou: float = 0.15,
    max_dist: float = 90.0,
) -> List[Any]:
    """Merge fragmented tracks belonging to the same physical road defect."""
    if not tracks:
        return []

    def _track_iou(a, b):
        x, y = max(a[0], b[0]), max(a[1], b[1])
        right, bottom = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
        inter = max(0, right - x) * max(0, bottom - y)
        return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1)

    def get_start_t(t):
        if hasattr(t, "frame_data_list") and t.frame_data_list:
            return t.frame_data_list[0].get("timestamp_s", 0.0)
        return 0.0

    sorted_tracks = sorted(tracks, key=get_start_t)
    merged: List[Any] = []

    for t in sorted_tracks:
        if not hasattr(t, "detections") or not t.detections:
            continue
        matched = False
        t_start = t.frame_data_list[0]["timestamp_s"]
        t_first_box = t.detections[0]["bbox"]
        t_first_c = (t_first_box[0] + t_first_box[2] / 2, t_first_box[1] + t_first_box[3] / 2)

        for m in merged:
            if m.type != t.type:
                continue
            m_end = m.frame_data_list[-1]["timestamp_s"]
            dt = t_start - m_end
            if dt < -0.1:  # Temporal overlap
                iou_best = _track_iou(m.best_bbox, t.best_bbox)
                c_m = (m.best_bbox[0] + m.best_bbox[2] / 2, m.best_bbox[1] + m.best_bbox[3] / 2)
                c_t = (t.best_bbox[0] + t.best_bbox[2] / 2, t.best_bbox[1] + t.best_bbox[3] / 2)
                dist_best = float(np.hypot(c_m[0] - c_t[0], c_m[1] - c_t[1]))
                if iou_best >= min_iou or dist_best <= max_dist:
                    existing_indices = {fd["index"] for fd in m.frame_data_list}
                    for d, fd in zip(t.detections, t.frame_data_list):
                        if fd["index"] not in existing_indices:
                            m.add(d, fd)
                            existing_indices.add(fd["index"])
                    matched = True
                    break
            elif 0 <= dt <= max_gap_s:  # Forward travel sequence
                m_last_box = m.detections[-1]["bbox"]
                m_last_c = (m_last_box[0] + m_last_box[2] / 2, m_last_box[1] + m_last_box[3] / 2)
                iou_edge = _track_iou(m_last_box, t_first_box)
                dist_edge = float(np.hypot(m_last_c[0] - t_first_c[0], m_last_c[1] - t_first_c[1]))
                if iou_edge >= min_iou or dist_edge <= max_dist:
                    for d, fd in zip(t.detections, t.frame_data_list):
                        m.add(d, fd)
                    matched = True
                    break
        if not matched:
            merged.append(t)

    for idx, m in enumerate(merged):
        combined = list(zip(m.detections, m.frame_data_list))
        combined.sort(key=lambda pair: pair[1]["timestamp_s"])
        m.detections = [p[0] for p in combined]
        m.frame_data_list = [p[1] for p in combined]
        m.track_id = idx + 1

    return merged


def _invoke_agent(
    agent_func: Callable,
    job_id: str,
    tracks: List[Any],
    frames: List[Dict[str, Any]],
) -> Any:
    """Invoke agent verification loop with compatible database helpers."""
    from app.db import add_agent_step, create_finding

    def _step_fn(*args, **kwargs):
        if kwargs:
            j_id = kwargs.get("job_id", job_id)
            f_ref = kwargs.get("finding_ref")
            s_no = kwargs.get("step_no", 1)
            act = kwargs.get("action", "")
            rsn = kwargs.get("reason", "")
            c_bef = kwargs.get("confidence_before", kwargs.get("conf_before"))
            c_aft = kwargs.get("confidence_after", kwargs.get("conf_after"))
            return add_agent_step(
                job_id=j_id,
                step_no=int(s_no) if str(s_no).isdigit() else 1,
                action=str(act),
                reason=str(rsn),
                finding_ref=str(f_ref) if f_ref else None,
                confidence_before=float(c_bef) if c_bef is not None else None,
                confidence_after=float(c_aft) if c_aft is not None else None,
            )
        if len(args) >= 5 and isinstance(args[1], str) and (isinstance(args[2], int) or str(args[2]).isdigit()):
            # Called as (job_id, finding_ref, step_no, action, reason, c_before, c_after)
            j_id, f_ref, s_no, act, rsn = args[0], args[1], args[2], args[3], args[4]
            c_bef = args[5] if len(args) > 5 else None
            c_aft = args[6] if len(args) > 6 else None
            return add_agent_step(
                job_id=j_id,
                step_no=int(s_no),
                action=str(act),
                reason=str(rsn),
                finding_ref=str(f_ref),
                confidence_before=float(c_bef) if c_bef is not None else None,
                confidence_after=float(c_aft) if c_aft is not None else None,
            )
        return add_agent_step(*args)

    def _create_finding_fn(finding_data):
        if isinstance(finding_data, dict):
            finding_data.setdefault("job_id", job_id)
            res = create_finding(finding_data)
            return res.get("id") if isinstance(res, dict) else res
        return create_finding(finding_data)

    sig = inspect.signature(agent_func)
    params = list(sig.parameters.keys())

    kwargs = {}
    if "tracks" in params:
        kwargs["tracks"] = tracks
    if "job_id" in params:
        kwargs["job_id"] = job_id
    if "frames" in params:
        kwargs["frames"] = frames
    elif "frame_data_list" in params:
        kwargs["frame_data_list"] = frames
    if "add_step_fn" in params or any("step" in p for p in params):
        step_p = next(p for p in params if "step" in p)
        kwargs[step_p] = _step_fn
    if "create_finding_fn" in params or any("finding" in p for p in params if p not in kwargs):
        find_p = next(p for p in params if "finding" in p and p not in kwargs)
        kwargs[find_p] = _create_finding_fn

    if len(kwargs) == len(params):
        return agent_func(**kwargs)

    try:
        return agent_func(tracks, _step_fn, _create_finding_fn, job_id, frames)
    except Exception:
        try:
            return agent_func(job_id, tracks, frames)
        except Exception:
            return agent_func(job_id, tracks)


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


def run_pipeline(job_id: str) -> None:
    """Execute the end-to-end processing pipeline for a job.
    
    1. Update status to 'processing'
    2. Download video from S3 to temp dir
    3. Extract frames
    4. Update frames_total in DB
    5. Run pothole detection
    6. Run crack detection
    7. Update frames_done for progress tracking
    8. Run multi-frame tracking
    9. Run agent loop
    10. Annotate and crop confirmed/needs_review findings and upload to S3
    11. Build summary_json
    12. Update status to 'done'
    """
    logger.info("Starting processing pipeline for job: %s", job_id)
    temp_dir = tempfile.mkdtemp(prefix=f"roadreport_{job_id}_")

    try:
        # Step 1: Update job status to 'processing'
        job = update_job(job_id, status="processing")
        if not job:
            raise ValueError(f"Job '{job_id}' not found in database.")

        video_key = job.get("video_key")
        if not video_key:
            raise ValueError(f"Job '{job_id}' is missing video_key.")

        # Step 2: Download video from S3 to temp directory
        ext = os.path.splitext(video_key)[1] or ".mp4"
        local_video_path = os.path.join(temp_dir, f"input_video{ext}")
        logger.info("Downloading video %s to %s", video_key, local_video_path)
        download_file(video_key, local_video_path)

        # Load CV and Agent components
        (
            extract_frames,
            detect_potholes,
            detect_cracks,
            build_tracks,
            run_agent,
        ) = _resolve_pipeline_components()

        # Step 3: Frame extraction
        logger.info("Extracting frames from video: %s", local_video_path)
        frames = extract_frames(local_video_path)
        frames_count = len(frames)
        logger.info("Extracted %d frames for job %s", frames_count, job_id)

        # Step 4: Update frames_total in DB
        update_job(job_id, frames_total=frames_count, frames_used=frames_count, frames_done=0)

        # Supervised detector is mandatory: never silently substitute dark patches.
        from app.cv.detector import get_detector
        detector = get_detector()
        all_detections: List[Dict[str, Any]] = []
        for i, frame_item in enumerate(frames):
            for det in detector.detect(frame_item['frame'], frame_item['roi_mask']):
                det['frame_index'] = frame_item['index']
                det['timestamp_s'] = frame_item['timestamp_s']
                all_detections.append(det)
            update_job(job_id, frames_done=i + 1)

        logger.info("Completed detection across %d frames: %d raw detections", frames_count, len(all_detections))

        # Step 8: Multi-frame tracking
        tracks = _invoke_tracker(build_tracks, frames, all_detections)
        logger.info("Multi-frame tracking produced %d raw tracks", len(tracks))
        tracks = _deduplicate_tracks(tracks)
        logger.info("After spatial-temporal deduplication: %d unique defect tracks", len(tracks))

        # Step 9: Agent loop
        agent_result = _invoke_agent(run_agent, job_id, tracks, frames)

        # An empty agent result is valid. Never resurrect dismissed tracks.

        # Refresh findings list
        findings = get_findings(job_id)
        logger.info("Persisted findings count: %d", len(findings))

        # Index frames by frame index for quick lookup
        frames_by_index = {f["index"]: f for f in frames}
        frame_indices = sorted(frames_by_index.keys())

        # Step 10: For each confirmed/needs_review finding, draw bbox, save annotated and crop, upload to S3
        for finding in findings:
            status = finding.get("status", "confirmed")
            if status not in ("confirmed", "needs_review"):
                continue

            frame_entry = None
            f_idx = finding.get("frame_index")
            if f_idx is not None and f_idx in frames_by_index:
                frame_entry = frames_by_index[f_idx]
            elif frames:
                if f_idx is not None and frame_indices:
                    closest_idx = min(frame_indices, key=lambda idx: abs(idx - f_idx))
                    frame_entry = frames_by_index[closest_idx]
                else:
                    frame_entry = frames[0]

            if frame_entry is None:
                continue

            annotated, crop = _draw_finding_box(frame_entry["frame"], finding)

            annotated_filename = f"finding_{finding['id']}_annotated.jpg"
            crop_filename = f"finding_{finding['id']}_crop.jpg"
            zoom_filename = f"finding_{finding['id']}_zoom.jpg"
            annotated_path = os.path.join(temp_dir, annotated_filename)
            crop_path = os.path.join(temp_dir, crop_filename)
            zoom_path = os.path.join(temp_dir, zoom_filename)

            zoom_crop = _generate_reinspection_zoom(crop, finding.get("type", "pothole"))

            cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(zoom_path, zoom_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

            image_key = f"jobs/{job_id}/findings/{annotated_filename}"
            crop_key = f"jobs/{job_id}/findings/{crop_filename}"
            zoom_key = f"jobs/{job_id}/findings/{zoom_filename}"

            upload_file(annotated_path, image_key)
            upload_file(crop_path, crop_key)
            upload_file(zoom_path, zoom_key)

            # Update measurements_json to include zoom_key
            current_meas = {}
            if finding.get("measurements_json"):
                try:
                    current_meas = json.loads(finding["measurements_json"])
                except Exception:
                    current_meas = {}
            current_meas["zoom_key"] = zoom_key
            new_meas_json = _safe_json_dumps(current_meas)

            update_finding(finding["id"], image_key=image_key, crop_key=crop_key, measurements_json=new_meas_json)

        # Step 11: Build summary_json with counts by type and severity
        final_findings = get_findings(job_id)
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}

        for f in final_findings:
            t = f.get("type", "unknown")
            s = f.get("severity_label", "unknown")
            st = f.get("status", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
            by_severity[s] = by_severity.get(s, 0) + 1
            by_status[st] = by_status.get(st, 0) + 1

        capture = cv2.VideoCapture(local_video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        video_meta = {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      "fps": fps, "duration_s": capture.get(cv2.CAP_PROP_FRAME_COUNT)/fps}
        capture.release()
        summary = {
            "video": video_meta,
            "detector": detector.info(),
            "total_findings": len(final_findings),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_status": by_status,
        }
        summary_json = _safe_json_dumps(summary)

        # Step 12: Update job status to 'done'
        update_job(job_id, status="done", summary_json=summary_json)
        logger.info("Pipeline completed successfully for job: %s", job_id)

    except Exception as exc:
        logger.exception("Pipeline failed for job %s: %s", job_id, exc)
        update_job(job_id, status="failed", error_message=str(exc))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
