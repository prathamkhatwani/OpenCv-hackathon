#!/usr/bin/env python3
"""Evaluation script for the Road Damage Reporter.

Computes precision and recall at IoU >= 0.3 per type,
comparing first-pass detection vs. after agent re-inspection.

Usage:
    python eval.py --labels labels.csv --video path/to/video.mp4
    python eval.py --labels labels.csv --frames-dir path/to/frames/
"""

import argparse
import csv
import json
import os
import sys
from typing import List, Dict, Tuple, Any
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.cv.frames import extract_frames
from app.cv.pothole import detect_potholes
from app.cv.crack import detect_cracks
from app.cv.severity import compute_severity
from app.cv.track import build_tracks


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes (x, y, w, h)."""
    xa1, ya1, wa, ha = box_a
    xb1, yb1, wb, hb = box_b
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb2, yb2 = xb1 + wb, yb1 + hb
    
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area_a = wa * ha
    area_b = wb * hb
    union_area = area_a + area_b - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_labels(csv_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load ground-truth labels from CSV.
    Format: frame_index, type, x, y, w, h
    
    Returns dict mapping frame_index -> list of {type, bbox}
    """
    labels: Dict[int, List[Dict[str, Any]]] = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_idx = int(row['frame'])
            label = {
                'type': row['type'].strip().lower(),
                'bbox': (
                    int(row['x']), int(row['y']),
                    int(row['w']), int(row['h'])
                )
            }
            if frame_idx not in labels:
                labels[frame_idx] = []
            labels[frame_idx].append(label)
    return labels


def match_detections(
    detections: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.3
) -> Tuple[int, int, int]:
    """
    Match detections to ground truths using greedy IoU matching.
    
    Returns: (true_positives, false_positives, false_negatives)
    """
    if not ground_truths and not detections:
        return 0, 0, 0
    if not ground_truths:
        return 0, len(detections), 0
    if not detections:
        return 0, 0, len(ground_truths)
    
    matched_gt = set()
    tp = 0
    fp = 0
    
    # Sort detections by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda d: d.get('confidence', 0), reverse=True)
    
    for det in sorted_dets:
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in matched_gt:
                continue
            if det.get('type') != gt.get('type'):
                continue
            iou = compute_iou(det['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn


def evaluate(
    video_path: str,
    labels: Dict[int, List[Dict[str, Any]]],
    iou_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.
    
    Returns dict with per-type precision, recall, F1 for both
    first-pass and post-agent results.
    """
    print("Extracting frames...")
    frame_data_list = extract_frames(video_path)
    print(f"Extracted {len(frame_data_list)} frames")
    
    # --- First pass: raw detections ---
    all_first_pass = {'pothole': {'tp': 0, 'fp': 0, 'fn': 0},
                       'crack': {'tp': 0, 'fp': 0, 'fn': 0}}
    
    all_frame_detections = []
    
    for fdata in frame_data_list:
        frame_idx = fdata['index']
        gt = labels.get(frame_idx, [])
        
        # Run detectors
        pothole_dets = detect_potholes(
            fdata['gray'], fdata['combined_mask'],
            fdata['roi_mask'], fdata['frame']
        )
        crack_dets = detect_cracks(
            fdata['gray'], fdata['hsv'],
            fdata['combined_mask'], fdata['roi_mask']
        )
        
        all_dets = pothole_dets + crack_dets
        all_frame_detections.append(all_dets)
        
        # Evaluate first pass per frame
        for det_type in ['pothole', 'crack']:
            type_dets = [d for d in all_dets if d['type'] == det_type]
            type_gt = [g for g in gt if g['type'] == det_type]
            tp, fp, fn = match_detections(type_dets, type_gt, iou_threshold)
            all_first_pass[det_type]['tp'] += tp
            all_first_pass[det_type]['fp'] += fp
            all_first_pass[det_type]['fn'] += fn
    
    # --- With agent: track-level results ---
    print("Building tracks...")
    tracks = build_tracks(all_frame_detections, frame_data_list)
    
    # Simulate agent decisions (without DB)
    from app.agent import HIGH, MID, LOW, run_detector_strict, pad_crop, unsharp_mask
    import cv2
    
    agent_findings = []
    for track in tracks:
        c = track.best_confidence
        best_det = track.best_detection
        best_fdata = track.best_frame_data
        
        if c >= HIGH and track.n >= 2:
            status = 'confirmed'
        elif c >= MID:
            crop = pad_crop(best_fdata['frame'], best_det['bbox'], pad=0.25)
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            crop = unsharp_mask(crop)
            c2 = run_detector_strict(crop, track.type)
            if c2 >= 0.70 and track.n >= 3:
                status = 'confirmed'
            elif c2 < LOW and track.n < 3:
                status = 'dismissed'
            else:
                status = 'needs_review'
        else:
            status = 'dismissed'
        
        if status != 'dismissed':
            agent_findings.append({
                'type': track.type,
                'bbox': best_det['bbox'],
                'confidence': max(c, track.best_confidence),
                'frame_index': best_fdata['index'],
                'status': status
            })
    
    # Evaluate agent results per frame
    all_agent = {'pothole': {'tp': 0, 'fp': 0, 'fn': 0},
                  'crack': {'tp': 0, 'fp': 0, 'fn': 0}}
    
    # Group agent findings by frame
    agent_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    for f in agent_findings:
        fidx = f['frame_index']
        if fidx not in agent_by_frame:
            agent_by_frame[fidx] = []
        agent_by_frame[fidx].append(f)
    
    # Evaluate against all labeled frames
    for frame_idx, gt in labels.items():
        for det_type in ['pothole', 'crack']:
            type_dets = [d for d in agent_by_frame.get(frame_idx, []) if d['type'] == det_type]
            type_gt = [g for g in gt if g['type'] == det_type]
            tp, fp, fn = match_detections(type_dets, type_gt, iou_threshold)
            all_agent[det_type]['tp'] += tp
            all_agent[det_type]['fp'] += fp
            all_agent[det_type]['fn'] += fn
    
    # Compute metrics
    results: Dict[str, Any] = {'iou_threshold': iou_threshold, 'first_pass': {}, 'with_agent': {}}
    
    for phase, data in [('first_pass', all_first_pass), ('with_agent', all_agent)]:
        for det_type in ['pothole', 'crack']:
            tp = data[det_type]['tp']
            fp = data[det_type]['fp']
            fn = data[det_type]['fn']
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            results[phase][det_type] = {
                'tp': tp, 'fp': fp, 'fn': fn,
                'precision': round(precision, 3),
                'recall': round(recall, 3),
                'f1': round(f1, 3)
            }
    
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print(f"IoU Threshold: {results['iou_threshold']}")
    print("=" * 70)
    
    for phase_name, phase_key in [('First Pass (raw OpenCV)', 'first_pass'),
                                    ('With Agent Re-inspection', 'with_agent')]:
        print(f"\n--- {phase_name} ---")
        print(f"{'Type':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>6}")
        print("-" * 50)
        for det_type in ['pothole', 'crack']:
            m = results[phase_key][det_type]
            print(f"{det_type:<12} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4} "
                  f"{m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>6.3f}")
    
    # Show improvement
    print("\n--- Agent Improvement ---")
    for det_type in ['pothole', 'crack']:
        fp_before = results['first_pass'][det_type]['precision']
        fp_after = results['with_agent'][det_type]['precision']
        fr_before = results['first_pass'][det_type]['recall']
        fr_after = results['with_agent'][det_type]['recall']
        print(f"{det_type}: Precision {fp_before:.3f} → {fp_after:.3f} "
              f"({'+'if fp_after>=fp_before else ''}{fp_after-fp_before:.3f}), "
              f"Recall {fr_before:.3f} → {fr_after:.3f} "
              f"({'+'if fr_after>=fr_before else ''}{fr_after-fr_before:.3f})")


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate road damage detection pipeline')
    parser.add_argument('--labels', required=True, help='Path to labels CSV (frame,type,x,y,w,h)')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--iou', type=float, default=0.3, help='IoU threshold (default: 0.3)')
    parser.add_argument('--output', help='Save results as JSON to this path')
    args = parser.parse_args()
    
    if not os.path.exists(args.labels):
        print(f"Error: Labels file not found: {args.labels}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    labels = load_labels(args.labels)
    print(f"Loaded {sum(len(v) for v in labels.values())} labels across {len(labels)} frames")
    
    results = evaluate(args.video, labels, args.iou)
    print_results(results)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
