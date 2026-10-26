"""YOLOv8 pothole/crack detector using OpenCV DNN inference.

This replaces the classical adaptive-threshold detector with a trained ONNX model
loaded through cv2.dnn.readNetFromONNX(). Pure OpenCV inference — no PyTorch required.

Class mapping (RDD2022):
  0: longitudinal_crack (D00)
  1: transverse_crack (D10)
  2: alligator_crack (D20)
  3: pothole (D40)
"""
import cv2
import numpy as np
import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# RDD2022 class mapping
CLASS_NAMES = ['longitudinal_crack', 'transverse_crack', 'alligator_crack', 'pothole']
# Map RDD2022 classes to our two types
CLASS_TO_TYPE = {
    0: 'crack',      # D00 longitudinal crack
    1: 'crack',      # D10 transverse crack
    2: 'crack',      # D20 alligator crack
    3: 'pothole',    # D40 pothole
}
CLASS_TO_SUBTYPE = {
    0: 'longitudinal_crack',
    1: 'transverse_crack',
    2: 'alligator_crack',
    3: 'pothole',
}


class YOLODetector:
    """YOLOv8 detector using OpenCV DNN backend."""
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.20,
                 nms_threshold: float = 0.45, input_size: int = 640):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to ONNX model. If None, uses bundled model.
            conf_threshold: Minimum confidence to keep a detection.
            nms_threshold: NMS IoU threshold.
            input_size: Model input size (square).
        """
        if model_path is None:
            model_path = str(Path(__file__).parent.parent / 'models' / 'pothole_yolov8n.onnx')
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.net = None
        self._inference_lock = threading.Lock()
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model using OpenCV DNN."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"YOLO ONNX model not found: {self.model_path}\n"
                f"Train the model first: python -m tools.train_yolo --data data/pothole_dataset/data.yaml"
            )
        
        logger.info(f"Loading YOLO model from {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        
        # Prefer CPU backend for portability
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Load metadata if available
        meta_path = Path(self.model_path).with_suffix('.json')
        if meta_path.exists():
            self.metadata = json.loads(meta_path.read_text())
            logger.info(f"Model metadata: {self.metadata.get('dataset', 'unknown')}, "
                        f"test mAP50={self.metadata.get('test_metrics', {}).get('mAP50', 'N/A')}")
        else:
            self.metadata = {}
        
        classes = self.metadata.get('classes')
        if classes:
            ordered = [classes.get(str(i), classes.get(i)) for i in range(len(classes))] if isinstance(classes, dict) else classes
            if ordered != CLASS_NAMES:
                raise ValueError('Model class order differs from RDD2022; refusing incorrect labels')
        logger.info("YOLO detector loaded successfully")
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Preprocess frame for YOLO inference with letterboxing.
        
        Returns:
            blob: Preprocessed input blob
            ratio: Scale ratio
            dw: Horizontal padding
            dh: Vertical padding
        """
        h, w = frame.shape[:2]
        
        # Calculate scale to fit in input_size while maintaining aspect ratio
        ratio = min(self.input_size / h, self.input_size / w)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterboxed image (pad with gray)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        dw = (self.input_size - new_w) // 2
        dh = (self.input_size - new_h) // 2
        canvas[dh:dh + new_h, dw:dw + new_w] = resized
        
        # Create blob: BGR -> RGB, normalize to [0,1], CHW format
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (self.input_size, self.input_size),
                                      swapRB=True, crop=False)
        
        return blob, ratio, dw, dh
    
    def _postprocess(self, output: np.ndarray, ratio: float, dw: int, dh: int,
                     orig_h: int, orig_w: int) -> List[Dict]:
        """
        Post-process YOLOv8 output.
        
        YOLOv8 output shape: [1, num_classes+4, num_detections]
        First 4 values per detection: cx, cy, w, h (in input_size coordinates)
        Remaining values: class confidences
        """
        # Transpose from [1, 4+nc, N] to [N, 4+nc]
        predictions = output[0].T
        
        num_classes = predictions.shape[1] - 4
        
        # Extract boxes and scores
        boxes = predictions[:, :4]  # cx, cy, w, h
        scores = predictions[:, 4:]  # class scores
        
        # Get max class score and class id for each detection
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Filter by confidence
        mask = max_scores >= self.conf_threshold
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from cx,cy,w,h to x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Remove letterbox padding and scale back to original image
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio
        
        # Clip to image bounds
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # Convert to x,y,w,h format for NMS
        widths = x2 - x1
        heights = y2 - y1
        
        # OpenCV NMS
        boxes_for_nms = np.column_stack([x1, y1, widths, heights]).tolist()
        indices = []
        for cls in np.unique(class_ids):
            candidates = np.where(class_ids == cls)[0]
            kept = cv2.dnn.NMSBoxes([boxes_for_nms[k] for k in candidates],
                                   max_scores[candidates].tolist(),
                                   self.conf_threshold, self.nms_threshold)
            indices.extend(candidates[np.asarray(kept, dtype=int).reshape(-1)].tolist())
        
        detections = []
        if len(indices) > 0:
            for i in indices:
                bx = int(round(x1[i]))
                by = int(round(y1[i]))
                bw = int(round(widths[i]))
                bh = int(round(heights[i]))
                
                # Skip tiny detections
                if bw < 5 or bh < 5:
                    continue
                
                cls_id = int(class_ids[i])
                confidence = float(max_scores[i])
                det_type = CLASS_TO_TYPE.get(cls_id, 'unknown')
                subtype = CLASS_TO_SUBTYPE.get(cls_id, 'unknown')
                
                center_x = bx + bw // 2
                center_y = by + bh // 2
                
                area_frac = (bw * bh) / max(orig_h * orig_w, 1)
                
                detection = {
                    'type': det_type,
                    'subtype': subtype,
                    'class_id': cls_id,
                    'class_name': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else 'unknown',
                    'bbox': (bx, by, bw, bh),
                    'confidence': round(confidence, 4),
                    'center_x': center_x,
                    'center_y': center_y,
                    'area_frac': round(area_frac, 5),
                    'measurements': {
                        'area_frac': round(area_frac, 5),
                        'confidence': round(confidence, 4),
                        'class_name': CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else 'unknown',
                        'detector': 'yolov8n_rdd2022',
                    }
                }
                
                # Add type-specific measurements
                if det_type == 'crack':
                    detection['measurements']['length_px'] = max(bw, bh)
                    detection['measurements']['elongation'] = round(max(bw, bh) / max(min(bw, bh), 1), 2)
                    detection['measurements']['mean_width_px'] = round(min(bw, bh), 2)
                    detection['measurements']['density'] = round(area_frac, 6)
                    detection['measurements']['continuity'] = round(min(1.0, max(bw, bh) / max(np.sqrt(bw**2 + bh**2), 1)), 3)
                
                detections.append(detection)
        
        return detections
    
    def detect(self, frame: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> List[Dict]:
        """
        Run detection on a frame.
        
        Args:
            frame: BGR image as numpy array
            roi_mask: Optional ROI mask (detections outside mask are filtered)
        
        Returns:
            List of detection dicts with keys:
              type, subtype, class_id, class_name, bbox (x,y,w,h),
              confidence, center_x, center_y, area_frac, measurements
        """
        if self.net is None:
            raise RuntimeError("Model not loaded")
        
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess
        blob, ratio, dw, dh = self._preprocess(frame)
        
        # Inference
        with self._inference_lock:
            self.net.setInput(blob)
            output = self.net.forward()
        
        # Postprocess
        detections = self._postprocess(output, ratio, dw, dh, orig_h, orig_w)
        
        # Filter by ROI mask if provided
        if roi_mask is not None and len(detections) > 0:
            filtered = []
            for det in detections:
                cx, cy = det['center_x'], det['center_y']
                if 0 <= cy < roi_mask.shape[0] and 0 <= cx < roi_mask.shape[1]:
                    if roi_mask[cy, cx] > 0:
                        filtered.append(det)
            detections = filtered
        
        # Sort by confidence
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        
        logger.info(f"YOLO detected {len(detections)} objects "
                     f"({sum(1 for d in detections if d['type'] == 'pothole')} potholes, "
                     f"{sum(1 for d in detections if d['type'] == 'crack')} cracks)")
        
        return detections
    
    def info(self) -> Dict:
        """Return model info."""
        return {
            'detector': 'yolov8n',
            'backend': 'opencv_dnn',
            'model_path': self.model_path,
            'input_size': self.input_size,
            'conf_threshold': self.conf_threshold,
            'nms_threshold': self.nms_threshold,
            'metadata': self.metadata,
        }


# Global detector instance (lazy loaded)
_detector: Optional[YOLODetector] = None
_detector_lock = threading.Lock()


def get_detector(model_path: Optional[str] = None) -> YOLODetector:
    """Get or create the global YOLO detector instance."""
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = YOLODetector(model_path=model_path)
    return _detector


def detect_all(frame: np.ndarray, roi_mask: Optional[np.ndarray] = None,
               model_path: Optional[str] = None) -> List[Dict]:
    """
    Convenience function to detect potholes and cracks in a frame.
    
    Args:
        frame: BGR image
        roi_mask: Optional ROI mask
        model_path: Optional path to ONNX model
    
    Returns:
        List of detection dicts
    """
    detector = get_detector(model_path)
    return detector.detect(frame, roi_mask)
