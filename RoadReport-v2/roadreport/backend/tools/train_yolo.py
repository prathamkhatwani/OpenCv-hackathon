"""Train YOLOv8n on RDD2022 pothole dataset and export to ONNX for OpenCV DNN.

Usage:
    python -m tools.train_yolo --data data/pothole_dataset/data.yaml --epochs 100
"""
import argparse
import json
import shutil
from pathlib import Path


def train(data_yaml: str, epochs: int = 100, imgsz: int = 640, batch: int = 16,
          output_dir: str = 'runs/train', device: str = 'auto'):
    """Train YOLOv8n and export to ONNX."""
    from ultralytics import YOLO
    
    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    
    print(f"Training YOLOv8n on {data_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch}")
    import torch
    if device is None or device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    # Load pretrained YOLOv8n
    model = YOLO('yolov8n.pt')
    
    # Train
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=4,
        cache=True,
        name='pothole_detector',
        project=output_dir,
        exist_ok=True,
        patience=15,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        verbose=True,
        # Augmentation for road imagery
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,     # No rotation (road stays horizontal)
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,       # No vertical flip (roads don't flip)
        fliplr=0.5,        # Horizontal flip is valid
        mosaic=1.0,
        mixup=0.1,
    )
    
    # Find best weights robustly
    trainer_best = getattr(model.trainer, 'best', None) if hasattr(model, 'trainer') else None
    trainer_save_dir = Path(getattr(model.trainer, 'save_dir', '')) if hasattr(model, 'trainer') else None
    
    best_pt = None
    if trainer_best and Path(trainer_best).exists():
        best_pt = Path(trainer_best)
    elif trainer_save_dir:
        for fname in ['best.pt', 'last.pt']:
            candidate = trainer_save_dir / 'weights' / fname
            if candidate.exists():
                best_pt = candidate
                break
    
    if not best_pt or not best_pt.exists():
        # Search common fallback paths
        search_dirs = [
            Path(output_dir) / 'pothole_detector' / 'weights',
            Path('runs/detect') / output_dir / 'pothole_detector' / 'weights',
            Path('runs') / output_dir / 'pothole_detector' / 'weights',
        ]
        for sdir in search_dirs:
            for fname in ['best.pt', 'last.pt']:
                candidate = sdir / fname
                if candidate.exists():
                    best_pt = candidate
                    break
            if best_pt:
                break
    
    if not best_pt or not best_pt.exists():
        raise FileNotFoundError(f"Training did not produce weights file. Searched in save_dir and {output_dir}")
    
    print(f"\nBest model: {best_pt}")
    
    # Export to ONNX for OpenCV DNN
    print("Exporting to ONNX...")
    best_model = YOLO(str(best_pt))
    onnx_path = best_model.export(format='onnx', imgsz=imgsz, simplify=True, opset=12)
    
    print(f"ONNX model exported: {onnx_path}")
    
    # Copy ONNX model to app/models/
    models_dir = Path(__file__).parent.parent / 'app' / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / 'pothole_yolov8n.onnx'
    shutil.copy2(onnx_path, dest)
    print(f"Model copied to: {dest}")
    
    # Save class names
    names = best_model.names
    meta = {
        'model': 'yolov8n',
        'dataset': 'RDD2022_India',
        'classes': names,
        'imgsz': imgsz,
        'epochs_trained': epochs,
        'onnx_path': str(dest),
    }
    meta_path = models_dir / 'pothole_yolov8n.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved: {meta_path}")
    
    # Validate on test set
    print("\nValidating on test set...")
    metrics = best_model.val(data=str(data_path), split='test')
    
    test_metrics = {
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map),
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
    }
    
    # Per-class metrics
    if hasattr(metrics.box, 'ap_class_index'):
        for i, cls_idx in enumerate(metrics.box.ap_class_index):
            cls_name = names[int(cls_idx)]
            test_metrics[f'{cls_name}_ap50'] = float(metrics.box.ap50[i])
    
    meta['test_metrics'] = test_metrics
    meta_path.write_text(json.dumps(meta, indent=2))
    
    print(f"\nTest metrics: {json.dumps(test_metrics, indent=2)}")
    print(f"\nDone! Model ready at: {dest}")
    
    return str(dest)


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 pothole detector')
    parser.add_argument('--data', required=True, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--output', default='runs/train', help='Output directory')
    parser.add_argument('--device', default='auto', help='Device (mps, cpu, cuda, or auto)')
    args = parser.parse_args()
    
    train(args.data, args.epochs, args.imgsz, args.batch, args.output, args.device)


if __name__ == '__main__':
    main()
