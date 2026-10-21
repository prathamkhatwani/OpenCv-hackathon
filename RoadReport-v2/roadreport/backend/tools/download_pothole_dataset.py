"""Download and prepare pothole dataset for YOLOv8 training.

Strategy:
1. Try RDD2022 India from Figshare (official, 6650 images)
2. Prepare into YOLO format using existing rdd2022.py
3. Generate data.yaml for ultralytics training

Usage:
    python -m tools.download_pothole_dataset --output data/pothole_dataset
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

# RDD2022 India download constants (from existing download_rdd2022.py)
RDD2022_URL = 'https://ndownloader.figshare.com/files/38030910'
HEADER = 609341032
SIZE = 526709145
TOTAL = 13264172619
CRC = 0xf186f6f8


def download_rdd2022_india(output_zip: Path):
    """Download RDD2022 India.zip from Figshare using range requests."""
    print(f"Downloading RDD2022 India dataset ({SIZE // 1048576} MiB)...")
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    
    # Verify header
    req = urllib.request.Request(RDD2022_URL, headers={'Range': f'bytes={HEADER}-{HEADER+46}'})
    with urllib.request.urlopen(req, timeout=60) as r:
        if r.status != 206:
            raise ValueError(f'Range request not supported (status {r.status})')
        h = r.read(48)
    
    if len(h) != 47 or h[:4] != b'PK\x03\x04':
        raise ValueError('Upstream archive layout changed')
    
    start = HEADER + 47
    end = start + SIZE - 1
    temp = output_zip.with_suffix('.partial')
    crc = 0
    total = 0
    sha = hashlib.sha256()
    
    try:
        req = urllib.request.Request(RDD2022_URL, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req, timeout=90) as r, temp.open('wb') as f:
            if r.status != 206:
                raise ValueError('Country range not honored')
            while block := r.read(1024 * 1024):
                total += len(block)
                if total > SIZE:
                    raise ValueError('Response too large')
                crc = zlib.crc32(block, crc)
                sha.update(block)
                f.write(block)
                if total % (32 * 1024 * 1024) < 1024 * 1024:
                    print(f'  Downloaded {total // 1048576} / {SIZE // 1048576} MiB', flush=True)
        
        if total != SIZE or crc != CRC:
            raise ValueError('Archive size/CRC mismatch')
        
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:
                raise ValueError('Corrupt archive')
        
        temp.replace(output_zip)
        print(f"Download complete: {output_zip}")
        
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def prepare_yolo_dataset(archive_path: Path, output_dir: Path):
    """Prepare RDD2022 archive into YOLO format dataset."""
    print(f"Preparing dataset from {archive_path}...")
    
    # Use existing rdd2022.py prepare function
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from app.datasets.rdd2022 import prepare
    
    summary = prepare(str(archive_path), str(output_dir))
    print(json.dumps(summary, indent=2))
    
    # The rdd2022.py already creates data.yaml, YOLO labels, and COCO annotations
    print(f"\nDataset prepared at: {output_dir}")
    print(f"  Images: {summary['images']}")
    print(f"  Classes: {summary['class_counts']}")
    print(f"  Splits: {json.dumps(summary['splits'], indent=2)}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Download and prepare pothole dataset')
    parser.add_argument('--output', default='data/pothole_dataset',
                        help='Output directory for prepared dataset')
    parser.add_argument('--archive', default=None,
                        help='Path to existing RDD2022_India.zip (skip download)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.archive:
        archive_path = Path(args.archive)
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
    else:
        archive_path = Path('data/downloads/RDD2022_India.zip')
        if not archive_path.exists():
            download_rdd2022_india(archive_path)
    
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Output directory {output_dir} already exists and is not empty.")
        print("Skipping preparation (use a new output path to re-prepare).")
        return
    
    prepare_yolo_dataset(archive_path, output_dir)


if __name__ == '__main__':
    main()
