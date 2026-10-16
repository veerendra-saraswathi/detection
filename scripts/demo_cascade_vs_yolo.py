#!/usr/bin/env python3
"""Load one thermal image and print cascade and YOLO detections.

    python scripts/demo_cascade_vs_yolo.py
    python scripts/demo_cascade_vs_yolo.py --image data/thermal_kaist/images/train/thermal_02048.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cascade_detector import CascadeDetector
from src.visualize import print_detections_table

DEFAULT_IMAGE = ROOT / "data" / "thermal_kaist" / "images" / "train" / "thermal_02048.jpg"


def _print_cascade_stats(detector: CascadeDetector) -> None:
    stats = detector.last_stats
    if not stats:
        return
    print(
        "cascade stages: "
        f"candidates {stats['candidate_count']}  "
        f"before_nms {stats['verified_count']}  "
        f"kept {stats['kept_count']}  "
        f"propose {stats['propose_ms']:.3f} ms  "
        f"verify {stats['verify_ms']:.3f} ms  "
        f"nms {stats['nms_ms']:.3f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the RCI cascade POC with YOLO on one image")
    parser.add_argument("--image", "-i", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--lib", type=Path, default=None, help="Path to librci_detect.dylib or .so")
    parser.add_argument("--conf", type=float, default=0.0,
                        help="Cascade confidence floor. Stub scores are uncalibrated and usually below 0.25.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence floor")
    args = parser.parse_args()

    if not args.image.is_file():
        print(f"image not found: {args.image}", file=sys.stderr)
        return 1

    image = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1

    print(f"image: {args.image} shape={image.shape} dtype={image.dtype}")

    cascade = CascadeDetector(library_path=str(args.lib) if args.lib else None)
    cascade.load()
    cascade_result = cascade.detect(image, conf_threshold=args.conf)
    cascade_result.image_path = str(args.image)
    print("\n=== cascade ===")
    _print_cascade_stats(cascade)
    print_detections_table(cascade_result)

    print("\n=== yolo ===")
    try:
        from src.yolo_detector import YOLODetector

        yolo = YOLODetector(model_name="yolo11n.pt", class_config="coco")
        yolo.load()
        yolo_result = yolo.detect(
            image, object_defs=["person", "car", "truck"], conf_threshold=args.yolo_conf
        )
        yolo_result.image_path = str(args.image)
        print_detections_table(yolo_result)
    except Exception as exc:
        print(f"YOLO did not run: {type(exc).__name__}: {exc}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
