#!/usr/bin/env python3
"""
Working detection demo for RCO EOI Problem #6.

Examples:
  # Normal COCO classes
  python demos/demo_detection.py --image data/public/bus.jpg --save

  # Only RCI-relevant classes
  python demos/demo_detection.py --image data/public/bus.jpg --rci --save

  # Specific classes
  python demos/demo_detection.py --image data/public/bus.jpg --classes person bus truck --save
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.yolo_detector import YOLODetector
from src.visualize import draw_detections, print_detections_table


# Mapping from COCO class names → RCI-relevant names
# (used while we are still on COCO-pretrained weights)
RCI_RELEVANT = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "airplane",
    "boat",
}


def main():
    parser = argparse.ArgumentParser(
        description="Automatic Object Detection – RCO EOI Problem #6 (Visible + IR)"
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--classes", "-c", nargs="+", default=None,
        help="Object classes to keep (e.g. person car truck)"
    )
    parser.add_argument(
        "--rci", action="store_true",
        help="Keep only RCI-relevant classes (person, car, truck, bus, motorcycle...)"
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--model", "-m", type=str, default="yolo11n.pt",
        help="YOLO model weights"
    )
    parser.add_argument("--save", action="store_true", help="Save visualization to outputs/")
    parser.add_argument("--no-show", action="store_true", help="Do not open display window")
    parser.add_argument(
        "--ir-method", type=str, default="percentile",
        choices=["minmax", "percentile", "clahe"],
        help="IR normalization method"
    )
    args = parser.parse_args()

    print("=" * 64)
    print("  Automatic Object Detection – RCO EOI Problem #6")
    print("  Cluttered Visible & Infrared Imagery")
    print("=" * 64)

    # Decide which classes to keep
    object_defs = args.classes
    if args.rci and object_defs is None:
        object_defs = list(RCI_RELEVANT)
        print(f"\n[RCI mode] Filtering to: {sorted(object_defs)}")

    detector = YOLODetector(model_name=args.model, class_config="coco")
    detector.load()

    print(f"\nRunning detection on: {args.image}")
    if object_defs:
        print(f"Keeping classes    : {object_defs}")

    result = detector.detect_from_path(
        args.image,
        object_defs=object_defs,
        conf_threshold=args.conf,
        ir_method=args.ir_method,
    )

    print_detections_table(result)

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    vis = draw_detections(img, result)

    if args.save:
        out_dir = ROOT / "outputs"
        out_dir.mkdir(exist_ok=True)
        stem = Path(args.image).stem
        suffix = "_rci" if args.rci else ""
        out_path = out_dir / f"{stem}{suffix}_det.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"\nSaved visualization → {out_path}")

    if not args.no_show:
        h, w = vis.shape[:2]
        max_dim = 1280
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

        cv2.imshow("RCO EOI – Object Detection (press any key)", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    