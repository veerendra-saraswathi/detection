#!/usr/bin/env python3
"""
Generate high-quality pseudo-labels from KAIST visible images.

Design goals:
- Over-detect rather than under-detect (low confidence threshold)
- Keep only the 4 target classes: person, car, truck, bus
- Save YOLO-format labels
- Save preview images with boxes for fast visual review
- Produce a summary report
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# COCO class name → our target class ID
TARGET_CLASSES = {
    "person": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
}

# Colors for preview (BGR)
COLORS = {
    0: (0, 255, 0),      # person - green
    1: (255, 128, 0),    # car
    2: (0, 165, 255),    # truck - orange
    3: (0, 255, 255),    # bus - yellow
}


def draw_preview(image, detections, class_names):
    vis = image.copy()
    for det in detections:
        cls_id, conf, x1, y1, x2, y2 = det
        color = COLORS.get(cls_id, (0, 255, 0))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        label = f"{class_names[cls_id]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (int(x1), int(y1) - th - 6), (int(x1) + tw + 4, int(y1)), color, -1)
        cv2.putText(vis, label, (int(x1) + 2, int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis


def process_split(
    model: YOLO,
    visible_dir: Path,
    labels_dir: Path,
    preview_dir: Path,
    conf_threshold: float,
):
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(visible_dir.glob("*.jpg")) + list(visible_dir.glob("*.png")))
    stats = defaultdict(int)
    images_with_det = 0
    total_dets = 0

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path.name}")
            continue

        h, w = img.shape[:2]
        results = model.predict(source=img, conf=conf_threshold, verbose=False, imgsz=640)

        detections = []
        yolo_lines = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id_coco = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = model.names[cls_id_coco]

                if class_name not in TARGET_CLASSES:
                    continue

                our_id = TARGET_CLASSES[class_name]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # YOLO normalized format
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                yolo_lines.append(f"{our_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                detections.append((our_id, conf, x1, y1, x2, y2))
                stats[class_name] += 1
                total_dets += 1

        # Save label file
        stem = img_path.stem.replace("visible_", "thermal_")
        label_path = labels_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # Save preview
        preview = draw_preview(img, detections, {v: k for k, v in TARGET_CLASSES.items()})
        preview_path = preview_dir / f"{stem}_pred.jpg"
        cv2.imwrite(str(preview_path), preview)

        if detections:
            images_with_det += 1

    return {
        "images": len(image_files),
        "images_with_detections": images_with_det,
        "total_detections": total_dets,
        "per_class": dict(stats),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels from visible images")
    parser.add_argument("--visible-dir", type=str, default="data/thermal_kaist/visible",
                        help="Root of visible images (contains train/ and val/)")
    parser.add_argument("--output-dir", type=str, default="data/thermal_kaist")
    parser.add_argument("--model", type=str, default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.12,
                        help="Low confidence threshold (over-detect is better for annotation)")
    args = parser.parse_args()

    print("=" * 64)
    print("  Pseudo-Label Generation for Thermal Dataset")
    print("=" * 64)
    print(f"Model          : {args.model}")
    print(f"Confidence     : {args.conf}")
    print(f"Target classes : {list(TARGET_CLASSES.keys())}")
    print()

    model = YOLO(args.model)

    visible_root = Path(args.visible_dir)
    output_root = Path(args.output_dir)

    for split in ["train", "val"]:
        vis_dir = visible_root / split
        if not vis_dir.exists():
            print(f"[SKIP] {vis_dir} does not exist")
            continue

        print(f"Processing {split} ...")
        labels_dir = output_root / "labels" / split
        preview_dir = output_root / "preview" / split

        stats = process_split(
            model=model,
            visible_dir=vis_dir,
            labels_dir=labels_dir,
            preview_dir=preview_dir,
            conf_threshold=args.conf,
        )

        print(f"  Images processed      : {stats['images']}")
        print(f"  Images with detections: {stats['images_with_detections']}")
        print(f"  Total detections      : {stats['total_detections']}")
        print(f"  Per class             : {stats['per_class']}")
        print()

    print("Done.")
    print("Next steps:")
    print("  1. Review the preview/ images")
    print("  2. Correct the labels/ in CVAT or Label Studio")
    print("  3. After correction, run fine-tuning")


if __name__ == "__main__":
    main()
    