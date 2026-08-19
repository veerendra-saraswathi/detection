#!/usr/bin/env python3
"""
Dataset checker for thermal_kaist labels.

Checks:
- Missing label files
- Empty label files
- Invalid class IDs
- Invalid coordinates (out of [0,1] range)
- Basic statistics

Usage:
  python scripts/check_dataset.py --data-dir data/thermal_kaist
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


VALID_CLASSES = {0, 1, 2, 3}  # person, car, truck, bus


def check_split(images_dir: Path, labels_dir: Path, split_name: str):
    print(f"\n=== {split_name.upper()} ===")

    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    label_files = {p.stem: p for p in labels_dir.glob("*.txt")}

    missing_labels = []
    empty_labels = []
    invalid_class = []
    invalid_coords = []
    class_counts = defaultdict(int)
    total_boxes = 0

    for img_path in image_files:
        stem = img_path.stem
        label_path = label_files.get(stem)

        if label_path is None:
            missing_labels.append(stem)
            continue

        content = label_path.read_text().strip()
        if not content:
            empty_labels.append(stem)
            continue

        for line_num, line in enumerate(content.splitlines(), 1):
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_coords.append(f"{stem} line {line_num}: wrong number of values")
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
            except ValueError:
                invalid_coords.append(f"{stem} line {line_num}: cannot parse")
                continue

            if cls_id not in VALID_CLASSES:
                invalid_class.append(f"{stem} line {line_num}: class_id={cls_id}")

            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                invalid_coords.append(f"{stem} line {line_num}: coords out of range")

            if w <= 0 or h <= 0:
                invalid_coords.append(f"{stem} line {line_num}: width/height <= 0")

            class_counts[cls_id] += 1
            total_boxes += 1

    print(f"Images              : {len(image_files)}")
    print(f"Label files         : {len(label_files)}")
    print(f"Missing labels      : {len(missing_labels)}")
    print(f"Empty labels        : {len(empty_labels)}")
    print(f"Invalid class IDs   : {len(invalid_class)}")
    print(f"Invalid coordinates : {len(invalid_coords)}")
    print(f"Total boxes         : {total_boxes}")
    print(f"Per-class counts    : {dict(class_counts)}")

    if missing_labels:
        print("  Missing:", missing_labels[:5], "..." if len(missing_labels) > 5 else "")
    if empty_labels:
        print("  Empty  :", empty_labels[:5], "..." if len(empty_labels) > 5 else "")
    if invalid_class:
        print("  Bad class examples:", invalid_class[:3])
    if invalid_coords:
        print("  Bad coord examples:", invalid_coords[:3])

    return {
        "images": len(image_files),
        "missing": len(missing_labels),
        "empty": len(empty_labels),
        "invalid_class": len(invalid_class),
        "invalid_coords": len(invalid_coords),
        "total_boxes": total_boxes,
    }


def main():
    parser = argparse.ArgumentParser(description="Check thermal_kaist dataset labels")
    parser.add_argument("--data-dir", type=str, default="data/thermal_kaist")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("=" * 60)
    print("  Dataset Checker – thermal_kaist")
    print("=" * 60)

    total_issues = 0

    for split in ["train", "val"]:
        images_dir = data_dir / "images" / split
        labels_dir = data_dir / "labels" / split

        if not images_dir.exists():
            print(f"\n[SKIP] {images_dir} does not exist")
            continue

        stats = check_split(images_dir, labels_dir, split)
        total_issues += (
            stats["missing"] + stats["empty"] +
            stats["invalid_class"] + stats["invalid_coords"]
        )

    print("\n" + "=" * 60)
    if total_issues == 0:
        print("Result: Dataset looks clean. Ready for fine-tuning.")
    else:
        print(f"Result: Found {total_issues} issues. Please fix before training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
    