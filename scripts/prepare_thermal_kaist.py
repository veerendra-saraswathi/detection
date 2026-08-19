#!/usr/bin/env python3
"""
Diversity-aware dataset preparation for official KAIST structure.

Supports the common layout:
  images/setXX/VXXX/lwir/*.jpg
  images/setXX/VXXX/visible/*.jpg
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from pathlib import Path
from typing import List, Tuple


def extract_frame_id(filename: str) -> int:
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def find_kaist_pairs(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    lwir_files = list(root.rglob("lwir/*.jpg")) + list(root.rglob("lwir/*.png"))
    lwir_files = sorted(lwir_files)

    for lwir_path in lwir_files:
        visible_path = lwir_path.parent.parent / "visible" / lwir_path.name
        if visible_path.exists():
            pairs.append((lwir_path, visible_path))

    return pairs


def uniform_sample(pairs: List[Tuple[Path, Path]], num_samples: int) -> List[Tuple[Path, Path]]:
    if len(pairs) <= num_samples:
        return pairs

    pairs = sorted(pairs, key=lambda p: extract_frame_id(p[0].name))
    step = len(pairs) / num_samples
    selected = [pairs[int(i * step)] for i in range(num_samples)]
    return selected


def prepare_dataset(
    root: Path,
    output_dir: Path,
    num_samples: int = 100,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    random.seed(seed)

    print("=" * 64)
    print("  Diversity-aware KAIST Thermal Dataset Preparation")
    print("=" * 64)

    root = Path(root)
    output_dir = Path(output_dir)

    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    print(f"\nScanning: {root}")
    pairs = find_kaist_pairs(root)
    print(f"Found {len(pairs)} thermal-visible pairs")

    if len(pairs) == 0:
        print("No pairs found. Expected structure: .../lwir/*.jpg and .../visible/*.jpg")
        return

    selected = uniform_sample(pairs, num_samples)
    print(f"Selected {len(selected)} diverse frames (uniform sampling)")

    random.shuffle(selected)
    n_val = max(1, int(len(selected) * val_ratio))
    val_pairs = selected[:n_val]
    train_pairs = selected[n_val:]

    print(f"Train: {len(train_pairs)}  |  Val: {len(val_pairs)}")

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "visible" / split).mkdir(parents=True, exist_ok=True)

    manifest = []

    def copy_split(pairs_list, split_name):
        for lwir_path, visible_path in pairs_list:
            frame_id = extract_frame_id(lwir_path.name)
            base_name = f"thermal_{frame_id:05d}.jpg"
            vis_name = f"visible_{frame_id:05d}.jpg"

            dst_thermal = output_dir / "images" / split_name / base_name
            shutil.copy2(lwir_path, dst_thermal)

            dst_visible = output_dir / "visible" / split_name / vis_name
            shutil.copy2(visible_path, dst_visible)

            label_path = output_dir / "labels" / split_name / (base_name.replace(".jpg", ".txt"))
            label_path.touch()

            manifest.append({
                "thermal_image": base_name,
                "visible_image": vis_name,
                "split": split_name,
                "original_lwir": str(lwir_path),
            })

    copy_split(train_pairs, "train")
    copy_split(val_pairs, "val")

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["thermal_image", "visible_image", "split", "original_lwir"])
        writer.writeheader()
        writer.writerows(manifest)

    yaml_path = output_dir / "dataset.yaml"
    yaml_content = """path: data/thermal_kaist
train: images/train
val: images/val

nc: 4

names:
  0: person
  1: car
  2: truck
  3: bus
"""
    yaml_path.write_text(yaml_content)

    print("\nDone.")
    print(f"Output directory : {output_dir}")
    print(f"Manifest         : {manifest_path}")
    print(f"Train images     : {len(train_pairs)}")
    print(f"Val images       : {len(val_pairs)}")
    print("\nNext step: Annotate the images (use the visible/ folder to accelerate labeling).")


def main():
    parser = argparse.ArgumentParser(description="Diversity-aware KAIST thermal dataset preparation")
    parser.add_argument("--root", type=str, required=True,
                        help="Root of KAIST images (contains setXX/VXXX/lwir and visible)")
    parser.add_argument("--output-dir", type=str, default="data/thermal_kaist")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prepare_dataset(
        root=Path(args.root),
        output_dir=Path(args.output_dir),
        num_samples=args.num_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
    