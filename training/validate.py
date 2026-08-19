#!/usr/bin/env python3
"""
Validation / evaluation script for RCO EOI Problem #6.
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Validate a trained detector")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--data", type=str, default="data/custom/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("  RCO EOI Problem #6 – Validation")
    print("=" * 60)

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("\nValidation complete.")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")


if __name__ == "__main__":
    main()
    