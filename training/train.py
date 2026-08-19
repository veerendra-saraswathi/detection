#!/usr/bin/env python3
"""
Fine-tuning script for RCO EOI Problem #6.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Fine-tune detector for RCO EOI Problem #6")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base model to start from (yolo11n.pt, yolo11s.pt, ...)")
    parser.add_argument("--data", type=str, default="data/custom/dataset.yaml",
                        help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (reduce if OOM)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / mps / 0")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project folder")
    parser.add_argument("--name", type=str, default="rci_finetune", help="Run name")
    parser.add_argument("--resume", action="store_true", help="Resume last training")
    args = parser.parse_args()

    print("=" * 60)
    print("  RCO EOI Problem #6 – Fine-tuning")
    print("=" * 60)

    if args.resume:
        print("Resuming previous training...")
        model = YOLO("last.pt")
        model.train(resume=True)
        return

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        print("Please create data/custom/dataset.yaml and add your images + labels first.")
        return

    print(f"Base model   : {args.model}")
    print(f"Dataset      : {args.data}")
    print(f"Epochs       : {args.epochs}")
    print(f"Image size   : {args.imgsz}")
    print(f"Batch size   : {args.batch}")
    print(f"Device       : {args.device}")
    print()

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
    )

    print("\nTraining finished.")
    print(f"Best weights saved under: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
    