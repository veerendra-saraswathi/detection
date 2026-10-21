# Detection Results — Cascade POC vs YOLO11n

Comparison of the two approaches currently in this repository for the RCI Automatic Object Detection problem.

## Test Images Available in the Repo

| Location | Files | Notes |
|----------|-------|-------|
| Repo root | `thermal.raw`, `thermal_00000.raw`, `thermal_00064.raw`, `thermal_00096.raw`, `thermal_00448.raw`, `thermal_02048.raw`, `thermal_03008.raw` | Raw 8-bit thermal frames (gray8) |
| `data/thermal_kaist/images/train/` | `thermal_02048.jpg` (and others if present) | JPEG version used by the Python demo |

## The Two Approaches

### 1. Pure C++ Cascade (primary POC)
- Stages: local-contrast candidates → border suppression → light aspect verifier → greedy NMS
- Self-contained, zero external runtime libraries
- Target: Jetson Xavier NX / Orin Nano
- Demo: `./build/rci_detect_demo frame.raw 640 512`

### 2. YOLO11n (comparison baseline)
- Stock COCO-pretrained model (not fine-tuned on thermal or RCI classes)
- Used only via Python for integration and speed comparison
- Demo: `python scripts/demo_cascade_vs_yolo.py --image ...`

## Comparison (from existing runs and README notes)

| Aspect | Cascade POC | YOLO11n (stock) |
|--------|-------------|-----------------|
| Dependencies | None (pure C++) | Ultralytics / PyTorch |
| Speed (laptop, 640×512) | ~5–8 ms | Faster in the demo (speed only) |
| Confidence scores | Low heuristic (0.11–0.19) | Higher (COCO-trained) |
| Known issue on `thermal_02048` | Boxes appear at y ≈ 8–72 (misaligned vs GT y ≈ 167–254) | Not adapted to thermal |
| Evaluation metrics (IoU, precision, recall) | Not yet measured | Not measured (COCO model) |
| Jetson FPS / memory | Not yet measured | Not measured |
| Suitable for offline edge | Yes (design goal) | Heavier |

## How to Reproduce

```bash
# Cascade only
cmake -S . -B build && cmake --build build
./build/rci_detect_demo thermal_02048.raw 640 512

# Both approaches
python scripts/demo_cascade_vs_yolo.py --image data/thermal_kaist/images/train/thermal_02048.jpg
```

## Status

This is an architectural POC. Exact reconstruction of detections, IoU against ground truth, and Jetson numbers are still missing (as noted in the main README).

Next useful additions:
- Save detection CSVs or visualized images into `results/sample_outputs/`
- Add a small evaluation script that computes IoU when annotations are available
- Record actual console output from the demo here
