cd ~/detection

# Create README.md
cat > README.md << 'EOF'
# RCI Automatic Object Detection — Cascade POC

Pure-C++ cascade for cluttered visible and infrared object detection.  
RCI Scope of Work response. Edge target: Jetson Xavier NX / Orin Nano. Offline, zero third-party runtime libraries.

**Status**: Architectural POC only. Not a detection MVP.

## Current implementation
- C API (`src/detect.h`): image + object definitions → detection records (id, type, name, x, y, w, h, confidence)
- Stages: local-contrast candidates + border suppression → light aspect verifier → greedy NMS
- Profiling and debug output
- CMake + demo for synthetic or raw gray8
- ~5–8 ms on 640×512 (laptop)

## Explicitly missing (not yet in this repo)
- Trained weights or class-specific models
- Evaluation run with recall / precision / IoU
- Jetson build, FPS or memory numbers
- Formal object-definition file interface
- Confidence calibration
- Segmentation mask output
- Occlusion flag / ratio
- Performance report

## Data
KAIST Multispectral Pedestrian (thermal). Subset used ≈ **1.2 GB+**.  
Official: https://github.com/SoonminHwang/rgbt-ped-detection (full set ~36 GB)

## Build
```bash
cmake -S . -B build && cmake --build build
./build/rci_detect_demo
./build/rci_detect_demo frame.raw 640 512
```

## Shared library and Python comparison

The same CMake build also writes `build/librci_detect.dylib` (macOS) or `build/librci_detect.so` (Linux, including Jetson). `src/cascade_detector.py` loads that library with ctypes and calls `rci_detect_profiled`.

```bash
cmake -S . -B build && cmake --build build
source .venv/bin/activate
python scripts/demo_cascade_vs_yolo.py --image data/thermal_kaist/images/train/thermal_02048.jpg
```

`RCI_DETECT_LIB` overrides the library path. The C demo still links the static archive.
## Known limitations (current POC run)
- On KAIST thermal_02048.jpg the cascade boxes appear in the upper part of the frame (y ≈ 8–72) while ground-truth objects are lower (y ≈ 167–254). This is inside the C++ scale/contrast logic.
- Cascade scores are heuristic and low (~0.11–0.19).
- YOLO11n is the unmodified COCO model.

## Known limitations (current POC run)
- On KAIST `thermal_02048.jpg` the cascade boxes appear in the upper part of the frame (y ≈ 8–72) while ground-truth objects are lower (y ≈ 167–254). This is inside the existing C++ scale/contrast logic, not the Python wrapper.
- Cascade confidence scores are heuristic and low (typically 0.11–0.19).
- YOLO11n is the stock COCO-pretrained model (not fine-tuned on thermal or RCI classes).
- Demo shows integration and relative speed, not final detection accuracy.
