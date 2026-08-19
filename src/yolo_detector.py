"""
YOLO-based detector implementation for RCO EOI Problem #6.

Supports:
- Visible and Infrared images
- COCO pretrained models or fine-tuned RCI models
- Easy switching between COCO and RCI class sets
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from .detector import BaseDetector, Detection, DetectionResult
from .ir_preprocess import prepare_for_detector


class YOLODetector(BaseDetector):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_name: str = "yolo11n.pt",
        class_config: str = "coco",          # "rci" or "coco"
    ):
        super().__init__(config)
        self.model_name = model_name
        self.class_config = class_config
        self.model: Optional[YOLO] = None
        self._load_class_config()

    def _load_class_config(self) -> None:
        """Load class names and thresholds."""
        root = Path(__file__).resolve().parents[1]

        if self.class_config == "rci":
            config_path = root / "configs" / "rci_classes.yaml"
        else:
            config_path = root / "configs" / "classes.yaml"

        if config_path.exists():
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            self.class_names = {int(k): v for k, v in data.get("names", {}).items()}
            self.thresholds = data.get("thresholds", {})
        else:
            self.class_names = {}
            self.thresholds = {}

    def load(self, model_path: Union[str, Path] = None) -> None:
        path = str(model_path) if model_path else self.model_name
        print(f"[YOLODetector] Loading model: {path}")
        self.model = YOLO(path)

        # When using a COCO-pretrained model we prefer the model's own class names
        if self.class_config != "rci" and hasattr(self.model, "names") and self.model.names:
            self.class_names = {int(k): v for k, v in self.model.names.items()}

        print(f"[YOLODetector] Model loaded. Classes available: {len(self.class_names)}")

    def detect(
        self,
        image: np.ndarray,
        object_defs: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        ir_method: str = "percentile",
    ) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        start = time.perf_counter()

        # Unified preprocessing (handles IR + Visible)
        img = prepare_for_detector(image, to_3channel=True, ir_method=ir_method)

        results = self.model.predict(
            source=img,
            conf=conf_threshold,
            verbose=False,
            imgsz=640,
        )

        detections: List[Detection] = []
        object_id = 0
        allowed_classes = {c.lower() for c in object_defs} if object_defs else None

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                if allowed_classes and class_name.lower() not in allowed_classes:
                    continue

                class_thresh = self.thresholds.get(class_name, conf_threshold)
                if conf < class_thresh:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                det = Detection(
                    object_id=object_id,
                    class_name=class_name,
                    x=float(x1),
                    y=float(y1),
                    width=float(x2 - x1),
                    height=float(y2 - y1),
                    confidence=conf,
                    class_id=cls_id,
                    mask=None,
                )
                detections.append(det)
                object_id += 1

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return DetectionResult(
            detections=detections,
            inference_time_ms=elapsed_ms,
            model_name=self.model_name,
        )

    def detect_from_path(
        self,
        image_path: Union[str, Path],
        object_defs: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        ir_method: str = "percentile",
    ) -> DetectionResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read image: {path}")

        result = self.detect(
            img,
            object_defs=object_defs,
            conf_threshold=conf_threshold,
            ir_method=ir_method,
        )
        result.image_path = str(path)
        return result
        