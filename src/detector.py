from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class Detection:
    object_id: int
    class_name: str
    x: float
    y: float
    width: float
    height: float
    confidence: float
    mask: Optional[np.ndarray] = None
    class_id: Optional[int] = None


@dataclass
class DetectionResult:
    image_path: Optional[str] = None
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    model_name: str = ""


class BaseDetector:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.class_names: Dict[int, str] = {}
        self.thresholds: Dict[str, float] = {}

    def load(self, model_path: str | Path) -> None:
        raise NotImplementedError

    def detect(
        self,
        image: np.ndarray,
        object_defs: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
    ) -> DetectionResult:
        raise NotImplementedError

    def detect_from_path(
        self,
        image_path: str | Path,
        object_defs: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
    ) -> DetectionResult:
        raise NotImplementedError
        