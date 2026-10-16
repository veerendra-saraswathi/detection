"""
ctypes wrapper around the RCI cascade shared library.

Calls rci_detect_profiled, which has the same contract as rci_detect and also
fills the caller-owned stage stats. Image bytes and the detection buffer stay
in Python; the library does not keep them.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_char_p, c_double, c_float, c_int, c_int32, c_size_t, c_uint8
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from .detector import BaseDetector, Detection, DetectionResult
from .ir_preprocess import prepare_for_detector

RCI_PIXFMT_GRAY8 = 0
RCI_PIXFMT_RGB8 = 1

_ERROR_TEXT = {
    0: "ok",
    -1: "bad argument",
    -2: "unsupported pixel format",
    -3: "image size out of range",
    -4: "out of memory",
}

# id, name, min_w, min_h, max_w, max_h, min_confidence
# Person limits match the KAIST size prior enforced in the cascade.
_DEFAULT_OBJECTS = (
    (3, "Person", 24, 48, 50, 100, 0.35),
    (2, "Car", 80, 32, 160, 64, 0.40),
    (1, "Truck", 96, 48, 192, 96, 0.45),
)


class RciObjectDef(ctypes.Structure):
    _fields_ = [
        ("id", c_int32),
        ("name", c_char_p),
        ("min_width", c_int32),
        ("min_height", c_int32),
        ("max_width", c_int32),
        ("max_height", c_int32),
        ("min_confidence", c_float),
    ]


class RciDetection(ctypes.Structure):
    _fields_ = [
        ("id", c_int32),
        ("type", c_int32),
        ("type_name", ctypes.c_char * 64),
        ("x", c_float),
        ("y", c_float),
        ("width", c_float),
        ("height", c_float),
        ("confidence", c_float),
    ]


class RciImage(ctypes.Structure):
    _fields_ = [
        ("data", POINTER(c_uint8)),
        ("width", c_int32),
        ("height", c_int32),
        ("stride", c_int32),
        ("format", c_int32),
    ]


class RciDetectParams(ctypes.Structure):
    _fields_ = [
        ("edge_threshold", c_float),
        ("nms_iou", c_float),
        ("max_detections", c_int32),
        ("scales", c_int32),
        ("scale_step", c_float),
        ("stride", c_int32),
    ]


class RciDetectStats(ctypes.Structure):
    _fields_ = [
        ("candidate_count", c_int32),
        ("verified_count", c_int32),
        ("kept_count", c_int32),
        ("propose_ms", c_double),
        ("verify_ms", c_double),
        ("nms_ms", c_double),
    ]


def _library_candidates(explicit: Optional[Path] = None) -> List[Path]:
    if explicit is not None:
        return [explicit]
    env = os.environ.get("RCI_DETECT_LIB")
    if env:
        return [Path(env)]
    root = Path(__file__).resolve().parents[1]
    names = ("librci_detect.dylib", "librci_detect.so", "rci_detect.dll")
    dirs = (root / "build", root / "build" / "Release", Path.cwd())
    return [directory / name for directory in dirs for name in names]


class CascadeDetector(BaseDetector):
    """BaseDetector backed by librci_detect."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, library_path: Optional[str] = None):
        super().__init__(config)
        self.library_path = Path(library_path) if library_path else None
        self._lib: Optional[ctypes.CDLL] = None
        self.last_stats: Dict[str, float] = {}
        self.model_name = "rci_detect"
        for spec in _DEFAULT_OBJECTS:
            self.class_names[spec[0]] = spec[1]
            self.thresholds[spec[1]] = spec[6]

    def load(self, model_path: Union[str, Path, None] = None) -> None:
        path = Path(model_path) if model_path else self.library_path
        found = None
        for candidate in _library_candidates(path):
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            looked = ", ".join(str(p) for p in _library_candidates(path))
            raise FileNotFoundError(f"librci_detect not found. Looked in: {looked}")

        lib = ctypes.CDLL(str(found))
        lib.rci_detect_params_init.argtypes = [POINTER(RciDetectParams)]
        lib.rci_detect_params_init.restype = None
        lib.rci_detect.argtypes = [
            POINTER(RciImage),
            POINTER(RciObjectDef),
            c_size_t,
            POINTER(RciDetectParams),
            POINTER(RciDetection),
            c_size_t,
            POINTER(c_size_t),
        ]
        lib.rci_detect.restype = c_int
        lib.rci_detect_profiled.argtypes = lib.rci_detect.argtypes + [POINTER(RciDetectStats)]
        lib.rci_detect_profiled.restype = c_int
        self._lib = lib
        self.library_path = found
        self.model_name = found.name

    def detect(
        self,
        image: np.ndarray,
        object_defs: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        ir_method: str = "percentile",
    ) -> DetectionResult:
        if self._lib is None:
            self.load()

        prepared = prepare_for_detector(image, to_3channel=True, ir_method=ir_method)
        # prepare_for_detector emits OpenCV BGR. The C API is RGB8.
        rgb = np.ascontiguousarray(cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB))
        height, width = rgb.shape[:2]

        specs = []
        allowed = {name.lower() for name in object_defs} if object_defs else None
        for class_id, name, min_w, min_h, max_w, max_h, class_conf in _DEFAULT_OBJECTS:
            if allowed is not None and name.lower() not in allowed:
                continue
            specs.append((class_id, name, min_w, min_h, max_w, max_h, class_conf))
        if not specs:
            return DetectionResult(detections=[], inference_time_ms=0.0, model_name=self.model_name)

        # Keep the name bytes alive for the duration of the call.
        name_bufs = [name.encode("utf-8") for _, name, *_rest in specs]
        c_defs = (RciObjectDef * len(specs))()
        for index, spec in enumerate(specs):
            class_id, _name, min_w, min_h, max_w, max_h, min_conf = spec
            c_defs[index].id = class_id
            c_defs[index].name = name_bufs[index]
            c_defs[index].min_width = min_w
            c_defs[index].min_height = min_h
            c_defs[index].max_width = max_w
            c_defs[index].max_height = max_h
            c_defs[index].min_confidence = min_conf

        c_image = RciImage(
            data=rgb.ctypes.data_as(POINTER(c_uint8)),
            width=width,
            height=height,
            stride=0,
            format=RCI_PIXFMT_RGB8,
        )
        params = RciDetectParams()
        self._lib.rci_detect_params_init(ctypes.byref(params))
        capacity = max(int(params.max_detections), 32)
        c_out = (RciDetection * capacity)()
        out_count = c_size_t(0)
        stats = RciDetectStats()

        rc = self._lib.rci_detect_profiled(
            ctypes.byref(c_image),
            c_defs,
            c_size_t(len(specs)),
            ctypes.byref(params),
            c_out,
            c_size_t(capacity),
            ctypes.byref(out_count),
            ctypes.byref(stats),
        )
        if rc != 0:
            raise RuntimeError(f"rci_detect failed ({rc}: {_ERROR_TEXT.get(rc, 'unknown')})")

        self.last_stats = {
            "candidate_count": int(stats.candidate_count),
            "verified_count": int(stats.verified_count),
            "kept_count": int(stats.kept_count),
            "propose_ms": float(stats.propose_ms),
            "verify_ms": float(stats.verify_ms),
            "nms_ms": float(stats.nms_ms),
        }

        detections: List[Detection] = []
        for index in range(int(out_count.value)):
            raw = c_out[index]
            class_name = raw.type_name.split(b"\0", 1)[0].decode("utf-8", errors="replace")
            if float(raw.confidence) < conf_threshold:
                continue
            detections.append(
                Detection(
                    object_id=int(raw.id),
                    class_name=class_name,
                    x=float(raw.x),
                    y=float(raw.y),
                    width=float(raw.width),
                    height=float(raw.height),
                    confidence=float(raw.confidence),
                    class_id=int(raw.type),
                )
            )

        elapsed = float(stats.propose_ms + stats.verify_ms + stats.nms_ms)
        return DetectionResult(
            detections=detections,
            inference_time_ms=elapsed,
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
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        result = self.detect(
            image,
            object_defs=object_defs,
            conf_threshold=conf_threshold,
            ir_method=ir_method,
        )
        result.image_path = str(path)
        return result
