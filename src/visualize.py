"""
Visualization utilities for detection results.
"""

from __future__ import annotations

from typing import List, Tuple
import cv2
import numpy as np

from .detector import Detection, DetectionResult


_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 128),
    (0, 128, 255),
]


def _get_color(class_id: int | None) -> Tuple[int, int, int]:
    if class_id is None:
        return (0, 255, 0)
    return _COLORS[class_id % len(_COLORS)]


def draw_detections(
    image: np.ndarray,
    result: DetectionResult,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
        if vis.dtype != np.uint8:
            vis = cv2.normalize(vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    for det in result.detections:
        x1 = int(det.x)
        y1 = int(det.y)
        x2 = int(det.x + det.width)
        y2 = int(det.y + det.height)

        color = _get_color(det.class_id)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        cv2.rectangle(
            vis,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            vis,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return vis


def print_detections_table(result: DetectionResult) -> None:
    print(f"\nModel          : {result.model_name}")
    print(f"Inference time : {result.inference_time_ms:.1f} ms")
    print(f"Detections     : {len(result.detections)}")
    print("-" * 72)
    print(f"{'ID':<4} {'Class':<18} {'Conf':>6}  {'x':>7} {'y':>7} {'w':>7} {'h':>7}")
    print("-" * 72)
    for d in result.detections:
        print(
            f"{d.object_id:<4} {d.class_name:<18} {d.confidence:6.3f}  "
            f"{d.x:7.1f} {d.y:7.1f} {d.width:7.1f} {d.height:7.1f}"
        )
    print("-" * 72)
    