"""
Infrared image preprocessing utilities for RCO EOI Problem #6.

Handles:
- 8/10/12/14/16-bit IR images
- Grayscale thermal data
- Normalization strategies suitable for detection models
- Optional CLAHE for contrast enhancement
"""

from __future__ import annotations

from typing import Optional
import cv2
import numpy as np


def normalize_ir(
    image: np.ndarray,
    method: str = "minmax",
    bit_depth: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize an IR image to uint8 [0, 255] range.

    Parameters
    ----------
    image : np.ndarray
        Input IR image (2D or 3D). Can be uint8/uint16/float.
    method : str
        "minmax" | "percentile" | "clahe"
    bit_depth : int, optional
        If known (e.g. 14), can be used for more accurate scaling.

    Returns
    -------
    np.ndarray
        uint8 image ready for detection models.
    """
    img = image.copy()

    if img.dtype == np.uint8:
        return img

    img = img.astype(np.float32)

    if method == "minmax":
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val) * 255.0
        else:
            img = np.zeros_like(img)

    elif method == "percentile":
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip(img, p2, p98)
        if p98 > p2:
            img = (img - p2) / (p98 - p2) * 255.0
        else:
            img = np.zeros_like(img)

    elif method == "clahe":
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip(img, p2, p98)
        if p98 > p2:
            img = (img - p2) / (p98 - p2) * 255.0
        img = img.astype(np.uint8)
        if len(img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        return img

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return np.clip(img, 0, 255).astype(np.uint8)


def prepare_for_detector(
    image: np.ndarray,
    to_3channel: bool = True,
    ir_method: str = "percentile",
) -> np.ndarray:
    """
    Full preprocessing pipeline for feeding an IR or visible image into a detector.
    """
    if image.dtype != np.uint8:
        image = normalize_ir(image, method=ir_method)

    if to_3channel:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement (useful for low-contrast IR)."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)
        