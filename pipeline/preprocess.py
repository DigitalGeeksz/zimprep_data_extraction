"""Image preprocessing for OCR: deskew, denoise, CLAHE, thresholding, sharpen."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .utils import get_logger, workspace_path

log = get_logger("zimprep.pre")


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Find dominant skew angle from non-white pixels and rotate to correct."""
    inv = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inv > 50))
    if coords.shape[0] < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.2:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, kernel)


def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15,
    )


def _maybe_upscale(gray: np.ndarray, min_height: int = 1600) -> np.ndarray:
    h, w = gray.shape[:2]
    if h >= min_height:
        return gray
    scale = min_height / h
    return cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def preprocess_image(
    image_path: str | Path,
    out_path: Optional[str | Path] = None,
    *,
    binarize: bool = False,
    upscale: bool = True,
    apply_clahe: bool = True,
    apply_denoise: bool = True,
    apply_sharpen: bool = True,
) -> Path:
    """Run the preprocessing pipeline on a single image and return the output path."""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if upscale:
        gray = _maybe_upscale(gray)
    if apply_denoise:
        gray = _denoise(gray)
    gray = _deskew(gray)
    if apply_clahe:
        gray = _clahe(gray)
    if apply_sharpen:
        gray = _sharpen(gray)
    if binarize:
        gray = _adaptive_threshold(gray)

    if out_path is None:
        out_dir = workspace_path("images", "preprocessed")
        out_path = out_dir / f"pp_{image_path.stem}.png"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), gray)
    return out_path
