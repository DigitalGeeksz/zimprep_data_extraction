"""Image preprocessing for OCR: deskew, denoise, CLAHE, thresholding, sharpen."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .utils import get_logger, workspace_path

log = get_logger("zimprep.pre")


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Find dominant skew angle from text lines and rotate to correct."""
    inv = cv2.bitwise_not(gray)
    
    # Resize to speed up morphological operations and contour detection
    h, w = gray.shape[:2]
    scale = 1.0
    if h > 1500:
        scale = 1500.0 / h
        gray_small = cv2.resize(inv, (int(w * scale), 1500), interpolation=cv2.INTER_AREA)
    else:
        gray_small = inv

    # Dilate horizontally to merge characters into text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(gray_small, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    angles = []
    for c in contours:
        # Scale back the contour points to original dimensions if we scaled down
        if scale != 1.0:
            c = (c / scale).astype(np.int32)
            
        rect = cv2.minAreaRect(c)
        box_w, box_h = rect[1]
        
        # Only consider contours that are clearly wide text lines
        if box_w > box_h * 2.5 and box_w > 40:
            angle = rect[2]
            # Normalize angle to [-45, 45] range
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            angles.append(angle)

    if not angles:
        return gray
        
    median_angle = float(np.median(angles))
    
    # Capping Safety filter: only correct minor scanner skews (e.g., within [-10, 10] degrees).
    # If the angle is larger, it is likely a false positive skew calculation due to vertical borders/noise.
    if abs(median_angle) < 10.0 and abs(median_angle) >= 0.1:
        log.info("  Calculated page skew: %.2f degrees. Applying rotation.", median_angle)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        if abs(median_angle) >= 10.0:
            log.info("  Calculated page skew: %.2f degrees exceeds safety threshold (10 deg). Skipping rotation.", median_angle)
        return gray



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
