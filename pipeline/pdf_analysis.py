"""PDF analysis: classify pages, render images, extract native text and diagrams."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .utils import (
    DiagramRegion,
    PageData,
    get_logger,
    workspace_path,
    make_run_id,
)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False


log = get_logger("zimprep.pdf")

# Heuristic: pages with fewer characters per cm^2 than this are likely scanned.
TEXT_DENSITY_THRESHOLD = 0.05  # chars per pixel-area (very low for native text)
SCANNED_MIN_TEXT_CHARS = 40
DEFAULT_DPI = 300


def render_pdf_pages(pdf_path: str | Path, run_id: str | None = None, dpi: int = DEFAULT_DPI) -> List[Path]:
    """Render each PDF page to a PNG and return the file paths."""
    pdf_path = Path(pdf_path)
    run_id = run_id or make_run_id()
    out_dir = workspace_path("images", run_id)
    paths: List[Path] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            zoom = dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out_path = out_dir / f"page_{i + 1:03d}.png"
            pix.save(out_path)
            paths.append(out_path)
    return paths


def _merge_boxes(boxes: List[Tuple[float, float, float, float]], threshold: float = 40.0) -> List[Tuple[float, float, float, float]]:
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        n = len(merged)
        for i in range(n):
            for j in range(i + 1, n):
                b1 = merged[i]
                b2 = merged[j]
                bloated1 = (b1[0] - threshold/2, b1[1] - threshold/2, b1[2] + threshold/2, b1[3] + threshold/2)
                bloated2 = (b2[0] - threshold/2, b2[1] - threshold/2, b2[2] + threshold/2, b2[3] + threshold/2)
                overlap_x = max(bloated1[0], bloated2[0]) < min(bloated1[2], bloated2[2])
                overlap_y = max(bloated1[1], bloated2[1]) < min(bloated1[3], bloated2[3])
                if overlap_x and overlap_y:
                    new_box = (
                        min(b1[0], b2[0]),
                        min(b1[1], b2[1]),
                        max(b1[2], b2[2]),
                        max(b1[3], b2[3])
                    )
                    merged[i] = new_box
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    return merged


def _detect_diagrams(
    page: fitz.Page,
    page_number: int,
    run_id: str,
    page_img_path: Path,
    zoom: float,
) -> List[DiagramRegion]:
    """Extract embedded images and figure-like vector drawing regions from a page by cropping."""
    diagrams: List[DiagramRegion] = []
    page_width = page.rect.width
    page_height = page.rect.height
    
    raw_boxes: List[Tuple[float, float, float, float]] = []
    
    # 1. Collect embedded raster images bounding boxes
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        rects = page.get_image_rects(xref)
        if rects:
            rect = rects[0]
            raw_boxes.append((rect.x0, rect.y0, rect.x1, rect.y1))
            
    # 2. Collect vector drawings bounding boxes
    for d in page.get_drawings():
        rect = d.get("rect")
        if rect:
            raw_boxes.append((rect.x0, rect.y0, rect.x1, rect.y1))
            
    # 3. Clean and filter raw bounding boxes
    filtered_boxes: List[Tuple[float, float, float, float]] = []
    for box in raw_boxes:
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        
        if w <= 0 or h <= 0:
            continue
            
        # Ignore horizontal dividing rules
        if w > page_width * 0.8 and h < 8.0:
            continue
            
        # Ignore vertical dividing rules
        if h > page_height * 0.8 and w < 8.0:
            continue
            
        # Ignore background page frames
        if w > page_width * 0.95 and h > page_height * 0.95:
            continue
            
        # Ignore very small coordinate noise
        if w < 10.0 and h < 10.0:
            continue
            
        # Constrain coordinates to page bounds
        x0 = max(0.0, min(x0, page_width))
        y0 = max(0.0, min(y0, page_height))
        x1 = max(0.0, min(x1, page_width))
        y1 = max(0.0, min(y1, page_height))
        
        filtered_boxes.append((x0, y0, x1, y1))
        
    # 4. Merge overlapping or very close diagram boxes
    merged_boxes = _merge_boxes(filtered_boxes, threshold=40.0)
    if not merged_boxes:
        return []
        
    # 5. Crop grouped boxes from the high-res rendered page image
    out_dir = workspace_path("images", run_id, "diagrams")
    try:
        img = Image.open(page_img_path)
    except Exception as e:
        log.warning("Could not open page image for cropping: %s", e)
        return []
        
    for idx, box in enumerate(merged_boxes):
        x0, y0, x1, y1 = box
        px0 = max(0, int(x0 * zoom))
        py0 = max(0, int(y0 * zoom))
        px1 = min(img.width, int(x1 * zoom))
        py1 = min(img.height, int(y1 * zoom))
        
        if px1 <= px0 or py1 <= py0:
            continue
            
        out_path = out_dir / f"p{page_number:03d}_d{idx + 1}.png"
        try:
            cropped = img.crop((px0, py0, px1, py1))
            cropped.save(out_path, "PNG")
            diagrams.append(
                DiagramRegion(
                    page=page_number,
                    bbox=(x0, y0, x1, y1),
                    image_path=str(out_path),
                )
            )
        except Exception as e:
            log.warning("Cropping failed for box %s: %s", box, e)
            
    return diagrams


def _page_text_density(page: fitz.Page) -> Tuple[str, float]:
    text = page.get_text("text") or ""
    area = max(page.rect.width * page.rect.height, 1.0)
    density = len(text.strip()) / area
    return text, density


def _has_tables(page: fitz.Page, plumber_page=None) -> bool:
    """Best-effort table detection via pdfplumber or stroke heuristics."""
    if plumber_page is not None:
        try:
            tables = plumber_page.find_tables()
            if tables:
                return True
        except Exception:
            pass
    drawings = page.get_drawings()
    horiz = vert = 0
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 1:
                    horiz += 1
                elif abs(p1.x - p2.x) < 1:
                    vert += 1
    return horiz >= 3 and vert >= 3


def analyze_pdf(pdf_path: str | Path, run_id: str | None = None, dpi: int = DEFAULT_DPI) -> List[PageData]:
    """Open a PDF, render each page, and return per-page metadata."""
    pdf_path = Path(pdf_path)
    run_id = run_id or make_run_id()
    img_dir = workspace_path("images", run_id)

    log.info("Analysing PDF %s (dpi=%d)", pdf_path.name, dpi)

    pages: List[PageData] = []
    plumber_doc = None
    if HAS_PDFPLUMBER:
        try:
            plumber_doc = pdfplumber.open(str(pdf_path))
        except Exception as e:
            log.warning("pdfplumber failed to open file: %s", e)
            plumber_doc = None

    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                page_no = i + 1
                native_text, density = _page_text_density(page)
                is_scanned = (
                    len(native_text.strip()) < SCANNED_MIN_TEXT_CHARS
                    or density < TEXT_DENSITY_THRESHOLD
                )

                zoom = dpi / 72
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img_path = img_dir / f"page_{page_no:03d}.png"
                pix.save(img_path)

                pl_page = None
                if plumber_doc is not None and i < len(plumber_doc.pages):
                    pl_page = plumber_doc.pages[i]

                diagrams = _detect_diagrams(page, page_no, run_id, img_path, zoom)
                has_table = _has_tables(page, pl_page)

                pages.append(
                    PageData(
                        page_number=page_no,
                        width=pix.width,
                        height=pix.height,
                        is_scanned=is_scanned,
                        text_density=float(density),
                        native_text=native_text,
                        image_path=str(img_path),
                        diagrams=diagrams,
                        has_table=has_table,
                    )
                )
                log.info(
                    "  page %d: %s | density=%.4f | diagrams=%d | tables=%s",
                    page_no,
                    "scanned" if is_scanned else "native",
                    density,
                    len(diagrams),
                    has_table,
                )
    finally:
        if plumber_doc is not None:
            try:
                plumber_doc.close()
            except Exception:
                pass

    return pages


def detect_scanned_diagrams(page: PageData, run_id: str) -> List[DiagramRegion]:
    """Fallback diagram detector for scanned pages. Runs AFTER OCR.
    Blanks out OCR text blocks, groups the remaining ink via morph operations,
    and crops the diagram regions from the page image."""
    import cv2
    diagrams: List[DiagramRegion] = []
    
    img_path = page.preprocessed_path or page.image_path
    if not img_path or not Path(img_path).exists():
        return []
        
    img = cv2.imread(str(img_path))
    if img is None:
        return []
        
    img_h, img_w = img.shape[:2]
    
    # Threshold the page to binary (ink pixels are 255)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    
    # Blank out all OCR text boxes to isolate diagrams
    for b in page.blocks:
        x0, y0, x1, y1 = b.bbox
        # Bloat slightly to erase trailing/leading punctuation and accents
        bx0 = max(0, int(x0) - 8)
        by0 = max(0, int(y0) - 8)
        bx1 = min(img_w, int(x1) + 8)
        by1 = min(img_h, int(y1) + 8)
        cv2.rectangle(thresh, (bx0, by0), (bx1, by1), 0, -1)
        
    # Run morphological dilation to merge close line fragments into solid boxes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    thresh_dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    out_dir = workspace_path("images", run_id, "diagrams")
    
    valid_boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Filter rules
        if w < 40 or h < 40:
            continue
        if w > img_w * 0.9 and h > img_h * 0.9:
            continue
        if w > img_w * 0.8 and h < 15:
            # horizontal rule
            continue
        if h > img_h * 0.8 and w < 15:
            # vertical rule
            continue
            
        valid_boxes.append((x, y, x + w, y + h))
        
    # Merge overlapping/close diagram regions
    float_boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in valid_boxes]
    merged_boxes = _merge_boxes(float_boxes, threshold=40.0)
    
    # Save cropped images
    try:
        pil_img = Image.open(img_path)
    except Exception as e:
        log.warning("Could not open page image for scanned cropping: %s", e)
        return []
        
    for idx, box in enumerate(merged_boxes):
        x0, y0, x1, y1 = box
        px0 = max(0, int(x0))
        py0 = max(0, int(y0))
        px1 = min(img_w, int(x1))
        py1 = min(img_h, int(y1))
        
        if px1 <= px0 or py1 <= py0:
            continue
            
        out_path = out_dir / f"p{page.page_number:03d}_sc_d{idx + 1}.png"
        try:
            cropped = pil_img.crop((px0, py0, px1, py1))
            cropped.save(out_path, "PNG")
            
            diagrams.append(
                DiagramRegion(
                    page=page.page_number,
                    bbox=(x0, y0, x1, y1),
                    image_path=str(out_path),
                )
            )
        except Exception as e:
            log.warning("Scanned cropping failed: %s", e)
            
    return diagrams
