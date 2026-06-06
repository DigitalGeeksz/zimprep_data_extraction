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


def _detect_diagrams(page: fitz.Page, page_number: int, run_id: str) -> List[DiagramRegion]:
    """Extract embedded images and figure-like regions from a page."""
    diagrams: List[DiagramRegion] = []
    out_dir = workspace_path("images", run_id, "diagrams")
    for idx, img_info in enumerate(page.get_images(full=True)):
        xref = img_info[0]
        try:
            base = page.parent.extract_image(xref)
        except Exception:
            continue
        ext = base.get("ext", "png")
        img_bytes = base.get("image")
        if not img_bytes:
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        rect = rects[0]
        out_path = out_dir / f"p{page_number:03d}_d{idx + 1}.{ext}"
        out_path.write_bytes(img_bytes)
        diagrams.append(
            DiagramRegion(
                page=page_number,
                bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                image_path=str(out_path),
            )
        )
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

                diagrams = _detect_diagrams(page, page_no, run_id)
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
