"""OCR engine wrapping PaddleOCR (primary) and DeepSeek-OCR (fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from .preprocess import preprocess_image
from .utils import OCRBlock, PageData, get_logger

log = get_logger("zimprep.ocr")


# Confidence below which we re-run with the heavier fallback model.
LOW_CONFIDENCE_THRESHOLD = 0.78
MIN_BLOCKS_FOR_CONFIDENCE = 4


@dataclass
class _OCRResult:
    blocks: List[OCRBlock]
    avg_confidence: float
    source: str


class _PaddleBackend:
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        from paddleocr import PaddleOCR
        log.info("Loading PaddleOCR (lang=%s, gpu=%s)", lang, use_gpu)
        self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu, show_log=False)

    def run(self, image_path: str) -> _OCRResult:
        result = self._ocr.ocr(image_path, cls=True)
        blocks: List[OCRBlock] = []
        confs: List[float] = []
        if result and result[0]:
            for line in result[0]:
                poly, (text, conf) = line[0], line[1]
                if not text or not text.strip():
                    continue
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                bbox = (min(xs), min(ys), max(xs), max(ys))
                blocks.append(OCRBlock(
                    text=text.strip(),
                    confidence=float(conf),
                    bbox=bbox,
                    source="paddle",
                ))
                confs.append(float(conf))
        avg = float(np.mean(confs)) if confs else 0.0
        return _OCRResult(blocks=blocks, avg_confidence=avg, source="paddle")


class _DeepSeekBackend:
    """DeepSeek-OCR via Transformers. Used as fallback for low-confidence pages."""

    MODEL_ID = "deepseek-ai/DeepSeek-OCR"

    def __init__(self, device: Optional[str] = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading DeepSeek-OCR on %s", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(self.device).eval()

    def run(self, image_path: str) -> _OCRResult:
        # DeepSeek-OCR exposes a high-level `infer` API on the model.
        # We treat its output as a single high-confidence block per page;
        # geometry is approximated by image extents.
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        try:
            text = self.model.infer(
                self.tokenizer,
                prompt="<|grounding|>Convert the page to clean markdown.",
                image_file=str(image_path),
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=False,
                test_compress=True,
            )
        except Exception as e:
            log.warning("DeepSeek-OCR failed (%s); returning empty result", e)
            return _OCRResult(blocks=[], avg_confidence=0.0, source="deepseek")

        if not isinstance(text, str):
            text = str(text)

        # Split into pseudo-lines so the reconstruction engine can still process structure.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        blocks: List[OCRBlock] = []
        line_h = max(h / max(len(lines), 1), 20)
        for i, line in enumerate(lines):
            y1 = i * line_h
            blocks.append(OCRBlock(
                text=line,
                confidence=0.92,  # treat fallback as high-confidence
                bbox=(0.0, y1, float(w), y1 + line_h),
                source="deepseek",
            ))
        return _OCRResult(blocks=blocks, avg_confidence=0.92 if blocks else 0.0, source="deepseek")


class OCREngine:
    """High-level OCR engine that orchestrates preprocessing + Paddle + DeepSeek fallback."""

    def __init__(
        self,
        *,
        lang: str = "en",
        use_gpu: bool = False,
        enable_fallback: bool = True,
        paddle_backend: Optional[_PaddleBackend] = None,
        deepseek_backend: Optional[_DeepSeekBackend] = None,
    ):
        self.lang = lang
        self.use_gpu = use_gpu
        self.enable_fallback = enable_fallback
        self._paddle = paddle_backend
        self._deepseek = deepseek_backend

    # Lazy-loaders keep startup snappy when only one backend is needed.
    def _get_paddle(self) -> _PaddleBackend:
        if self._paddle is None:
            self._paddle = _PaddleBackend(lang=self.lang, use_gpu=self.use_gpu)
        return self._paddle

    def _get_deepseek(self) -> _DeepSeekBackend:
        if self._deepseek is None:
            self._deepseek = _DeepSeekBackend()
        return self._deepseek

    def run_page(self, page: PageData, *, force_ocr: bool = False) -> PageData:
        """OCR a single page. Skips OCR for native PDFs unless force_ocr=True."""
        if not force_ocr and not page.is_scanned and page.native_text.strip():
            page.blocks = _native_text_to_blocks(page)
            page.ocr_confidence = 0.99
            return page

        pre_path = preprocess_image(page.image_path)
        page.preprocessed_path = str(pre_path)

        primary = self._get_paddle().run(str(pre_path))
        page.blocks = primary.blocks
        page.ocr_confidence = primary.avg_confidence

        low_conf = (
            primary.avg_confidence < LOW_CONFIDENCE_THRESHOLD
            or len(primary.blocks) < MIN_BLOCKS_FOR_CONFIDENCE
        )
        if self.enable_fallback and low_conf:
            log.info(
                "  page %d: paddle confidence %.2f below threshold; running DeepSeek-OCR",
                page.page_number, primary.avg_confidence,
            )
            try:
                fallback = self._get_deepseek().run(str(pre_path))
                if fallback.blocks:
                    page.blocks = fallback.blocks
                    page.ocr_confidence = fallback.avg_confidence
                    page.used_fallback = True
            except Exception as e:
                log.warning("DeepSeek fallback failed: %s", e)

        for b in page.blocks:
            b.page = page.page_number
        return page

    def run(self, pages: Sequence[PageData], *, force_ocr: bool = False) -> List[PageData]:
        out = []
        for page in pages:
            self.run_page(page, force_ocr=force_ocr)
            log.info(
                "  page %d OCR done: %d blocks | avg_conf=%.2f | fallback=%s",
                page.page_number, len(page.blocks), page.ocr_confidence, page.used_fallback,
            )
            out.append(page)
        return out


def _native_text_to_blocks(page: PageData) -> List[OCRBlock]:
    """Convert PyMuPDF native text into OCRBlocks ordered top-to-bottom."""
    import fitz  # local import to avoid hard dep when reusing PageData elsewhere
    blocks: List[OCRBlock] = []
    try:
        with fitz.open() as _:
            pass
    except Exception:
        pass

    # Re-open underlying PDF? Cheaper: parse the cached native_text into lines
    # using approximate line geometry. We don't have block bboxes here, so we
    # fall back to evenly-spaced y coordinates.
    lines = [ln.strip() for ln in page.native_text.splitlines() if ln.strip()]
    h = float(page.height or 1000)
    w = float(page.width or 1000)
    if not lines:
        return blocks
    step = h / len(lines)
    for i, ln in enumerate(lines):
        y1 = i * step
        blocks.append(OCRBlock(
            text=ln,
            confidence=0.99,
            bbox=(0.0, y1, w, y1 + step),
            source="pdf-native",
            page=page.page_number,
        ))
    return blocks
