"""Top-level orchestrator that runs the full extraction pipeline end-to-end."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .pdf_analysis import analyze_pdf
from .ocr_engine import OCREngine
from .vlm_engine import VLMEngine
from .reconstruction import reconstruct_paper
from .validation import validate_extraction
from .utils import ExtractionResult, PaperMeta, get_logger, make_run_id

log = get_logger("zimprep.run")


@dataclass
class PipelineOptions:
    use_vlm: bool = False
    use_fallback_ocr: bool = True
    use_gpu: bool = False
    force_ocr: bool = False
    vlm_model: Optional[str] = None
    subject: str = ""
    paper: str = ""
    year: str = ""
    level: str = ""
    board: str = ""


ProgressFn = Callable[[str, float], None]


def run_pipeline(
    pdf_path: str | Path,
    options: Optional[PipelineOptions] = None,
    *,
    progress: Optional[ProgressFn] = None,
    ocr_engine: Optional[OCREngine] = None,
    vlm_engine: Optional[VLMEngine] = None,
) -> ExtractionResult:
    options = options or PipelineOptions()

    def report(msg: str, frac: float):
        log.info(msg)
        if progress:
            try:
                progress(msg, frac)
            except Exception:
                pass

    run_id = make_run_id()
    report(f"Analysing PDF ({Path(pdf_path).name})...", 0.05)
    pages = analyze_pdf(pdf_path, run_id=run_id)
    report(f"PDF analysed ({len(pages)} pages).", 0.20)

    engine = ocr_engine or OCREngine(
        use_gpu=options.use_gpu,
        enable_fallback=options.use_fallback_ocr,
    )
    for i, page in enumerate(pages):
        engine.run_page(page, force_ocr=options.force_ocr)
        report(
            f"OCR page {page.page_number}/{len(pages)} "
            f"(conf={page.ocr_confidence:.2f}{', fallback' if page.used_fallback else ''}).",
            0.20 + 0.45 * (i + 1) / max(len(pages), 1),
        )

    vlm_results = None
    if options.use_vlm:
        vlm = vlm_engine or VLMEngine(model_id=options.vlm_model)
        report("Running VLM structural pass...", 0.70)
        # Only run VLM on pages flagged as ambiguous: low conf, or had a fallback.
        target_pages = [
            p for p in pages
            if p.used_fallback or p.ocr_confidence < 0.80 or p.has_table or p.diagrams
        ] or pages
        vlm_results = vlm.analyze_pages(target_pages)
        report(f"VLM finished ({len(vlm_results)} pages analysed).", 0.80)

    report("Reconstructing question hierarchy...", 0.85)
    meta = PaperMeta(
        subject=options.subject,
        paper=options.paper,
        year=options.year,
        level=options.level,
        board=options.board,
    ) if any([options.subject, options.paper, options.year, options.level, options.board]) else None

    result = reconstruct_paper(pages, vlm_results=vlm_results, meta_override=meta)

    report("Validating extraction...", 0.95)
    result = validate_extraction(result)
    result.stats["run_id"] = run_id

    report(
        f"Done: {len(result.questions)} questions | "
        f"flagged={result.stats.get('flagged_questions', 0)}.",
        1.0,
    )
    return result
