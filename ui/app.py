"""Gradio UI for the ZimPrep exam paper extraction platform.

Layout:
    LEFT   — upload + paper metadata + extraction settings + start button
    CENTER — extraction logs + per-page previews
    RIGHT  — question review + inline edit + approve/reject + export

Launches with `demo.launch(share=True)` from `run.py` or the Colab notebook.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from pipeline import (
    ExtractionResult,
    OCREngine,
    PipelineOptions,
    VLMEngine,
    get_logger,
    run_pipeline,
    save_export,
    to_export_dict,
    workspace_path,
)
from pipeline.utils import Question, SubQuestion


log = get_logger("zimprep.ui")

SUBJECTS = [
    "Mathematics", "English", "Biology", "Chemistry", "Physics",
    "Combined Science", "Geography", "History", "Accounting", "Economics",
    "Commerce", "Business Studies", "Computing", "Computer Science",
    "Agriculture", "Shona", "Ndebele",
]

LEVELS = ["Grade 7", "O-Level", "A-Level", "ZJC", "Other"]

EXTRACTION_MODES = [
    ("Fast (OCR only)", "fast"),
    ("Balanced (OCR + fallback)", "balanced"),
    ("Accurate (OCR + fallback + VLM)", "accurate"),
]


# --- Shared engines (loaded once, reused across requests) -------------------

_ENGINE_LOCK = threading.Lock()
_OCR: Optional[OCREngine] = None
_VLM: Optional[VLMEngine] = None


def _get_ocr(use_gpu: bool, enable_fallback: bool) -> OCREngine:
    global _OCR
    with _ENGINE_LOCK:
        if _OCR is None:
            _OCR = OCREngine(use_gpu=use_gpu, enable_fallback=enable_fallback)
        return _OCR


def _get_vlm() -> VLMEngine:
    global _VLM
    with _ENGINE_LOCK:
        if _VLM is None:
            _VLM = VLMEngine()
        return _VLM


# --- Session state ---------------------------------------------------------

@dataclass
class SessionState:
    result: Optional[ExtractionResult] = None
    pdf_name: str = ""
    edits: Dict[str, dict] = field(default_factory=dict)  # path -> override

    def question_choices(self) -> List[str]:
        if not self.result:
            return []
        return [
            f"Q{q.question_number} — {(q.text or '').strip()[:90] or '(empty)'}"
            for q in self.result.questions
        ]

    def find_question(self, label: str) -> Optional[Question]:
        if not self.result or not label:
            return None
        try:
            num = int(label.split("Q", 1)[1].split(" ", 1)[0])
        except (IndexError, ValueError):
            return None
        for q in self.result.questions:
            if q.question_number == num:
                return q
        return None


# --- Log capture ------------------------------------------------------------

class _UILogHandler(logging.Handler):
    def __init__(self, sink: io.StringIO):
        super().__init__(level=logging.INFO)
        self.sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.sink.write(self.format(record) + "\n")
        except Exception:
            pass


def _attach_log_sink(sink: io.StringIO) -> _UILogHandler:
    handler = _UILogHandler(sink)
    for name in ("zimprep", "zimprep.pdf", "zimprep.ocr", "zimprep.pre",
                 "zimprep.recon", "zimprep.val", "zimprep.vlm", "zimprep.run"):
        logger = logging.getLogger(name)
        logger.addHandler(handler)
    return handler


def _detach_log_sink(handler: _UILogHandler):
    for name in ("zimprep", "zimprep.pdf", "zimprep.ocr", "zimprep.pre",
                 "zimprep.recon", "zimprep.val", "zimprep.vlm", "zimprep.run"):
        logger = logging.getLogger(name)
        try:
            logger.removeHandler(handler)
        except Exception:
            pass


# --- Helpers ---------------------------------------------------------------

def _format_question_markdown(q: Question) -> str:
    lines: List[str] = []
    lines.append(f"### Question {q.question_number}  ·  marks: **{q.marks if q.marks is not None else '—'}**  ·  type: `{q.question_type}`  ·  conf: `{q.confidence:.2f}`")
    if q.page:
        lines.append(f"_page {q.page}_")
    if q.issues:
        lines.append("**Issues:** " + ", ".join(f"`{i}`" for i in q.issues))
    if q.text:
        lines.append("")
        lines.append(q.text)
    for sub in q.subquestions:
        lines.append("")
        lines.append(_format_sub_markdown(sub, depth=1))
    return "\n".join(lines)


def _format_sub_markdown(sub: SubQuestion, depth: int = 1) -> str:
    indent = "  " * (depth - 1)
    head = f"{indent}**({sub.label})**"
    if sub.marks is not None:
        head += f"  _[{sub.marks}]_"
    body = f"{head}  {sub.text}".strip()
    out = [body]
    for child in sub.subquestions:
        out.append(_format_sub_markdown(child, depth=depth + 1))
    if sub.issues:
        out.append(f"{indent}_issues: {', '.join(sub.issues)}_")
    return "\n".join(out)


def _question_table(result: Optional[ExtractionResult]):
    if not result:
        return []
    rows = []
    for q in result.questions:
        rows.append([
            q.question_number,
            (q.text[:120] + "…") if len(q.text) > 120 else q.text,
            q.marks if q.marks is not None else "",
            q.question_type,
            len(q.subquestions),
            round(q.confidence, 2),
            ", ".join(q.issues) if q.issues else "",
            "✓" if q.approved else ("✗" if q.approved is False else ""),
        ])
    return rows


def _diagrams_for_question(q: Optional[Question], result: Optional[ExtractionResult]) -> List[str]:
    if not q or not result:
        return []
    refs = set(q.image_refs)
    for s in q.subquestions:
        refs.update(s.image_refs)
        for r in s.subquestions:
            refs.update(r.image_refs)
    return [d.image_path for d in result.diagrams if d.ref_id in refs and d.image_path]


# --- Main extraction handler ----------------------------------------------

def _run_extraction(
    pdf_file,
    subject: str,
    level: str,
    paper_label: str,
    year: str,
    board: str,
    mode_label: str,
    force_ocr: bool,
    use_gpu: bool,
    state: SessionState,
    progress=gr.Progress(track_tqdm=False),
):
    if pdf_file is None:
        return (
            "Please upload a PDF first.",
            "",
            [],
            gr.update(choices=[]),
            state,
            None,
        )

    sink = io.StringIO()
    handler = _attach_log_sink(sink)

    try:
        pdf_path = Path(pdf_file.name if hasattr(pdf_file, "name") else pdf_file)
        target = workspace_path("uploads", pdf_path.name)
        if str(pdf_path) != str(target):
            shutil.copy(pdf_path, target)
        state.pdf_name = target.name

        mode = next((v for label, v in EXTRACTION_MODES if label == mode_label), "balanced")
        options = PipelineOptions(
            use_vlm=(mode == "accurate"),
            use_fallback_ocr=(mode in ("balanced", "accurate")),
            use_gpu=use_gpu,
            force_ocr=force_ocr,
            subject=subject,
            level=level,
            paper=paper_label,
            year=year,
            board=board,
        )

        def _progress(msg: str, frac: float):
            progress(frac, desc=msg)

        ocr_engine = _get_ocr(use_gpu=use_gpu, enable_fallback=options.use_fallback_ocr)
        vlm_engine = _get_vlm() if options.use_vlm else None

        result = run_pipeline(
            target, options,
            progress=_progress,
            ocr_engine=ocr_engine,
            vlm_engine=vlm_engine,
        )
        state.result = result
        state.edits = {}

        summary = (
            f"**{state.pdf_name}** — {len(result.questions)} questions "
            f"({result.stats.get('flagged_questions', 0)} flagged)  ·  "
            f"avg OCR conf: {result.stats.get('avg_ocr_confidence', 0):.2f}"
        )
        if result.warnings:
            summary += "\n\n**Warnings:** " + ", ".join(result.warnings)

        return (
            summary,
            sink.getvalue(),
            _question_table(result),
            gr.update(choices=state.question_choices(), value=(state.question_choices()[0] if result.questions else None)),
            state,
            _page_previews(result),
        )
    except Exception as e:
        log.exception("Extraction failed: %s", e)
        return (
            f"Extraction failed: `{e}`",
            sink.getvalue() + f"\nERROR: {e}",
            [],
            gr.update(choices=[]),
            state,
            None,
        )
    finally:
        _detach_log_sink(handler)


def _page_previews(result: ExtractionResult) -> List[str]:
    return [p.image_path for p in result.pages if p.image_path]


# --- Review-panel handlers -------------------------------------------------

def _on_select_question(label: str, state: SessionState):
    q = state.find_question(label)
    if q is None:
        return (
            "_No question selected._",
            "",
            None,
            None,
            "",
            [],
        )
    return (
        _format_question_markdown(q),
        q.text,
        q.marks if q.marks is not None else 0,
        q.question_type,
        ", ".join(q.issues),
        _diagrams_for_question(q, state.result),
    )


def _save_edit(label: str, new_text: str, new_marks, new_type: str, state: SessionState):
    q = state.find_question(label)
    if q is None:
        return state, "No question selected."
    q.text = (new_text or "").strip()
    if new_marks is None or (isinstance(new_marks, str) and not new_marks.strip()):
        q.marks = None
    else:
        try:
            q.marks = int(new_marks)
        except (TypeError, ValueError):
            pass
    if new_type:
        q.question_type = new_type
    return state, f"Saved Q{q.question_number}."


def _approve(label: str, state: SessionState):
    q = state.find_question(label)
    if q is None:
        return state, "No question selected.", None
    q.approved = True
    return state, f"Approved Q{q.question_number}.", _question_table(state.result)


def _reject(label: str, state: SessionState):
    q = state.find_question(label)
    if q is None:
        return state, "No question selected.", None
    q.approved = False
    return state, f"Rejected Q{q.question_number}.", _question_table(state.result)


def _export(state: SessionState):
    if not state.result:
        return None, "Nothing to export — run an extraction first."
    out_path = save_export(state.result, filename=f"{Path(state.pdf_name).stem}.json")
    preview = json.dumps(to_export_dict(state.result), indent=2, ensure_ascii=False)
    return str(out_path), preview


# --- UI construction --------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="ZimPrep — Exam Paper Extractor",
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        css="""
        #left-col, #center-col, #right-col { min-height: 720px; }
        .small-note { font-size: 12px; color: #64748b; }
        """,
    ) as demo:
        gr.Markdown(
            "## ZimPrep Exam Paper Extractor\n"
            "_Upload a paper, extract structured questions, review, and export production-ready JSON._"
        )

        state = gr.State(SessionState())

        with gr.Row():
            # ----------- LEFT ----------------------------------------------
            with gr.Column(scale=3, elem_id="left-col"):
                gr.Markdown("### 1. Upload")
                pdf_in = gr.File(label="Exam paper (PDF)", file_types=[".pdf"])
                subject = gr.Dropdown(SUBJECTS, label="Subject", allow_custom_value=True)
                level = gr.Dropdown(LEVELS, label="Level", value="O-Level", allow_custom_value=True)
                board = gr.Dropdown(["ZIMSEC", "Cambridge", "Other"], label="Board", value="ZIMSEC", allow_custom_value=True)
                paper_label = gr.Textbox(label="Paper", placeholder="e.g. Paper 1")
                year = gr.Textbox(label="Year", placeholder="e.g. 2023")

                gr.Markdown("### 2. Extraction settings")
                mode = gr.Radio(
                    choices=[label for label, _ in EXTRACTION_MODES],
                    label="Mode",
                    value="Balanced (OCR + fallback)",
                )
                force_ocr = gr.Checkbox(label="Force OCR (ignore native PDF text)", value=False)
                use_gpu = gr.Checkbox(label="Use GPU if available", value=True)

                run_btn = gr.Button("Start extraction", variant="primary")
                summary = gr.Markdown("_Awaiting upload._")

            # ----------- CENTER --------------------------------------------
            with gr.Column(scale=5, elem_id="center-col"):
                gr.Markdown("### Extraction log")
                logs = gr.Code(label="Pipeline log", language=None, lines=18, interactive=False)
                gr.Markdown("### Page previews")
                previews = gr.Gallery(label="Pages", columns=3, height=320, preview=True)
                gr.Markdown("### Question overview")
                qtable = gr.Dataframe(
                    headers=["#", "Stem", "Marks", "Type", "Subparts", "Conf", "Issues", "Approval"],
                    datatype=["number", "str", "str", "str", "number", "number", "str", "str"],
                    interactive=False,
                    wrap=True,
                    row_count=(0, "dynamic"),
                )

            # ----------- RIGHT ---------------------------------------------
            with gr.Column(scale=4, elem_id="right-col"):
                gr.Markdown("### 3. Review")
                q_selector = gr.Dropdown(
                    label="Question",
                    choices=[],
                    interactive=True,
                )
                q_preview = gr.Markdown("_Select a question to review._")
                q_diagrams = gr.Gallery(label="Associated diagrams", columns=2, height=180)

                with gr.Accordion("Edit question", open=True):
                    edit_text = gr.Textbox(label="Stem text", lines=6)
                    with gr.Row():
                        edit_marks = gr.Number(label="Marks", precision=0)
                        edit_type = gr.Dropdown(
                            ["structured", "mcq", "essay", "instruction"],
                            label="Type",
                        )
                    edit_issues = gr.Textbox(label="Issues (read-only)", interactive=False)
                    with gr.Row():
                        save_btn = gr.Button("Save edits")
                        approve_btn = gr.Button("Approve", variant="primary")
                        reject_btn = gr.Button("Reject", variant="stop")
                    edit_status = gr.Markdown("")

                gr.Markdown("### 4. Export")
                export_btn = gr.Button("Export JSON", variant="primary")
                export_file = gr.File(label="Download JSON")
                export_preview = gr.Code(label="Preview", language="json", lines=12)

        # ----- wiring -------------------------------------------------------
        run_btn.click(
            _run_extraction,
            inputs=[pdf_in, subject, level, paper_label, year, board, mode, force_ocr, use_gpu, state],
            outputs=[summary, logs, qtable, q_selector, state, previews],
        )

        q_selector.change(
            _on_select_question,
            inputs=[q_selector, state],
            outputs=[q_preview, edit_text, edit_marks, edit_type, edit_issues, q_diagrams],
        )

        save_btn.click(
            _save_edit,
            inputs=[q_selector, edit_text, edit_marks, edit_type, state],
            outputs=[state, edit_status],
        ).then(
            _on_select_question,
            inputs=[q_selector, state],
            outputs=[q_preview, edit_text, edit_marks, edit_type, edit_issues, q_diagrams],
        ).then(
            lambda s: _question_table(s.result),
            inputs=[state],
            outputs=[qtable],
        )

        approve_btn.click(
            _approve,
            inputs=[q_selector, state],
            outputs=[state, edit_status, qtable],
        )
        reject_btn.click(
            _reject,
            inputs=[q_selector, state],
            outputs=[state, edit_status, qtable],
        )

        export_btn.click(
            _export,
            inputs=[state],
            outputs=[export_file, export_preview],
        )

    return demo


def launch(share: bool = True, server_name: str = "0.0.0.0", server_port: int = 7860):
    demo = build_demo()
    demo.queue(max_size=8).launch(
        share=share,
        server_name=server_name,
        server_port=server_port,
        show_error=True,
    )


if __name__ == "__main__":
    launch()
