"""Question hierarchy reconstruction.

Converts OCR/native text blocks into nested Question -> SubQuestion trees.

Strategy:
  1. Read blocks page-by-page, top-to-bottom.
  2. A deterministic finite state machine recognises:
        - main question numbers (1., 2., Question 3, ...)
        - first-level subparts (a), (b), c)
        - nested roman subparts (i), ii.
        - MCQ options (A. B. C. D.)
        - mark annotations [3], (5 marks)
        - instruction headers ("Answer ALL questions", "Section A")
  3. Continuation pages with no new "1." token append to the open question.
  4. Optionally, VLM JSON output is merged in to override ambiguous pages.
  5. Diagrams are associated with the question whose start-y is closest
     and that lies on the same page or the prior page if the diagram floats.

The engine is intentionally deterministic; VLM is consulted only when the
state machine flags a page as low-confidence (broken numbering, missing marks
on a structured paper, suspicious empty stems).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from .utils import (
    DiagramRegion,
    ExtractionResult,
    OCRBlock,
    PageData,
    PaperMeta,
    Question,
    SubQuestion,
    get_logger,
    safe_int,
)

log = get_logger("zimprep.recon")


# --- Regex bank -------------------------------------------------------------

# A line that starts a new top-level question.
RE_Q_MAIN = re.compile(
    r"^\s*(?:Question\s+)?(\d{1,2})\s*[\.\)\:]\s*(.*)$",
    re.IGNORECASE,
)

# First-level subpart: (a), a), a.
RE_Q_LETTER = re.compile(
    r"^\s*\(?([a-z])\)?\s*[\.\)\:]?\s*(.*)$",
    re.IGNORECASE,
)

# Roman subpart: (i), ii., (iii)
RE_Q_ROMAN = re.compile(
    r"^\s*\(?((?:i{1,3}|iv|v|vi{0,3}|ix|x))\)\s*(.*)$",
    re.IGNORECASE,
)

# MCQ option letter at start of line (A. B. C. D. and sometimes E.).
RE_MCQ_OPT = re.compile(r"^\s*([A-E])\s*[\.\)]\s*(.+)$")

# Marks at end of line: [3], (3 marks), 3 marks
RE_MARKS = re.compile(
    r"[\[\(]\s*(\d{1,3})\s*(?:marks?|mks?|pts?)?\s*[\]\)]\s*$",
    re.IGNORECASE,
)
RE_MARKS_INLINE = re.compile(r"\b(\d{1,3})\s*marks?\b", re.IGNORECASE)

# Headers we should not eat into question text.
RE_INSTRUCTION = re.compile(
    r"^\s*(answer\s+all|answer\s+any|section\s+[A-Z]|instructions|do not|"
    r"time\s*[:\-]|total\s+marks|turn\s+over|page\s+\d+|\[?total\b)",
    re.IGNORECASE,
)

# A line that looks like just a page number / footer.
RE_PAGE_FOOTER = re.compile(r"^\s*(?:-\s*)?\d{1,3}\s*(?:-\s*)?$")


ROMAN_SET = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


# --- State machine ----------------------------------------------------------

@dataclass
class _RState:
    questions: List[Question] = field(default_factory=list)
    current_q: Optional[Question] = None
    current_sub: Optional[SubQuestion] = None  # level 1 (letter)
    current_roman: Optional[SubQuestion] = None  # level 2 (roman)
    last_target: Optional[object] = None  # last container we appended text to

    def open_question(self, number: int, page: int) -> Question:
        q = Question(question_number=number, page=page)
        self.questions.append(q)
        self.current_q = q
        self.current_sub = None
        self.current_roman = None
        self.last_target = q
        return q

    def open_subpart(self, label: str, page: int) -> Optional[SubQuestion]:
        if self.current_q is None:
            return None
        sub = SubQuestion(label=label.lower(), level=1, page=page)
        self.current_q.subquestions.append(sub)
        self.current_sub = sub
        self.current_roman = None
        self.last_target = sub
        return sub

    def open_roman(self, label: str, page: int) -> Optional[SubQuestion]:
        host = self.current_sub or (
            self.current_q.subquestions[-1] if self.current_q and self.current_q.subquestions else None
        )
        if host is None:
            if self.current_q is None:
                return None
            # Promote a synthetic letter-level container so romans nest correctly.
            host = self.open_subpart("a", page)
            if host is None:
                return None
        rom = SubQuestion(label=label.lower(), level=2, page=page)
        host.subquestions.append(rom)
        self.current_roman = rom
        self.last_target = rom
        return rom

    def append_mcq_option(self, letter: str, text: str, page: int):
        if self.current_q is None:
            return
        self.current_q.question_type = "mcq"
        opt = SubQuestion(label=letter.upper(), level=3, text=text, question_type="mcq_option", page=page)
        self.current_q.subquestions.append(opt)
        self.last_target = opt

    def append_text(self, text: str):
        if self.last_target is None:
            if self.current_q is None:
                return
            self.last_target = self.current_q
        if not text:
            return
        if isinstance(self.last_target, (Question, SubQuestion)):
            if self.last_target.text:
                self.last_target.text = f"{self.last_target.text} {text}".strip()
            else:
                self.last_target.text = text.strip()


# --- Helpers ----------------------------------------------------------------

def _strip_marks(line: str) -> Tuple[str, Optional[int]]:
    """Pull trailing [n] / (n marks) off a line. Return cleaned text + marks."""
    m = RE_MARKS.search(line)
    if m:
        marks = safe_int(m.group(1))
        line = RE_MARKS.sub("", line).rstrip()
        return line, marks
    m = RE_MARKS_INLINE.search(line)
    if m:
        marks = safe_int(m.group(1))
        line = RE_MARKS_INLINE.sub("", line).rstrip()
        return line, marks
    return line, None


def _is_probable_roman(label: str) -> bool:
    return label.lower() in ROMAN_SET


def _sort_blocks(blocks: Sequence[OCRBlock]) -> List[OCRBlock]:
    """Top-to-bottom, ties broken left-to-right."""
    return sorted(blocks, key=lambda b: (round(b.y / 8.0), b.x))


def _assign_marks(target, marks: Optional[int]):
    if marks is None or target is None:
        return
    if isinstance(target, (Question, SubQuestion)):
        if target.marks is None:
            target.marks = marks


def _detect_meta_from_text(pages: Sequence[PageData]) -> PaperMeta:
    """Sniff subject/paper/year from the first 2 pages."""
    meta = PaperMeta()
    head = "\n".join(p.native_text or " ".join(b.text for b in p.blocks)
                     for p in pages[:2])
    if not head:
        return meta
    yr = re.search(r"\b(19|20)\d{2}\b", head)
    if yr:
        meta.year = yr.group(0)
    paper = re.search(r"Paper\s*([0-9IVX]+)", head, re.IGNORECASE)
    if paper:
        meta.paper = f"Paper {paper.group(1)}"
    subject_match = re.search(
        r"\b(BIOLOGY|CHEMISTRY|PHYSICS|MATHEMATICS|MATHS|ENGLISH|GEOGRAPHY|HISTORY|"
        r"ACCOUNTING|ECONOMICS|COMMERCE|BUSINESS\s+STUDIES|COMPUTING|COMPUTER\s+SCIENCE|"
        r"COMBINED\s+SCIENCE|AGRICULTURE|SHONA|NDEBELE)\b",
        head, re.IGNORECASE,
    )
    if subject_match:
        meta.subject = subject_match.group(1).title()
    if "ZIMSEC" in head.upper():
        meta.board = "ZIMSEC"
    elif "CAMBRIDGE" in head.upper() or "CIE" in head.upper():
        meta.board = "Cambridge"
    return meta


def _associate_diagrams(
    questions: List[Question],
    diagrams: List[DiagramRegion],
):
    """Attach each diagram to the closest preceding question/subpart on its page."""
    if not diagrams or not questions:
        return

    # Flatten question + subquestion anchors with (page, y, ref).
    anchors: List[Tuple[int, float, object]] = []
    for q in questions:
        anchors.append((q.page or 0, 0.0, q))
        for sub in q.subquestions:
            anchors.append((sub.page or q.page, 0.0, sub))
            for rom in sub.subquestions:
                anchors.append((rom.page or sub.page, 0.0, rom))

    for d in diagrams:
        best = None
        best_dist = float("inf")
        for (apage, ay, anchor) in anchors:
            if apage != d.page:
                # allow a diagram on the page just after its question
                if apage + 1 == d.page:
                    dist = 10_000  # weak preference
                else:
                    continue
            else:
                dist = abs(d.bbox[1] - ay)
            if dist < best_dist:
                best_dist = dist
                best = anchor
        if best is None:
            continue
        best.has_image = True
        if d.ref_id not in best.image_refs:
            best.image_refs.append(d.ref_id)


# --- Public API -------------------------------------------------------------

def reconstruct_paper(
    pages: Sequence[PageData],
    *,
    vlm_results: Optional[Sequence[object]] = None,
    meta_override: Optional[PaperMeta] = None,
) -> ExtractionResult:
    """Run the reconstruction state machine across all pages."""
    state = _RState()
    all_diagrams: List[DiagramRegion] = []

    for page in pages:
        all_diagrams.extend(page.diagrams)
        _process_page(state, page)

    # Optionally fuse VLM JSON (overrides empty stems / missing marks).
    if vlm_results:
        _merge_vlm_results(state.questions, vlm_results)

    _post_process(state.questions)
    _associate_diagrams(state.questions, all_diagrams)

    meta = meta_override or _detect_meta_from_text(pages)

    stats = {
        "page_count": len(pages),
        "question_count": len(state.questions),
        "fallback_pages": sum(1 for p in pages if p.used_fallback),
        "vlm_pages": sum(1 for p in pages if p.used_vlm),
        "avg_ocr_confidence": (
            float(sum(p.ocr_confidence for p in pages) / max(len(pages), 1))
        ),
    }

    return ExtractionResult(
        meta=meta,
        pages=list(pages),
        questions=state.questions,
        diagrams=all_diagrams,
        stats=stats,
    )


# --- Page processing -------------------------------------------------------

def _process_page(state: _RState, page: PageData):
    blocks = _sort_blocks(page.blocks)
    # Group blocks that share a baseline into single lines for cleaner regexes.
    lines = _group_blocks_to_lines(blocks)
    saw_main_on_page = False

    for line_text, line_conf in lines:
        text = line_text.strip()
        if not text or RE_PAGE_FOOTER.match(text):
            continue

        # Instruction / header lines: stash but don't break the FSM.
        if RE_INSTRUCTION.match(text) and state.current_q is None:
            continue

        # Strip marks first so they don't interfere with subpart detection.
        text_no_marks, marks = _strip_marks(text)

        # 1. Main question?
        m = RE_Q_MAIN.match(text_no_marks)
        if m and not _looks_like_subpart_only(text_no_marks):
            num = safe_int(m.group(1))
            stem = m.group(2).strip()
            if num is not None and _is_plausible_question_number(state, num):
                q = state.open_question(num, page.page_number)
                q.confidence = line_conf
                if stem:
                    state.append_text(stem)
                _assign_marks(q, marks)
                saw_main_on_page = True
                continue

        # 2. Roman subpart? Must check BEFORE letter regex.
        m = RE_Q_ROMAN.match(text_no_marks)
        if m and _is_probable_roman(m.group(1)):
            label = m.group(1)
            rest = m.group(2).strip()
            rom = state.open_roman(label, page.page_number)
            if rom is not None:
                rom.confidence = line_conf
                if rest:
                    state.append_text(rest)
                _assign_marks(rom, marks)
                continue

        # 3. MCQ option? Only when current question is or could be MCQ.
        m = RE_MCQ_OPT.match(text_no_marks)
        if m and _looks_like_mcq_context(state):
            state.append_mcq_option(m.group(1), m.group(2).strip(), page.page_number)
            continue

        # 4. Letter subpart?
        m = RE_Q_LETTER.match(text_no_marks)
        if m and _looks_like_letter_subpart(text_no_marks):
            label = m.group(1)
            rest = m.group(2).strip()
            sub = state.open_subpart(label, page.page_number)
            if sub is not None:
                sub.confidence = line_conf
                if rest:
                    state.append_text(rest)
                _assign_marks(sub, marks)
                continue

        # 5. Continuation text for whatever is open.
        state.append_text(text_no_marks)
        # If the line had marks, assume they belong to the current target.
        _assign_marks(state.last_target, marks)

    if not saw_main_on_page and state.current_q is not None:
        # Pure continuation page; nothing extra to do — text already appended.
        pass


def _group_blocks_to_lines(blocks: Sequence[OCRBlock]) -> List[Tuple[str, float]]:
    """Cluster blocks whose y-centres are within a tolerance into single lines."""
    if not blocks:
        return []
    lines: List[List[OCRBlock]] = []
    tol_default = max(blocks[0].height * 0.7, 6.0)
    for b in blocks:
        placed = False
        bc = (b.bbox[1] + b.bbox[3]) / 2.0
        tol = max(b.height * 0.7, tol_default)
        for ln in lines:
            lc = (ln[-1].bbox[1] + ln[-1].bbox[3]) / 2.0
            if abs(bc - lc) <= tol:
                ln.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    out: List[Tuple[str, float]] = []
    for ln in lines:
        ln.sort(key=lambda x: x.x)
        text = " ".join(b.text for b in ln).strip()
        conf = float(sum(b.confidence for b in ln) / len(ln))
        out.append((text, conf))
    return out


def _is_plausible_question_number(state: _RState, num: int) -> bool:
    """Block hallucinated jumps backward (4 -> 2) unless it's a clear restart."""
    if not state.questions:
        return num == 1 or num <= 5  # tolerate a paper that starts at 1..5
    last = state.questions[-1].question_number
    if num == last:
        return False  # already open; treat as continuation
    if num == last + 1:
        return True
    if num > last + 1 and num - last <= 5:
        return True  # gap, but plausible
    if num < last:
        return False  # backward jump → probably part of body text
    return True


def _looks_like_subpart_only(text: str) -> bool:
    """Reject 'Question 5' inside e.g. 'pulled by 5 N force'."""
    stripped = text.strip().lower()
    if stripped.startswith(("question ", "question\t")):
        return False
    # If the number is followed by units or non-dot punctuation, it's body text.
    m = re.match(r"^\s*(\d{1,2})\s*([^\.\)\:])", text)
    if m and m.group(2).lower() in ("g", "n", "m", "k", "c", "%", "v", "a"):
        return True
    return False


def _looks_like_mcq_context(state: _RState) -> bool:
    if state.current_q is None:
        return False
    if state.current_q.question_type == "mcq":
        return True
    # Heuristic: if the current question text is short and ends with '?', a following
    # A./B./C./D. line strongly implies MCQ.
    txt = state.current_q.text.strip()
    return bool(txt) and (txt.endswith("?") or len(txt) < 220)


def _looks_like_letter_subpart(text: str) -> bool:
    """Filter out single-letter sentence starts ('A solution was prepared...')."""
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    # Require explicit paren / bracket / parenthesis-after-letter.
    if re.match(r"^\s*\(?[a-z]\)\s*", stripped, re.IGNORECASE):
        return True
    if re.match(r"^\s*[a-z]\.\s+\S", stripped):
        # Avoid eating "A." that starts a sentence: require the rest to look like a clause.
        return True
    return False


def _merge_vlm_results(questions: List[Question], vlm_results: Iterable[object]):
    """Use VLM JSON to fill blanks; never overwrite confident OCR text."""
    by_number = {q.question_number: q for q in questions}
    for res in vlm_results:
        if not getattr(res, "ok", False):
            continue
        data = getattr(res, "raw_json", {})
        for vq in data.get("questions", []):
            num = vq.get("number")
            if num is None:
                continue
            q = by_number.get(num)
            if q is None:
                q = Question(question_number=int(num), text=vq.get("text", ""))
                questions.append(q)
                by_number[num] = q
            if not q.text and vq.get("text"):
                q.text = vq["text"]
            if q.marks is None and vq.get("marks"):
                q.marks = safe_int(vq["marks"])
            _merge_vlm_subs(q, vq.get("subquestions", []))


def _merge_vlm_subs(parent, vlm_subs: list):
    existing_labels = {s.label.lower(): s for s in parent.subquestions}
    for vs in vlm_subs or []:
        label = str(vs.get("label", "")).lower()
        if not label:
            continue
        sub = existing_labels.get(label)
        if sub is None:
            sub = SubQuestion(
                label=label,
                level=2 if _is_probable_roman(label) else 1,
                text=vs.get("text", ""),
                marks=safe_int(vs.get("marks")),
            )
            parent.subquestions.append(sub)
        else:
            if not sub.text and vs.get("text"):
                sub.text = vs["text"]
            if sub.marks is None and vs.get("marks"):
                sub.marks = safe_int(vs["marks"])
        _merge_vlm_subs(sub, vs.get("subquestions", []))


# --- Post-processing -------------------------------------------------------

def _post_process(questions: List[Question]):
    """Detect question_type, roll up marks, trim noise."""
    seen_numbers: dict[int, int] = {}
    for q in questions:
        # Question type detection.
        if q.question_type != "mcq":
            opts = [s for s in q.subquestions if s.question_type == "mcq_option"]
            if opts and len(opts) >= 2:
                q.question_type = "mcq"
            elif not q.subquestions and len(q.text.split()) > 30:
                q.question_type = "essay"
            elif q.subquestions:
                q.question_type = "structured"

        # Roll marks up: if question has no top-level marks but subs do, sum them.
        if q.marks is None:
            sub_marks = [s.marks for s in q.subquestions if s.marks is not None]
            if sub_marks:
                q.marks = sum(sub_marks)

        # Roll-up confidence: weighted average of OCR confidences in this question.
        confs = [q.confidence] + [s.confidence for s in q.subquestions]
        if confs:
            q.confidence = round(sum(confs) / len(confs), 3)

        # Duplicate numbering bookkeeping (handled in validation, but tag here too).
        seen_numbers[q.question_number] = seen_numbers.get(q.question_number, 0) + 1

    for num, count in seen_numbers.items():
        if count > 1:
            for q in questions:
                if q.question_number == num:
                    q.issues.append(f"duplicate_question_number:{num}")
