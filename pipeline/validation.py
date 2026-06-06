"""Validation engine that flags hierarchy / quality issues before export."""

from __future__ import annotations

from collections import Counter
from typing import List

from .utils import ExtractionResult, Question, SubQuestion, get_logger

log = get_logger("zimprep.val")


LOW_CONFIDENCE = 0.72
ESSAY_MIN_WORDS = 8


def validate_extraction(result: ExtractionResult) -> ExtractionResult:
    """Annotate questions with `issues` and a refined `confidence`."""
    if not result.questions:
        result.warnings.append("no_questions_detected")
        return result

    _check_duplicates(result.questions)
    for q in result.questions:
        _validate_question(q)

    _check_sequence(result.questions, result.warnings)

    # Surface aggregate stats.
    flagged = sum(1 for q in result.questions if q.issues)
    result.stats["flagged_questions"] = flagged
    result.stats["low_confidence_questions"] = sum(
        1 for q in result.questions if q.confidence < LOW_CONFIDENCE
    )
    return result


def _check_duplicates(questions: List[Question]):
    counts = Counter(q.question_number for q in questions)
    for q in questions:
        if counts[q.question_number] > 1:
            tag = f"duplicate_question_number:{q.question_number}"
            if tag not in q.issues:
                q.issues.append(tag)


def _check_sequence(questions: List[Question], warnings: List[str]):
    nums = [q.question_number for q in questions]
    expected = list(range(min(nums), max(nums) + 1)) if nums else []
    missing = [n for n in expected if n not in nums]
    if missing:
        warnings.append(f"missing_question_numbers:{missing}")


def _validate_question(q: Question):
    # Empty stem and no subparts -> almost certainly broken.
    if not q.text.strip() and not q.subquestions:
        q.issues.append("empty_question")

    # Missing marks: only flag if siblings have marks (else paper just doesn't show marks).
    if q.marks is None and any(s.marks is not None for s in q.subquestions):
        # A multi-part question where the rollup couldn't determine total.
        if not all(s.marks is not None for s in q.subquestions):
            q.issues.append("missing_subpart_marks")

    if q.confidence < LOW_CONFIDENCE:
        q.issues.append(f"low_confidence:{q.confidence:.2f}")

    # MCQ structure check.
    if q.question_type == "mcq":
        opts = [s for s in q.subquestions if s.question_type == "mcq_option"]
        labels = [o.label for o in opts]
        if len(opts) < 3:
            q.issues.append("mcq_too_few_options")
        if len(set(labels)) != len(labels):
            q.issues.append("mcq_duplicate_option_labels")
        # Options should be contiguous from A.
        expected = [chr(ord("A") + i) for i in range(len(opts))]
        if labels and labels != expected:
            q.issues.append(f"mcq_option_sequence:{labels}")

    # Hierarchy sanity for letter subparts: should be a, b, c, ...
    letters = [s.label for s in q.subquestions if s.level == 1]
    if letters:
        expected = [chr(ord("a") + i) for i in range(len(letters))]
        if letters != expected:
            q.issues.append(f"subpart_sequence:{letters}")

    # Roman sub-sub sanity per letter.
    for sub in q.subquestions:
        if sub.level == 1 and sub.subquestions:
            romans = [r.label for r in sub.subquestions]
            expected_roman = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"][: len(romans)]
            if romans != expected_roman:
                q.issues.append(f"roman_sequence:{sub.label}:{romans}")

    # Essay sanity.
    if q.question_type == "essay" and len(q.text.split()) < ESSAY_MIN_WORDS:
        q.issues.append("essay_too_short")
