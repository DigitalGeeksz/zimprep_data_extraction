"""Serialise an ExtractionResult to the production JSON schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .utils import (
    ExtractionResult,
    Question,
    SubQuestion,
    workspace_path,
)


def _serialise_sub(sub: SubQuestion) -> dict:
    return {
        "label": sub.label,
        "level": sub.level,
        "text": sub.text,
        "marks": sub.marks,
        "question_type": sub.question_type,
        "has_image": sub.has_image,
        "image_refs": list(sub.image_refs),
        "confidence": round(sub.confidence, 3),
        "page": sub.page,
        "issues": list(sub.issues),
        "subquestions": [_serialise_sub(s) for s in sub.subquestions],
    }


def _serialise_question(q: Question) -> dict:
    return {
        "question_number": q.question_number,
        "text": q.text,
        "marks": q.marks,
        "question_type": q.question_type,
        "subquestions": [_serialise_sub(s) for s in q.subquestions],
        "has_image": q.has_image,
        "image_refs": list(q.image_refs),
        "confidence": round(q.confidence, 3),
        "page": q.page,
        "issues": list(q.issues),
        "approved": q.approved,
    }


def to_export_dict(result: ExtractionResult) -> dict:
    return {
        "subject": result.meta.subject,
        "paper": result.meta.paper,
        "year": result.meta.year,
        "level": result.meta.level,
        "board": result.meta.board,
        "questions": [_serialise_question(q) for q in result.questions],
        "diagrams": [
            {
                "ref_id": d.ref_id,
                "page": d.page,
                "bbox": list(d.bbox),
                "image_path": d.image_path,
                "caption": d.caption,
            }
            for d in result.diagrams
        ],
        "stats": result.stats,
        "warnings": result.warnings,
    }


def save_export(
    result: ExtractionResult,
    filename: Optional[str] = None,
    *,
    pretty: bool = True,
) -> Path:
    data = to_export_dict(result)
    if filename is None:
        meta = result.meta
        slug = "_".join(filter(None, [meta.subject or "paper", meta.year or "", meta.paper or ""]))
        slug = slug.replace(" ", "_") or "extraction"
        filename = f"{slug}.json"
    out_path = workspace_path("exports", filename)
    out_path.write_text(
        json.dumps(data, indent=2 if pretty else None, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
