"""Shared data structures, logging, and filesystem helpers."""

from __future__ import annotations

import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


WORKSPACE_ENV = "ZIMPREP_WORKSPACE"
DEFAULT_WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"


def workspace_path(*parts: str) -> Path:
    """Return a path under the workspace dir, creating parents as needed."""
    base = Path(os.environ.get(WORKSPACE_ENV, DEFAULT_WORKSPACE))
    p = base.joinpath(*parts) if parts else base
    if not parts or "." not in parts[-1]:
        p.mkdir(parents=True, exist_ok=True)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "zimprep") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    _LOGGERS[name] = logger
    return logger


@dataclass
class OCRBlock:
    """A single OCR text block with geometry and confidence."""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    page: int = 0
    source: str = "paddle"  # paddle | deepseek | vlm | pdf-native

    @property
    def y(self) -> float:
        return self.bbox[1]

    @property
    def x(self) -> float:
        return self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class DiagramRegion:
    """Image / figure region detected on a page."""
    page: int
    bbox: Tuple[float, float, float, float]
    image_path: Optional[str] = None
    caption: Optional[str] = None
    ref_id: str = field(default_factory=lambda: f"img_{uuid.uuid4().hex[:8]}")


@dataclass
class SubQuestion:
    label: str  # "a", "b", "i", "ii", "A", "B" ...
    level: int  # 1 = letter, 2 = roman, 3 = mcq option
    text: str = ""
    marks: Optional[int] = None
    question_type: str = "structured"
    subquestions: List["SubQuestion"] = field(default_factory=list)
    has_image: bool = False
    image_refs: List[str] = field(default_factory=list)
    confidence: float = 1.0
    page: int = 0
    issues: List[str] = field(default_factory=list)


@dataclass
class Question:
    question_number: int
    text: str = ""
    marks: Optional[int] = None
    question_type: str = "structured"  # structured | mcq | essay | instruction
    subquestions: List[SubQuestion] = field(default_factory=list)
    has_image: bool = False
    image_refs: List[str] = field(default_factory=list)
    confidence: float = 1.0
    page: int = 0
    issues: List[str] = field(default_factory=list)
    approved: Optional[bool] = None


@dataclass
class PageData:
    """Everything we know about a single PDF page."""
    page_number: int
    width: int
    height: int
    is_scanned: bool
    text_density: float
    native_text: str = ""
    image_path: Optional[str] = None
    preprocessed_path: Optional[str] = None
    blocks: List[OCRBlock] = field(default_factory=list)
    diagrams: List[DiagramRegion] = field(default_factory=list)
    ocr_confidence: float = 0.0
    used_fallback: bool = False
    used_vlm: bool = False
    has_table: bool = False


@dataclass
class PaperMeta:
    subject: str = ""
    paper: str = ""
    year: str = ""
    level: str = ""
    board: str = ""


@dataclass
class ExtractionResult:
    meta: PaperMeta
    pages: List[PageData]
    questions: List[Question]
    diagrams: List[DiagramRegion] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def make_run_id() -> str:
    return uuid.uuid4().hex[:10]
