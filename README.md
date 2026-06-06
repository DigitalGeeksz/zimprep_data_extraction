# ZimPrep — Exam Paper Extraction Platform

Internal web tool that turns exam PDFs (ZIMSEC, Cambridge, scanned or native)
into structured JSON, with a Gradio review interface for non-technical staff.

## Architecture

```
PDF upload
   ↓ pdf_analysis.py        — classify scanned/native, render pages, find diagrams
   ↓ preprocess.py          — deskew, denoise, CLAHE, sharpen, threshold
   ↓ ocr_engine.py          — PaddleOCR primary, DeepSeek-OCR fallback
   ↓ vlm_engine.py          — Qwen2.5-VL on flagged pages only
   ↓ reconstruction.py      — deterministic FSM + regex hierarchy rebuild
   ↓ validation.py          — duplicates, broken hierarchy, low-confidence flags
   ↓ ui/app.py              — Gradio review + edit + approve/reject
   ↓ export_json.py         — production-ready JSON
```

## Layout

```
paper_pipeline/
├── pipeline/
│   ├── __init__.py
│   ├── utils.py             # dataclasses, logging, workspace paths
│   ├── pdf_analysis.py      # PyMuPDF + pdfplumber: page classify + diagram extract
│   ├── preprocess.py        # OpenCV preprocessing
│   ├── ocr_engine.py        # PaddleOCR + DeepSeek-OCR fallback
│   ├── vlm_engine.py        # Qwen2.5-VL structural pass
│   ├── reconstruction.py    # hierarchy state machine
│   ├── validation.py        # validation engine
│   ├── export_json.py       # JSON serialisation
│   └── orchestrator.py      # end-to-end runner
├── ui/
│   └── app.py               # Gradio Blocks UI (left / center / right)
├── notebooks/
│   └── colab_run.ipynb      # one-click Colab launcher
├── workspace/               # uploads / images / exports (created at runtime)
├── requirements.txt
├── run.py                   # CLI entry point
└── README.md
```

## Local quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py            # local: http://localhost:7860
python run.py --no-share # don't create a public URL
```

## Google Colab deployment (recommended)

1. Open `notebooks/colab_run.ipynb` in Colab.
2. Runtime → Change runtime type → **GPU** (T4 minimum; A100 for Qwen-7B).
3. Run the install cell.
4. Upload the entire `paper_pipeline/` folder to `/content/` (or `!git clone`).
5. Run the launch cell. Gradio prints a `*.gradio.live` URL — share that
   with employees. The link is valid for 72 hours; re-run the cell to refresh.

For a stable long-lived URL, host on a small GPU VM:

```bash
nohup python run.py --port 7860 > server.log 2>&1 &
# put behind nginx + a corporate auth proxy
```

## Extraction modes

| Mode      | OCR fallback | VLM | Speed | Best for |
|-----------|:-:|:-:|:-:|----------|
| Fast      | ❌ | ❌ | fast  | clean native PDFs |
| Balanced  | ✅ | ❌ | mid   | most ZIMSEC / Cambridge papers |
| Accurate  | ✅ | ✅ | slow  | scanned, dense diagrams, exotic layouts |

## JSON output (per paper)

```json
{
  "subject": "Biology",
  "paper": "Paper 2",
  "year": "2023",
  "level": "O-Level",
  "board": "ZIMSEC",
  "questions": [
    {
      "question_number": 1,
      "text": "Fig. 1.1 shows a section through a leaf.",
      "marks": 10,
      "question_type": "structured",
      "subquestions": [
        { "label": "a", "level": 1, "text": "Name the structure labelled X.", "marks": 1, ... }
      ],
      "has_image": true,
      "image_refs": ["img_8f2e1a90"],
      "confidence": 0.94,
      "page": 2,
      "issues": [],
      "approved": true
    }
  ],
  "diagrams": [...],
  "stats": {...},
  "warnings": []
}
```

## Design notes

- **Determinism first.** The reconstruction engine is a regex + finite state
  machine. It does **not** call a VLM unless explicitly asked, so structural
  output is reproducible and auditable.
- **VLM only where needed.** Qwen2.5-VL is consulted for pages flagged
  ambiguous (low OCR confidence, fallback triggered, tables, or floating
  diagrams). Its JSON is merged conservatively — never overwrites confident
  OCR text, only fills blanks.
- **Confidence everywhere.** Every question/subpart carries an OCR-derived
  confidence so reviewers can sort by uncertainty.
- **Human in the loop.** Approve / reject / edit are first-class. Employees
  do not need to know any code.

## Known limits / TODO

- Maths notation: OCR returns ASCII; preserving full LaTeX requires a math-
  aware model (Pix2Tex / Nougat) — easy to add as a third backend in
  `ocr_engine.py`.
- Table parsing returns rows as plain lines — extend `pdf_analysis._has_tables`
  to actually emit a structured `table` field if needed.
- DeepSeek-OCR's geometry is approximate; if you need true line bounding
  boxes from the fallback, switch to Nougat / TrOCR.
