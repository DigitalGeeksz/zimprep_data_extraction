import os
import sys
import traceback
from pathlib import Path

# Add project root to sys.path
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pipeline.orchestrator import run_pipeline, PipelineOptions

def main():
    pdf_path = HERE / "workspace" / "uploads" / "ICS 2203  (19) JUNE 2023 DAA (1).pdf"
    print(f"Running end-to-end pipeline test on: {pdf_path}")
    sys.stdout.flush()

    options = PipelineOptions(
        use_vlm=False,
        use_fallback_ocr=True,
        use_gpu=False,
        force_ocr=False,
        subject="Computer Science",
        paper="Paper 2",
        year="2023",
        level="O-Level",
        board="ZIMSEC",
    )

    try:
        result = run_pipeline(pdf_path, options)
        print("\nPipeline execution completed successfully!")
        print(f"Total Questions Extracted: {len(result.questions)}")
        print(f"Average OCR Confidence: {result.stats.get('avg_ocr_confidence', 0.0):.4f}")
        print(f"Flagged Questions: {result.stats.get('flagged_questions', 0)}")
        print(f"Warnings: {result.warnings}")
        sys.stdout.flush()
    except Exception as e:
        print("\nPipeline execution failed:")
        traceback.print_exc()
        sys.stdout.flush()
        exit(1)

if __name__ == "__main__":
    main()
