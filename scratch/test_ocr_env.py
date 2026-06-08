import os
import sys

# Disable the new executor / PIR globally
os.environ["FLAGS_enable_new_ir"] = "0"
os.environ["FLAGS_enable_new_executor"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_standalone_executor"] = "0"

import traceback
from paddleocr import PaddleOCR

print("Initializing PaddleOCR with env flags...")
sys.stdout.flush()
try:
    ocr = PaddleOCR(
        use_textline_orientation=True,
        lang='en',
        device='cpu',
        enable_mkldnn=False
    )
    print("PaddleOCR initialized successfully.")
    sys.stdout.flush()
except Exception as e:
    print("Initialization failed:")
    traceback.print_exc()
    sys.stdout.flush()
    exit(1)

print("Running OCR on sample page...")
sys.stdout.flush()
try:
    result = ocr.ocr('workspace/images/09f8c2493f/page_001.png')
    print("OCR run completed successfully.")
    print("Result:", result)
    sys.stdout.flush()
except Exception as e:
    print("OCR run failed:")
    traceback.print_exc()
    sys.stdout.flush()
