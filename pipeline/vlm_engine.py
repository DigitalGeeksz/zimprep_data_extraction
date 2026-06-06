"""Qwen2.5-VL wrapper used selectively for difficult structural reasoning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .utils import PageData, get_logger

log = get_logger("zimprep.vlm")


DEFAULT_MODEL_3B = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"


STRUCTURE_PROMPT = """You are reading a single page of an exam paper.
Return STRICT JSON with this schema and nothing else:
{
  "page_role": "questions|cover|instructions|formula|blank|other",
  "questions": [
    {
      "number": <int>,
      "text": "<question stem text without subparts>",
      "marks": <int or null>,
      "subquestions": [
        {"label": "a", "text": "...", "marks": <int or null>,
         "subquestions": [{"label":"i","text":"...","marks":null}]}
      ]
    }
  ]
}
Rules:
- Use the exact question numbering on the page (1, 2, 3...).
- Use letters a, b, c for first-level subparts and roman numerals i, ii, iii for nested ones.
- If the page only continues a question from a previous page, still emit it with its number.
- Do not invent content. If text is unreadable, write "[illegible]".
- Output JSON only, no commentary, no markdown fence.
"""


@dataclass
class VLMPageStructure:
    page_role: str
    raw_json: dict
    ok: bool


class VLMEngine:
    """Lazy-loaded Qwen2.5-VL engine. Skips loading until first use."""

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        self.model_id = model_id or DEFAULT_MODEL_3B
        self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        log.info("Loading VLM %s on %s", self.model_id, device)
        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if device == "cpu":
            self._model = self._model.to(device)
        self._model.eval()
        self.device = device

    def analyze_page(self, image_path: str | Path, prompt: str = STRUCTURE_PROMPT) -> VLMPageStructure:
        self._ensure_loaded()
        from qwen_vl_utils import process_vision_info
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
        raw = self._processor.batch_decode(gen_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return _parse_vlm_json(raw)

    def analyze_pages(self, pages: List[PageData]) -> List[VLMPageStructure]:
        results: List[VLMPageStructure] = []
        for page in pages:
            if not page.image_path:
                results.append(VLMPageStructure("other", {}, False))
                continue
            try:
                res = self.analyze_page(page.image_path)
                page.used_vlm = True
            except Exception as e:
                log.warning("VLM failed on page %d: %s", page.page_number, e)
                res = VLMPageStructure("other", {}, False)
            results.append(res)
        return results


def _parse_vlm_json(raw: str) -> VLMPageStructure:
    """Strip fences and parse JSON; return a structured result with ok flag."""
    txt = raw.strip()
    # Remove markdown fences if the model emits them.
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    # Find the first { ... last } so trailing chatter is tolerated.
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return VLMPageStructure("other", {"_raw": raw}, False)
    snippet = txt[start:end + 1]
    try:
        data = json.loads(snippet)
        return VLMPageStructure(
            page_role=data.get("page_role", "questions"),
            raw_json=data,
            ok=True,
        )
    except Exception:
        return VLMPageStructure("other", {"_raw": raw}, False)
