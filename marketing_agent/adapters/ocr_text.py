from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

class QwenVLOCREvaluator:
    def __init__(self, project_root: Path, device: str = "cuda:0") -> None:
        self.root = project_root
        self.checkpoint = self.root / "runtime/models/Qwen2.5-VL-3B-Instruct"
        self.device = device

    @staticmethod
    def _parse_output(text: str) -> list[str]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"OCR model returned no JSON object: {text[-500:]}")
        payload = json.loads(match.group(0))
        return [str(item).strip() for item in payload.get("texts", []) if str(item).strip()]

    @staticmethod
    def assess_texts(expected: list[str], texts: list[str]) -> dict[str, Any]:
        normalized = [re.sub(r"\s+", "", text).casefold() for text in texts]
        joined_text = "".join(normalized)
        expected_normalized = [re.sub(r"\s+", "", text).casefold() for text in expected]
        missing = [text for text, norm in zip(expected, expected_normalized) if norm not in joined_text]
        duplicated = [text for text, norm in zip(expected, expected_normalized) if joined_text.count(norm) > 1]
        matched = len(expected) - len(missing)
        return {
            "missing": missing,
            "duplicated": duplicated,
            "accuracy": matched / len(expected) if expected else 1.0,
            "uniqueness": 1.0 if not duplicated else 0.0,
        }

    def evaluate(self, *, image_path: str, campaign_text: str) -> dict[str, Any]:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint, local_files_only=True)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(image_path).resolve())},
                {"type": "text", "text": (
                    "Act as a strict OCR engine. Read every visible text region, including small "
                    "or malformed text. Return only JSON as {\"texts\": [exact transcriptions in "
                    "reading order]}. Do not correct spelling or add explanations."
                )},
            ],
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(self.device)
        generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = [output[len(source):] for source, output in zip(inputs.input_ids, generated)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        texts = self._parse_output(raw)

        expected = re.findall(r"[“\"]([^”\"]+)[”\"]", campaign_text)
        assessment = self.assess_texts(expected, texts)
        missing = assessment["missing"]
        duplicated = assessment["duplicated"]
        issues = []
        recommendations = []
        if missing:
            issues.append("text_accuracy")
            recommendations.append("准确显示缺失或错误文字：" + "、".join(missing))
        if duplicated:
            issues.append("text_duplication")
            recommendations.append("每个指定文本仅出现一次，删除重复文字：" + "、".join(duplicated))
        return {
            "texts": texts,
            "expected_texts": expected,
            "missing": missing,
            "duplicated": duplicated,
            "dimensions": {
                "text_accuracy": assessment["accuracy"],
                "text_uniqueness": assessment["uniqueness"],
            },
            "issues": issues,
            "recommendations": recommendations,
        }
