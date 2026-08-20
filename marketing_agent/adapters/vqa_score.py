from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ..vision_contracts import EvaluationResult


DIMENSIONS = {
    "composition": "The image has a professional and balanced composition",
    "color": "The image has harmonious colors suitable for the campaign",
    "lighting": "The lighting clearly presents the product",
    "focus": "The intended main product is visually prominent and in focus",
    "emotion": "The visual mood matches the campaign intent",
    "creativity": "The image is visually distinctive without harming clarity",
    "subject": "The requested product or subject is present",
    "scene": "The requested application scene is represented",
    "spatial": "The requested spatial relationships and layout are correct",
    "brand": "The requested style and brand constraints are respected",
}


class VQAScoreEvaluator:
    def __init__(
        self, project_root: Path, model: str = "qwen2.5-vl-3b", device: str = "cuda"
    ) -> None:
        self.root = project_root
        self.repo = self.root / "runtime/repositories/t2v_metrics"
        self.cache = self.root / "runtime/cache/vqascore"
        self.checkpoint = self.root / "runtime/models/Qwen2.5-VL-3B-Instruct"
        self.model = model
        self.device = device
        self._scorer: Any | None = None

    def load(self) -> None:
        if self._scorer is not None:
            return
        if not self.repo.is_dir():
            raise FileNotFoundError("t2v_metrics repository is missing")
        if not (self.checkpoint / "config.json").is_file() or any(
            self.checkpoint.rglob("*.incomplete")
        ):
            raise FileNotFoundError("Qwen2.5-VL-3B evaluator model is incomplete")
        sys.path.insert(0, str(self.repo))
        import t2v_metrics

        self._scorer = t2v_metrics.VQAScore(
            model=self.model,
            device=self.device,
            cache_dir=str(self.cache),
            checkpoint=str(self.checkpoint),
        )

    def evaluate(self, *, image_path: str, campaign_text: str) -> EvaluationResult:
        self.load()
        queries = [f"{statement}. Campaign requirement: {campaign_text}" for statement in DIMENSIONS.values()]
        scores = self._scorer(images=[image_path], texts=queries)
        values = scores.detach().float().cpu().reshape(-1).tolist()
        dimensions = dict(zip(DIMENSIONS, values))
        issues = [name for name, value in dimensions.items() if value < 0.7]
        recommendations = [f"improve {name}" for name in issues]
        return EvaluationResult(
            overall=sum(values) / len(values),
            dimensions=dimensions,
            issues=issues,
            recommendations=recommendations,
        )
