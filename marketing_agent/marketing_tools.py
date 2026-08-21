from __future__ import annotations

from typing import Any

from .schemas import Observation, ObservationStatus
from .tools import AgentTool, ToolRegistry
from .vision_contracts import ImageEditor, ImageEvaluator, Segmenter


class SegmentThenEditTool(AgentTool):
    name = "edit_image"

    def __init__(self, segmenter: Segmenter, editor: ImageEditor) -> None:
        self.segmenter = segmenter
        self.editor = editor

    def execute(self, arguments: dict[str, Any]) -> Observation:
        expression = str(arguments.get("target_expression") or "main product")
        segmented = self.segmenter.segment(
            image_path=str(arguments["input_image"]), expression=expression
        )
        result = self.editor.edit(
            image_path=str(arguments["input_image"]),
            mask_path=segmented.mask_path,
            prompt=str(arguments["prompt"]),
            seed=int(arguments.get("seed", 42)),
        )
        latency = float(result.pop("latency_seconds", 0.0))
        result["detected_labels"] = segmented.labels
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs=result,
            metrics={"latency_seconds": latency},
        )

    def close(self) -> None:
        close = getattr(self.editor, "close", None)
        if callable(close):
            close()


class VQAEvaluateTool(AgentTool):
    name = "evaluate_image"

    def __init__(
        self,
        evaluator: ImageEvaluator,
        threshold: float = 0.7,
        max_repair_dimensions: int = 3,
        text_evaluator: Any | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.threshold = threshold
        self.max_repair_dimensions = max_repair_dimensions
        self.text_evaluator = text_evaluator

    def execute(self, arguments: dict[str, Any]) -> Observation:
        result = self.evaluator.evaluate(
            image_path=str(arguments["image_path"]),
            campaign_text=str(arguments["campaign_text"]),
        )
        text_result = (
            self.text_evaluator.evaluate(
                image_path=str(arguments["image_path"]),
                campaign_text=str(arguments["campaign_text"]),
            )
            if self.text_evaluator is not None
            else {"dimensions": {}, "issues": [], "recommendations": [], "texts": []}
        )
        passed = result.overall >= self.threshold and not text_result["issues"]
        issues = list(result.issues)
        recommendations = list(result.recommendations)
        if not passed:
            ranked = sorted(
                (
                    (name, value)
                    for name, value in result.dimensions.items()
                    if value < self.threshold
                ),
                key=lambda item: item[1],
            )
            priority = list(text_result["issues"]) + [name for name, _ in ranked]
            issues = list(dict.fromkeys(priority))[: self.max_repair_dimensions]
            if not issues:
                issues = list(result.issues[: self.max_repair_dimensions])
            text_recommendations = list(text_result["recommendations"])
            recommendations = text_recommendations + [
                f"improve {name}" for name in issues if name not in text_result["issues"]
            ]
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS if passed else ObservationStatus.PARTIAL,
            outputs={"recognized_texts": text_result["texts"]},
            metrics={
                "marketing_alignment": result.overall,
                **result.dimensions,
                **text_result["dimensions"],
            },
            issues=issues,
            recommended_actions=recommendations,
        )

    def close(self) -> None:
        for dependency in (self.evaluator, self.text_evaluator):
            close = getattr(dependency, "close", None)
            if callable(close):
                close()
