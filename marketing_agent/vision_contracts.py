from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SegmentationResult:
    mask_path: str
    boxes: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationResult:
    overall: float
    dimensions: dict[str, float]
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class Segmenter(Protocol):
    def segment(self, *, image_path: str, expression: str) -> SegmentationResult: ...


class ImageEditor(Protocol):
    def edit(
        self, *, image_path: str, mask_path: str, prompt: str, seed: int
    ) -> dict[str, object]: ...


class ImageEvaluator(Protocol):
    def evaluate(self, *, image_path: str, campaign_text: str) -> EvaluationResult: ...
