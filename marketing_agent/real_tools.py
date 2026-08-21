from __future__ import annotations

import os
from typing import Any, Protocol

from .adapters import QwenImageGenerator
from .adapters.grounded_sam2 import GroundedSam2Segmenter
from .adapters.subprocess_adapters import (
    SubprocessOCRTextEvaluator,
    SubprocessPowerPaintEditor,
    SubprocessVQAScoreEvaluator,
)
from .gpu_scheduler import GpuScheduler, ScheduledTool
from .marketing_tools import SegmentThenEditTool, VQAEvaluateTool
from .runtime_config import RuntimeConfig
from .schemas import Observation, ObservationStatus
from .tools import AgentTool, MockEditTool, MockEvaluateTool, ToolRegistry


class ImageGenerator(Protocol):
    def generate(self, **kwargs: Any) -> dict[str, Any]: ...


class QwenGenerateTool(AgentTool):
    name = "generate_image"

    def __init__(self, generator: ImageGenerator) -> None:
        self.generator = generator

    def execute(self, arguments: dict[str, Any]) -> Observation:
        attempt = int(arguments.get("attempt", 1))
        seed = int(arguments.get("seed", 42))
        output_format = str(arguments.get("output_format", "1:1"))
        format_slug = output_format.replace(":", "x")
        result = self.generator.generate(
            prompt=str(arguments.get("prompt", "")),
            seed=seed,
            width=int(arguments.get("width", 1024)),
            height=int(arguments.get("height", 1024)),
            output_name=f"qwen_{format_slug}_s{seed}_v{attempt}.png",
        )
        result.setdefault("output_format", output_format)
        latency = float(result.pop("latency_seconds", 0.0))
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs=result,
            metrics={"latency_seconds": latency},
        )


def build_qwen_registry(generator: ImageGenerator | None = None) -> ToolRegistry:
    real_generator = generator or QwenImageGenerator(RuntimeConfig.from_env())
    registry = ToolRegistry()
    registry.register(QwenGenerateTool(real_generator))
    registry.register(MockEditTool())
    registry.register(MockEvaluateTool())
    return registry


def build_production_registry(evaluation_threshold: float = 0.7) -> ToolRegistry:
    config = RuntimeConfig.from_env()
    scheduler = GpuScheduler(int(os.environ.get("GPU_MAX_CONCURRENT", "1")))
    registry = ToolRegistry()
    registry.register(ScheduledTool(QwenGenerateTool(QwenImageGenerator(config)), scheduler))
    registry.register(ScheduledTool(
        SegmentThenEditTool(
            GroundedSam2Segmenter(config.project_root),
            SubprocessPowerPaintEditor(config.project_root),
        ), scheduler,
    ))
    registry.register(ScheduledTool(
        VQAEvaluateTool(
            SubprocessVQAScoreEvaluator(config.project_root),
            threshold=evaluation_threshold,
            text_evaluator=SubprocessOCRTextEvaluator(config.project_root),
        ), scheduler,
    ))
    return registry
