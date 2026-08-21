from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schemas import Observation, ObservationStatus


class AgentTool(ABC):
    name: str

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> Observation:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}

    def register(self, tool: AgentTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def close(self) -> None:
        for tool in self._tools.values():
            close = getattr(tool, "close", None)
            if callable(close):
                close()

    def runtime_status(self) -> dict[str, object]:
        schedulers = []
        for tool in self._tools.values():
            scheduler = getattr(tool, "scheduler", None)
            if scheduler is not None and scheduler not in schedulers:
                schedulers.append(scheduler)
        return {
            "gpu_schedulers": [scheduler.snapshot() for scheduler in schedulers]
        }


class MockGenerateTool(AgentTool):
    name = "generate_image"

    def execute(self, arguments: dict[str, Any]) -> Observation:
        attempt = int(arguments.get("attempt", 1))
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs={
                "file_path": f"data/tasks/mock/generated_v{attempt}.png",
                "seed": int(arguments.get("seed", 42)),
                "prompt": str(arguments.get("prompt", "")),
                "width": int(arguments.get("width", 1024)),
                "height": int(arguments.get("height", 1024)),
                "output_format": str(arguments.get("output_format", "1:1")),
            },
            metrics={"latency_seconds": 0.01},
            cost=0.02,
        )


class MockEditTool(AgentTool):
    name = "edit_image"

    def execute(self, arguments: dict[str, Any]) -> Observation:
        attempt = int(arguments.get("attempt", 1))
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs={
                "file_path": f"data/tasks/mock/edited_v{attempt}.png",
                "seed": int(arguments.get("seed", 42)),
                "prompt": str(arguments.get("prompt", "")),
                "width": int(arguments.get("width", 1024)),
                "height": int(arguments.get("height", 1024)),
                "output_format": str(arguments.get("output_format", "1:1")),
            },
            metrics={"latency_seconds": 0.01},
            cost=0.02,
        )


class MockEvaluateTool(AgentTool):
    name = "evaluate_image"

    def execute(self, arguments: dict[str, Any]) -> Observation:
        attempt = int(arguments.get("attempt", 1))
        if attempt == 1:
            return Observation(
                tool_name=self.name,
                status=ObservationStatus.PARTIAL,
                metrics={"marketing_alignment": 0.62},
                issues=["标题安全区不足"],
                recommended_actions=["根据评估意见重新生成或编辑"],
                cost=0.01,
            )
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            metrics={"marketing_alignment": 0.88},
            cost=0.01,
        )


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(MockGenerateTool())
    registry.register(MockEditTool())
    registry.register(MockEvaluateTool())
    return registry
