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

    def restricted(self, allowed: frozenset[str] | set[str]) -> "RestrictedToolRegistry":
        return RestrictedToolRegistry(self, frozenset(allowed))


class RestrictedToolRegistry:
    """Capability view that enforces an Agent's declared tool allow-list."""

    def __init__(self, registry: ToolRegistry, allowed: frozenset[str]) -> None:
        self._registry = registry
        self._allowed = allowed

    def get(self, name: str) -> AgentTool:
        if name not in self._allowed:
            raise PermissionError(f"tool '{name}' is not allowed for this agent")
        return self._registry.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._registry.names) & set(self._allowed)))


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
        expected_texts = [str(x) for x in arguments.get("expected_texts") or []]
        if attempt == 1:
            return Observation(
                tool_name=self.name,
                status=ObservationStatus.PARTIAL,
                outputs={"recognized_texts": expected_texts},
                metrics={
                    "marketing_alignment": 0.62,
                    "text_accuracy": 1.0,
                    "text_uniqueness": 1.0,
                    "text_cleanliness": 1.0,
                },
                issues=["标题安全区不足"],
                recommended_actions=["根据评估意见重新生成或编辑"],
                cost=0.01,
            )
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs={"recognized_texts": expected_texts},
            metrics={
                "marketing_alignment": 0.88,
                "text_accuracy": 1.0,
                "text_uniqueness": 1.0,
                "text_cleanliness": 1.0,
            },
            cost=0.01,
        )


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(MockGenerateTool())
    registry.register(MockEditTool())
    registry.register(MockEvaluateTool())
    from .creative_tools import register_creative_tools
    register_creative_tools(registry, strict_typography=False)
    return registry
