from typing import Any

from marketing_agent.graph import run_task
from marketing_agent.real_tools import QwenGenerateTool, build_qwen_registry
from marketing_agent.schemas import ObservationStatus, TaskRequest


class FakeGenerator:
    def generate(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "file_path": "runtime/outputs/fake.png",
            "prompt": kwargs["prompt"],
            "seed": kwargs["seed"],
            "backend": "Qwen/Qwen-Image",
            "latency_seconds": 1.25,
        }


def test_qwen_tool_converts_adapter_result_to_observation() -> None:
    tool = QwenGenerateTool(FakeGenerator())
    observation = tool.execute({"prompt": "campaign poster", "seed": 7})
    assert observation.status == ObservationStatus.SUCCESS
    assert observation.outputs["backend"] == "Qwen/Qwen-Image"
    assert observation.metrics["latency_seconds"] == 1.25


def test_qwen_registry_preserves_agent_tool_contract() -> None:
    registry = build_qwen_registry(FakeGenerator())
    assert registry.names == ("edit_image", "evaluate_image", "generate_image")


def test_langgraph_accepts_qwen_registry() -> None:
    result = run_task(
        TaskRequest(prompt="premium skincare campaign poster", max_iterations=8),
        registry=build_qwen_registry(FakeGenerator()),
    )
    assert result.status == "completed"
    assert result.terminal_reason == "quality_gate_passed"
    assert len(result.assets) == 2
    assert all(asset.tool_name == "generate_image" for asset in result.assets)
