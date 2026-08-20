from marketing_agent.graph import run_task
from marketing_agent.schemas import Observation, ObservationStatus, TaskRequest
from marketing_agent.tools import AgentTool, ToolRegistry


class RecordingGenerator(AgentTool):
    name = "generate_image"

    def __init__(self) -> None:
        self.calls = []

    def execute(self, arguments):
        self.calls.append(arguments)
        attempt = arguments["attempt"]
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs={
                "file_path": f"generated_v{attempt}.png",
                "prompt": arguments["prompt"],
                "seed": arguments["seed"],
            },
        )


class FlatEvaluator(AgentTool):
    name = "evaluate_image"

    def execute(self, arguments):
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.PARTIAL,
            metrics={"marketing_alignment": 0.8, "composition": 0.8},
            issues=["composition"],
            recommended_actions=["improve composition"],
        )


class UnusedEditor(AgentTool):
    name = "edit_image"

    def execute(self, arguments):
        raise AssertionError("edit tool should not be called")


def test_repair_uses_dynamic_seeds_concrete_prompt_best_asset_and_plateau_stop():
    generator = RecordingGenerator()
    registry = ToolRegistry()
    registry.register(generator)
    registry.register(FlatEvaluator())
    registry.register(UnusedEditor())

    result = run_task(
        TaskRequest(prompt="生成高端精华液商业海报", max_iterations=12), registry=registry
    )

    assert result.status == "aborted"
    assert result.terminal_reason == "quality_plateau"
    assert [call["seed"] for call in generator.calls] == [42, 1051, 2060]
    assert "优化构图" in generator.calls[1]["prompt"]
    assert result.best_asset_id == result.assets[0].asset_id
    assert result.best_score == 0.8
