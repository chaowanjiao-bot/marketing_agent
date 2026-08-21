from marketing_agent.candidates import CANDIDATE_SEED_STRIDE, candidate_seed, run_candidate_batch
from marketing_agent.schemas import Observation, ObservationStatus, TaskRequest
from marketing_agent.tools import AgentTool, MockEditTool, ToolRegistry


class SeedScoredGenerator(AgentTool):
    name = "generate_image"

    def execute(self, arguments):
        seed = int(arguments["seed"])
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs={"file_path": f"candidate_{seed}.png", "seed": seed,
                     "prompt": arguments["prompt"]},
        )


class CandidateEvaluator(AgentTool):
    name = "evaluate_image"

    def execute(self, arguments):
        path = arguments["image_path"]
        if f"_{42 + CANDIDATE_SEED_STRIDE}.png" in path:
            metrics = {"marketing_alignment": 0.95, "text_accuracy": 0.5,
                       "text_uniqueness": 1.0}
        elif f"_{42 + 2 * CANDIDATE_SEED_STRIDE}.png" in path:
            metrics = {"marketing_alignment": 0.89, "text_accuracy": 1.0,
                       "text_uniqueness": 1.0}
        else:
            metrics = {"marketing_alignment": 0.82, "text_accuracy": 1.0,
                       "text_uniqueness": 1.0}
        return Observation(tool_name=self.name, status=ObservationStatus.SUCCESS, metrics=metrics)


def registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(SeedScoredGenerator())
    tools.register(CandidateEvaluator())
    tools.register(MockEditTool())
    return tools


def test_candidate_seeds_are_deterministic_and_separated() -> None:
    assert [candidate_seed(42, index) for index in range(3)] == [
        42, 42 + CANDIDATE_SEED_STRIDE, 42 + 2 * CANDIDATE_SEED_STRIDE
    ]


def test_batch_selects_best_compliant_candidate_over_aesthetic_best() -> None:
    result = run_candidate_batch(TaskRequest(
        prompt="生成高端精华活动海报", candidate_count=3, max_iterations=4,
    ), registry())
    assert result.selected_candidate_index == 2
    assert result.best_score == 0.89
    assert len(result.candidate_summaries) == 3
    assert result.candidate_summaries[1].best_score == 0.95
    assert result.candidate_summaries[1].compliant is False
    assert result.candidate_summaries[2].selected is True
    assert len(result.assets) == 3


def test_single_candidate_preserves_existing_result_shape() -> None:
    result = run_candidate_batch(TaskRequest(
        prompt="生成香水活动海报", candidate_count=1, max_iterations=4,
    ), registry())
    assert result.candidate_summaries == []
    assert result.selected_candidate_index == 0
