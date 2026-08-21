from marketing_agent.adapters.ocr_text import QwenVLOCREvaluator
from marketing_agent.graph import run_task
from marketing_agent.schemas import Observation, ObservationStatus, TaskRequest
from marketing_agent.tools import AgentTool, ToolRegistry


def test_ocr_matching_accepts_combined_lines_and_detects_duplicate_brand():
    result = QwenVLOCREvaluator.assess_texts(
        ["焕亮新生", "奢润修护精华", "LUMIÈRE"],
        ["焕亮新生\n奢润修护精华", "LUMIÈRE", "LUMIÈRE"],
    )
    assert result["missing"] == []
    assert result["duplicated"] == ["LUMIÈRE"]
    assert result["accuracy"] == 1.0
    assert result["uniqueness"] == 0.0


class Generator(AgentTool):
    name = "generate_image"

    def execute(self, arguments):
        attempt = arguments["attempt"]
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs={
                "file_path": f"v{attempt}.png",
                "prompt": arguments["prompt"],
                "seed": arguments["seed"],
            },
        )


class SequencedEvaluator(AgentTool):
    name = "evaluate_image"

    def __init__(self):
        self.index = 0
        self.values = [
            (0.90, 1.0, 0.0),
            (0.85, 1.0, 1.0),
            (0.84, 1.0, 1.0),
            (0.83, 1.0, 1.0),
        ]

    def execute(self, arguments):
        score, accuracy, uniqueness = self.values[self.index]
        self.index += 1
        issues = ["text_duplication"] if uniqueness < 1.0 else ["scene"]
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.PARTIAL,
            metrics={
                "marketing_alignment": score,
                "text_accuracy": accuracy,
                "text_uniqueness": uniqueness,
            },
            issues=issues,
            recommended_actions=issues,
        )


class Editor(AgentTool):
    name = "edit_image"

    def execute(self, arguments):
        raise AssertionError("unused")


def test_compliant_candidate_beats_higher_scoring_noncompliant_candidate():
    registry = ToolRegistry()
    registry.register(Generator())
    registry.register(SequencedEvaluator())
    registry.register(Editor())
    result = run_task(TaskRequest(prompt="生成高端精华液海报", max_iterations=12), registry)

    assert result.best_aesthetic_asset_id == result.assets[0].asset_id
    assert result.best_aesthetic_score == 0.90
    assert result.best_compliant_asset_id == result.assets[1].asset_id
    assert result.best_compliant_score == 0.85
    assert result.best_asset_id == result.assets[1].asset_id
    assert result.best_score == 0.85
