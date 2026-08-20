from marketing_agent.marketing_tools import SegmentThenEditTool, VQAEvaluateTool
from marketing_agent.schemas import ObservationStatus
from marketing_agent.vision_contracts import EvaluationResult, SegmentationResult


class FakeSegmenter:
    def segment(self, *, image_path: str, expression: str) -> SegmentationResult:
        assert expression == "left bottle"
        return SegmentationResult("mask.png", labels=["bottle"])


class FakeEditor:
    def edit(self, **kwargs):
        return {
            "file_path": "edited.png",
            "prompt": kwargs["prompt"],
            "seed": kwargs["seed"],
            "latency_seconds": 2.0,
        }


class FakeEvaluator:
    def __init__(self, score: float) -> None:
        self.score = score

    def evaluate(self, **kwargs) -> EvaluationResult:
        return EvaluationResult(
            overall=self.score,
            dimensions={"composition": self.score},
            issues=[] if self.score >= 0.7 else ["composition"],
            recommendations=[] if self.score >= 0.7 else ["improve composition"],
        )


def test_segment_then_edit_composes_fixed_technical_steps() -> None:
    observation = SegmentThenEditTool(FakeSegmenter(), FakeEditor()).execute(
        {
            "input_image": "input.png",
            "target_expression": "left bottle",
            "prompt": "replace the bottle",
            "seed": 9,
        }
    )
    assert observation.status == ObservationStatus.SUCCESS
    assert observation.outputs["detected_labels"] == ["bottle"]
    assert observation.outputs["file_path"] == "edited.png"


def test_vqa_tool_controls_replan_threshold() -> None:
    failed = VQAEvaluateTool(FakeEvaluator(0.62)).execute(
        {"image_path": "a.png", "campaign_text": "campaign"}
    )
    passed = VQAEvaluateTool(FakeEvaluator(0.88)).execute(
        {"image_path": "b.png", "campaign_text": "campaign"}
    )
    assert failed.status == ObservationStatus.PARTIAL
    assert failed.recommended_actions == ["improve composition"]
    assert passed.status == ObservationStatus.SUCCESS
