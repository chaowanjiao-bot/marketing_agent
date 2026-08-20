from marketing_agent.marketing_tools import VQAEvaluateTool
from marketing_agent.schemas import ObservationStatus
from marketing_agent.vision_contracts import EvaluationResult


class Evaluator:
    def evaluate(self, **arguments):
        return EvaluationResult(
            overall=0.86,
            dimensions={
                "composition": 0.865,
                "color": 0.878,
                "lighting": 0.891,
                "focus": 0.891,
                "emotion": 0.879,
                "creativity": 0.850,
                "subject": 0.865,
                "scene": 0.815,
                "spatial": 0.833,
                "brand": 0.849,
            },
            issues=[],
            recommendations=[],
        )


def test_failed_evaluation_selects_only_three_lowest_dimensions():
    observation = VQAEvaluateTool(Evaluator(), threshold=0.9).execute(
        {"image_path": "poster.png", "campaign_text": "premium serum poster"}
    )

    assert observation.status == ObservationStatus.PARTIAL
    assert observation.issues == ["scene", "spatial", "brand"]
    assert observation.recommended_actions == [
        "improve scene",
        "improve spatial",
        "improve brand",
    ]
