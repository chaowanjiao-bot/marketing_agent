from marketing_agent.contracts import CampaignStatus, MultiAgentCampaign
from marketing_agent.creative_tools import register_creative_tools
from marketing_agent.schemas import Observation, ObservationStatus, OutputFormat, TaskRequest
from marketing_agent.tools import AgentTool, MockEditTool, MockGenerateTool, ToolRegistry
from marketing_agent.workflows import run_multi_agent_campaign


class CountingEvaluator(AgentTool):
    name = "evaluate_image"

    def __init__(self):
        self.calls = 0

    def execute(self, arguments):
        self.calls += 1
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS,
            outputs={"recognized_texts": list(arguments.get("expected_texts") or [])},
            metrics={
                "marketing_alignment": .9,
                "text_accuracy": 1.0,
                "text_uniqueness": 1.0,
                "text_cleanliness": 1.0,
            },
        )


def registry():
    value = ToolRegistry()
    value.register(MockGenerateTool())
    value.register(MockEditTool())
    evaluator = CountingEvaluator()
    value.register(evaluator)
    register_creative_tools(value, strict_typography=False)
    return value, evaluator


def test_each_format_is_generated_and_evaluated_once():
    tools, evaluator = registry()
    request = TaskRequest(
        prompt="生成高端精华活动海报《焕亮新生》",
        orchestration_mode="multi_agent",
        output_formats=[OutputFormat.SQUARE, OutputFormat.PORTRAIT],
    )
    result = run_multi_agent_campaign(MultiAgentCampaign(request=request), tools)
    assert result.status == CampaignStatus.COMPLETED
    assert {x.output_format for x in result.assets} == {OutputFormat.SQUARE, OutputFormat.PORTRAIT}
    assert evaluator.calls == 2
    assert len(result.quality_reviews) == 2
    assert len(result.compliance_reviews) == 2
    assert result.director_records
    assert all(record.decision.reason for record in result.director_records)
