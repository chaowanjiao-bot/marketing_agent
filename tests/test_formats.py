import pytest
from pydantic import ValidationError

from marketing_agent.candidates import run_campaign_batch
from marketing_agent.formats import OutputFormat, get_format_spec
from marketing_agent.schemas import Observation, ObservationStatus, TaskRequest
from marketing_agent.tools import AgentTool, MockEditTool, ToolRegistry


class FormatGenerator(AgentTool):
    name = "generate_image"

    def execute(self, arguments):
        return Observation(
            tool_name=self.name, status=ObservationStatus.SUCCESS,
            outputs={
                "file_path": f"{arguments['output_format']}_{arguments['seed']}.png",
                "prompt": arguments["prompt"], "seed": arguments["seed"],
                "width": arguments["width"], "height": arguments["height"],
                "output_format": arguments["output_format"],
            },
        )


class PassingEvaluator(AgentTool):
    name = "evaluate_image"

    def execute(self, arguments):
        return Observation(
            tool_name=self.name, status=ObservationStatus.SUCCESS,
            metrics={"marketing_alignment": 0.9, "text_accuracy": 1.0,
                     "text_uniqueness": 1.0},
        )


def registry() -> ToolRegistry:
    tools = ToolRegistry()
    tools.register(FormatGenerator())
    tools.register(PassingEvaluator())
    tools.register(MockEditTool())
    return tools


def test_supported_format_dimensions_are_model_safe() -> None:
    expected = {
        OutputFormat.SQUARE: (1024, 1024), OutputFormat.PORTRAIT: (1024, 1280),
        OutputFormat.STORY: (768, 1360), OutputFormat.LANDSCAPE: (1360, 768),
    }
    for output_format, dimensions in expected.items():
        spec = get_format_spec(output_format)
        assert (spec.width, spec.height) == dimensions
        assert spec.width % 8 == 0 and spec.height % 8 == 0


def test_campaign_generates_and_selects_each_requested_format() -> None:
    formats = list(OutputFormat)
    result = run_campaign_batch(TaskRequest(
        prompt="生成高端护肤品全渠道活动素材", output_formats=formats,
        candidate_count=2, max_iterations=4,
    ), registry())
    assert result.primary_output_format == OutputFormat.SQUARE
    assert [item.output_format for item in result.format_summaries] == formats
    assert len(result.assets) == 8
    assert len(result.candidate_summaries) == 8
    for summary in result.format_summaries:
        asset = next(asset for asset in result.assets if asset.asset_id == summary.best_asset_id)
        assert (asset.width, asset.height) == (summary.width, summary.height)
        assert asset.output_format == summary.output_format


def test_duplicate_formats_and_edit_variants_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unique"):
        TaskRequest(prompt="生成产品广告", output_formats=["1:1", "1:1"])
    with pytest.raises(ValidationError, match="text-to-image"):
        TaskRequest(prompt="编辑产品图", input_asset_id="upload_000000000000",
                    output_formats=["4:5"])
