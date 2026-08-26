from marketing_agent.agents.layout_agent import LayoutAgent
from marketing_agent.contracts import (
    AgentRole, CreativeBrief, CreativeStrategy, MultiAgentCampaign,
    RepairOperation, RevisionPlan,
)
from marketing_agent.contracts.multi_agent import RevisionTask
from marketing_agent.schemas import TaskRequest
from marketing_agent.tools import build_default_registry


def test_layout_agent_consumes_structured_safe_area_repair():
    strategy = CreativeStrategy(
        author=AgentRole.BRAND_STRATEGY, name="brand", target_insight="insight",
        core_message="message", visual_concept="concept", composition="centered",
        color_and_lighting="brand colors", call_to_action="buy",
    )
    campaign = MultiAgentCampaign(
        request=TaskRequest(prompt="生成精华活动海报"),
        brief=CreativeBrief(product_name="精华", channels=["活动海报"]),
        strategies=[strategy], selected_strategy_id=strategy.strategy_id,
        revision_plan=RevisionPlan(reason="safe area", tasks=[RevisionTask(
            target_agent=AgentRole.LAYOUT,
            action="move logo inside safe area",
            operation=RepairOperation.MOVE_INSIDE_SAFE_AREA,
            parameters={"component_id":"logo"},
        )]),
    )
    agent = LayoutAgent(registry=build_default_registry().restricted(LayoutAgent.allowed_tools))
    layout = agent.act(campaign)["layouts"][0]
    logo = next(x for x in layout.components if x.component_id == "logo")
    assert logo.y + logo.height <= 1 - layout.safe_margin
    assert any("safe-area repair" in note for note in layout.typography_notes)
