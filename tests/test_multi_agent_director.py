from marketing_agent.agents.director import CreativeDirectorAgent
from marketing_agent.contracts import AgentRole, CampaignStatus, MultiAgentCampaign
from marketing_agent.contracts.multi_agent import CreativeBrief
from marketing_agent.schemas import AssetVersion, TaskRequest
from marketing_agent.contracts import ComplianceReview, QualityReview


def campaign(**updates):
    value = MultiAgentCampaign(request=TaskRequest(prompt="生成高端精华活动海报"))
    return value.model_copy(update=updates)


def test_director_starts_with_brief_agent():
    decision = CreativeDirectorAgent().decide(campaign())
    assert decision.next_agents == [AgentRole.BRIEF]


def test_director_requests_parallel_strategy_panel():
    brief = CreativeBrief(product_name="精华", channels=["活动海报"])
    decision = CreativeDirectorAgent().decide(campaign(brief=brief, status=CampaignStatus.RUNNING))
    assert decision.execution_mode == "parallel"
    assert set(decision.next_agents) == {
        AgentRole.BRAND_STRATEGY,
        AgentRole.PERFORMANCE_STRATEGY,
        AgentRole.MERCHANDISING_STRATEGY,
    }


def test_budget_is_a_hard_terminal_guard():
    value = campaign()
    value = value.model_copy(update={
        "budget": value.budget.model_copy(update={
            "director_rounds": value.budget.max_director_rounds
        })
    })
    decision = CreativeDirectorAgent().decide(value)
    assert decision.terminal
    assert decision.requested_status == CampaignStatus.ABORTED


def test_compliance_wins_conflict_against_aesthetic_approval():
    value = campaign(
        assets=[AssetVersion(tool_name="generate_image", file_path="asset.png", prompt="x", seed=1, asset_id="asset_x")],
        quality_reviews=[QualityReview(asset_id="asset_x", overall_score=.9, passed=True)],
        compliance_reviews=[ComplianceReview(asset_id="asset_x", passed=False, hard_veto=True)],
    )
    conflict = CreativeDirectorAgent().resolve_conflict(value)
    assert conflict is not None
    assert conflict.winning_constraint == "compliance_hard_gate"
