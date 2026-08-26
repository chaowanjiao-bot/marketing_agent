from pathlib import Path

from marketing_agent.campaign_checkpoint_store import CampaignCheckpointStore
from marketing_agent.contracts import (
    AgentRole, AgentRunMetric, CampaignCheckpoint, MultiAgentCampaign,
)
from marketing_agent.observability import campaign_run_report
from marketing_agent.resilience import RetryPolicy, execute_with_retry
from marketing_agent.schemas import TaskRequest


def test_retry_records_failed_and_successful_attempts():
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("temporary")
        return "ok"

    value, metrics = execute_with_retry(
        node="generation", agent=AgentRole.GENERATION, operation=flaky,
        policy=RetryPolicy(max_attempts=2),
    )
    assert value == "ok"
    assert [item.status for item in metrics] == ["failed", "succeeded"]


def test_report_and_checkpoint_round_trip(tmp_path: Path):
    campaign = MultiAgentCampaign(
        request=TaskRequest(prompt="新品海报"),
        run_metrics=[AgentRunMetric(
            agent=AgentRole.BRIEF, node="brief", status="succeeded",
            duration_ms=12.5, estimated_cost_usd=0.001,
        )],
        checkpoints=[CampaignCheckpoint(
            node="brief", director_round=1, generation_count=0,
        )],
    )
    report = campaign_run_report(campaign)
    assert report["agent_runs"] == 1
    assert report["estimated_cost_usd"] == 0.001

    store = CampaignCheckpointStore(tmp_path)
    store.save(campaign)
    restored = store.load(campaign.campaign_id)
    assert restored == campaign
