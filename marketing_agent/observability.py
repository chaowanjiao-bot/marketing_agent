from __future__ import annotations

from collections import Counter
from typing import Any

from .contracts import MultiAgentCampaign


def campaign_run_report(campaign: MultiAgentCampaign) -> dict[str, Any]:
    metrics = campaign.run_metrics
    status_counts = Counter(item.status for item in metrics)
    agent_counts = Counter(item.agent.value for item in metrics)
    total_duration_ms = sum(item.duration_ms for item in metrics)
    return {
        "campaign_id": campaign.campaign_id,
        "status": campaign.status.value,
        "terminal_reason": campaign.terminal_reason,
        "director_rounds": campaign.budget.director_rounds,
        "generations": campaign.budget.generations,
        "assets": len(campaign.assets),
        "agent_runs": len(metrics),
        "successful_runs": status_counts["succeeded"],
        "failed_attempts": status_counts["failed"],
        "degraded_runs": status_counts["degraded"],
        "retry_count": sum(max(0, item.attempt - 1) for item in metrics),
        "total_agent_duration_ms": round(total_duration_ms, 3),
        "estimated_cost_usd": round(sum(item.estimated_cost_usd for item in metrics), 6),
        "runs_by_agent": dict(sorted(agent_counts.items())),
        "checkpoint_count": len(campaign.checkpoints),
        "latest_checkpoint": (
            campaign.checkpoints[-1].model_dump(mode="json") if campaign.checkpoints else None
        ),
        "quality_passed": bool(campaign.quality_reviews and campaign.quality_reviews[-1].passed),
        "compliance_passed": bool(campaign.compliance_reviews and campaign.compliance_reviews[-1].passed),
        "conflict_count": len(campaign.conflict_resolutions),
    }
