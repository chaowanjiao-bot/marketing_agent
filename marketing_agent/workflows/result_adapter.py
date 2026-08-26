from __future__ import annotations

from ..contracts import CampaignStatus, MultiAgentCampaign
from ..observability import campaign_run_report
from ..schemas import (
    FinalResult, FormatSummary, GoalSpec, ReviewStatus, TaskType,
)


def campaign_to_final_result(campaign: MultiAgentCampaign) -> FinalResult:
    task_type = TaskType.IMAGE_EDIT if campaign.request.input_image else TaskType.TEXT_TO_IMAGE
    status = {
        CampaignStatus.COMPLETED: "completed",
        CampaignStatus.WAITING_FOR_REVIEW: "waiting_for_review",
        CampaignStatus.WAITING_FOR_USER: "waiting_for_user",
        CampaignStatus.ABORTED: "aborted",
        CampaignStatus.FAILED: "failed",
    }.get(campaign.status, "completed")
    format_summaries = []
    selected_assets = []
    for layout in campaign.layouts:
        assets = [x for x in campaign.assets if x.output_format == layout.output_format]
        asset = assets[-1] if assets else None
        if asset:
            selected_assets.append(asset)
        quality = next((x for x in reversed(campaign.quality_reviews) if asset and x.asset_id == asset.asset_id), None)
        compliance = next((x for x in reversed(campaign.compliance_reviews) if asset and x.asset_id == asset.asset_id), None)
        format_summaries.append(FormatSummary(
            output_format=layout.output_format,
            width=layout.width,
            height=layout.height,
            best_asset_id=asset.asset_id if asset else None,
            best_score=quality.overall_score if quality else None,
            compliant=bool(compliance and compliance.passed),
        ))
    primary = selected_assets[0] if selected_assets else (campaign.assets[-1] if campaign.assets else None)
    primary_quality = next((x for x in reversed(campaign.quality_reviews) if primary and x.asset_id == primary.asset_id), None)
    primary_compliance = next((x for x in reversed(campaign.compliance_reviews) if primary and x.asset_id == primary.asset_id), None)
    trace = [{
        "event": "agent_message",
        **message.model_dump(mode="json"),
    } for message in campaign.messages]
    trace.extend({
        "event": "director_decision",
        **record.model_dump(mode="json"),
    } for record in campaign.director_records)
    trace.extend({
        "event": "conflict_resolution",
        **conflict.model_dump(mode="json"),
    } for conflict in campaign.conflict_resolutions)
    trace.extend({
        "event": "agent_run_metric",
        **metric.model_dump(mode="json"),
    } for metric in campaign.run_metrics)
    trace.extend({
        "event": "campaign_checkpoint",
        **checkpoint.model_dump(mode="json"),
    } for checkpoint in campaign.checkpoints)
    trace.append({
        "event": "campaign_run_report",
        **campaign_run_report(campaign),
    })
    trace.append({
        "event": "multi_agent_campaign_completed",
        "campaign_id": campaign.campaign_id,
        "status": campaign.status.value,
        "director_rounds": campaign.budget.director_rounds,
        "generations": campaign.budget.generations,
    })
    return FinalResult(
        status=status,
        terminal_reason=campaign.terminal_reason or campaign.status.value,
        goal=GoalSpec(business_goal=campaign.request.prompt, task_type=task_type),
        assets=campaign.assets,
        observations=[x.observation for x in campaign.shared_evaluations],
        trace=trace,
        best_asset_id=primary.asset_id if primary else None,
        best_score=primary_quality.overall_score if primary_quality else None,
        best_aesthetic_asset_id=primary.asset_id if primary else None,
        best_aesthetic_score=primary_quality.overall_score if primary_quality else None,
        best_compliant_asset_id=primary.asset_id if primary and primary_compliance and primary_compliance.passed else None,
        best_compliant_score=primary_quality.overall_score if primary_quality and primary_compliance and primary_compliance.passed else None,
        brand_id=campaign.request.brand_profile.brand_id if campaign.request.brand_profile else None,
        format_summaries=format_summaries,
        primary_output_format=campaign.request.output_formats[0],
        review_status=ReviewStatus.WAITING if campaign.status == CampaignStatus.WAITING_FOR_REVIEW else ReviewStatus.NOT_REQUIRED,
        review_round=campaign.request.review_round,
    )
