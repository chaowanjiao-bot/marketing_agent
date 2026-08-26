from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from ..agents import (
    BriefAgent, ComplianceAgent, CreativeDirectorAgent, GenerationAgent,
    LayoutAgent, QualityCriticAgent, build_strategy_team,
)
from ..agents.base import StructuredModel
from ..campaign_checkpoint_store import CampaignCheckpointStore
from ..contracts import (
    AgentMessage, AgentRole, CampaignCheckpoint, CampaignStatus, DirectorRecord,
    MessageType, ComplianceReview, MultiAgentCampaign, SharedEvaluation,
)
from ..resilience import RetryPolicy, execute_with_retry
from ..tools import ToolRegistry


class MultiAgentGraphState(TypedDict):
    campaign: MultiAgentCampaign
    route: str
    director_reason: str


class MultiAgentOptions(BaseModel):
    parallel_strategies: bool = True
    enable_compliance_agent: bool = True
    enable_structured_layout_repair: bool = True
    max_node_attempts: int = 2
    enable_checkpoints: bool = True
    checkpoint_directory: str | None = None


def _append(campaign: MultiAgentCampaign, updates: dict[str, Any]) -> MultiAgentCampaign:
    data: dict[str, Any] = {}
    for key, value in updates.items():
        if key == "layouts":
            data[key] = list(value)
        elif key in {
            "messages", "strategies", "assets", "quality_reviews", "compliance_reviews",
            "shared_evaluations", "director_records", "conflict_resolutions",
            "run_metrics", "checkpoints",
        }:
            data[key] = list(getattr(campaign, key)) + list(value)
        elif key == "generation_increment":
            data["budget"] = campaign.budget.model_copy(update={"generations": campaign.budget.generations + int(value)})
        else:
            data[key] = value
    return campaign.model_copy(update=data)


def build_multi_agent_graph(
    registry: ToolRegistry,
    *,
    model: StructuredModel | None = None,
    quality_threshold: float = 0.75,
    options: MultiAgentOptions | None = None,
):
    options = options or MultiAgentOptions()
    director = CreativeDirectorAgent()
    brief = BriefAgent(model)
    strategies = build_strategy_team(model)
    layout = LayoutAgent(model, registry.restricted(LayoutAgent.allowed_tools))
    generation = GenerationAgent(registry.restricted(GenerationAgent.allowed_tools))
    quality = QualityCriticAgent(registry.restricted(QualityCriticAgent.allowed_tools), quality_threshold)
    compliance = ComplianceAgent(registry.restricted(ComplianceAgent.allowed_tools))
    retry_policy = RetryPolicy(max_attempts=options.max_node_attempts)
    checkpoint_store = (
        CampaignCheckpointStore(Path(options.checkpoint_directory))
        if options.checkpoint_directory else None
    )

    def persist(campaign: MultiAgentCampaign) -> MultiAgentCampaign:
        if checkpoint_store is not None:
            checkpoint_store.save(campaign)
        return campaign

    def execute_node(
        campaign: MultiAgentCampaign, *, node: str, role: AgentRole,
        operation, estimated_cost_usd: float = 0.0,
    ) -> MultiAgentCampaign:
        updates, metrics = execute_with_retry(
            node=node, agent=role, operation=operation, policy=retry_policy,
            estimated_cost_usd=estimated_cost_usd,
        )
        merged = _append(campaign, updates)
        checkpoint = CampaignCheckpoint(
            node=node, director_round=merged.budget.director_rounds,
            generation_count=merged.budget.generations,
            asset_ids=[asset.asset_id for asset in merged.assets],
        )
        extras: dict[str, Any] = {"run_metrics": metrics}
        if options.enable_checkpoints:
            extras["checkpoints"] = [checkpoint]
        return persist(_append(merged, extras))

    def director_node(state: MultiAgentGraphState) -> dict[str, Any]:
        campaign = state["campaign"]
        if campaign.strategies and campaign.selected_strategy_id is None:
            campaign = campaign.model_copy(update={"selected_strategy_id": director.select_strategy(campaign)})
        if campaign.quality_reviews and campaign.compliance_reviews:
            latest_asset = campaign.assets[-1].asset_id if campaign.assets else None
            if (
                latest_asset
                and campaign.quality_reviews[-1].asset_id == latest_asset
                and campaign.compliance_reviews[-1].asset_id == latest_asset
                and not (campaign.quality_reviews[-1].passed and campaign.compliance_reviews[-1].passed)
                and campaign.revision_plan is None
            ):
                campaign = campaign.model_copy(update={"revision_plan": director.build_revision_plan(campaign)})
        budget = campaign.budget.model_copy(update={"director_rounds": campaign.budget.director_rounds + 1})
        campaign = campaign.model_copy(update={"budget": budget})
        decision = director.decide(campaign)
        record = DirectorRecord(
            round=budget.director_rounds,
            decision=decision,
            state_summary={
                "status": campaign.status.value,
                "strategies": len(campaign.strategies),
                "layouts": len(campaign.layouts),
                "assets": len(campaign.assets),
                "latest_asset_id": campaign.assets[-1].asset_id if campaign.assets else None,
                "generations": campaign.budget.generations,
            },
        )
        conflicts = list(campaign.conflict_resolutions)
        conflict = director.resolve_conflict(campaign)
        if conflict and not any(x.asset_id == conflict.asset_id and x.issue == conflict.issue for x in conflicts):
            conflicts.append(conflict)
        campaign = campaign.model_copy(update={
            "director_records": campaign.director_records + [record],
            "conflict_resolutions": conflicts,
        })
        recipients = decision.next_agents or [AgentRole.HUMAN]
        director_messages = [AgentMessage(
            correlation_id=campaign.campaign_id,
            sender=AgentRole.DIRECTOR,
            receiver=receiver,
            message_type=MessageType.APPROVAL if decision.terminal else MessageType.TASK,
            payload=decision.model_dump(mode="json"),
            evidence=[decision.reason],
            confidence=1.0,
        ) for receiver in recipients]
        campaign = campaign.model_copy(update={"messages": campaign.messages + director_messages})
        if decision.terminal:
            campaign = campaign.model_copy(update={
                "status": decision.requested_status,
                "terminal_reason": decision.reason,
            })
            return {"campaign": campaign, "route": "end", "director_reason": decision.reason}
        route = _route_name(decision.next_agents)
        if route == "review_panel" and campaign.assets and campaign.evaluation_for(campaign.assets[-1].asset_id) is None:
            route = "evaluation"
        return {"campaign": campaign, "route": route, "director_reason": decision.reason}

    def brief_node(state: MultiAgentGraphState) -> dict[str, Any]:
        campaign = state["campaign"]
        return {"campaign": execute_node(
            campaign, node="brief", role=AgentRole.BRIEF,
            operation=lambda: brief.act(campaign), estimated_cost_usd=0.001,
        )}

    def strategy_panel_node(state: MultiAgentGraphState) -> dict[str, Any]:
        campaign = state["campaign"]
        def run_strategy(agent):
            return execute_with_retry(
                node="strategy_panel", agent=agent.role,
                operation=lambda: agent.act(campaign), policy=retry_policy,
                estimated_cost_usd=0.002,
            )
        if options.parallel_strategies:
            with ThreadPoolExecutor(max_workers=len(strategies), thread_name_prefix="strategy-agent") as pool:
                results = list(pool.map(run_strategy, strategies))
        else:
            results = [run_strategy(agent) for agent in strategies]
        merged = campaign
        metrics = []
        for result, run_metrics in results:
            merged = _append(merged, result)
            metrics.extend(run_metrics)
        extras: dict[str, Any] = {"run_metrics": metrics}
        if options.enable_checkpoints:
            extras["checkpoints"] = [CampaignCheckpoint(
                node="strategy_panel", director_round=merged.budget.director_rounds,
                generation_count=merged.budget.generations,
                asset_ids=[asset.asset_id for asset in merged.assets],
            )]
        return {"campaign": persist(_append(merged, extras))}

    def layout_node(state: MultiAgentGraphState) -> dict[str, Any]:
        campaign = state["campaign"]
        if campaign.revision_plan and not options.enable_structured_layout_repair:
            campaign = campaign.model_copy(update={
                "revision_plan": campaign.revision_plan.model_copy(update={
                    "tasks": [task.model_copy(update={"target_agent": AgentRole.GENERATION}) for task in campaign.revision_plan.tasks]
                })
            })
        updates, metrics = execute_with_retry(
            node="layout", agent=AgentRole.LAYOUT, operation=lambda: layout.act(campaign),
            policy=retry_policy, estimated_cost_usd=0.002,
        )
        if campaign.revision_plan:
            tasks = [
                task.model_copy(update={"target_agent": AgentRole.GENERATION})
                if task.target_agent == AgentRole.LAYOUT else task
                for task in campaign.revision_plan.tasks
            ]
            updates["revision_plan"] = campaign.revision_plan.model_copy(update={"tasks": tasks})
        # A revised layout invalidates the previous review but keeps asset history.
        merged = _append(campaign, updates)
        checkpoint = CampaignCheckpoint(
            node="layout", director_round=merged.budget.director_rounds,
            generation_count=merged.budget.generations,
            asset_ids=[asset.asset_id for asset in merged.assets],
        )
        extras: dict[str, Any] = {"run_metrics": metrics}
        if options.enable_checkpoints:
            extras["checkpoints"] = [checkpoint]
        return {"campaign": persist(_append(merged, extras))}

    def generation_node(state: MultiAgentGraphState) -> dict[str, Any]:
        campaign = state["campaign"]
        return {"campaign": execute_node(
            campaign, node="generation", role=AgentRole.GENERATION,
            operation=lambda: generation.act(campaign), estimated_cost_usd=0.04,
        )}

    def review_panel_node(state: MultiAgentGraphState) -> dict[str, Any]:
        campaign = state["campaign"]
        def run_review(agent):
            return execute_with_retry(
                node="review_panel", agent=agent.role,
                operation=lambda: agent.act(campaign), policy=retry_policy,
                estimated_cost_usd=0.003,
            )
        if options.enable_compliance_agent:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="review-agent") as pool:
                quality_future = pool.submit(run_review, quality)
                compliance_future = pool.submit(run_review, compliance)
                results = [quality_future.result(), compliance_future.result()]
        else:
            quality_result = run_review(quality)
            results = [quality_result, ({
                "compliance_reviews": [ComplianceReview(asset_id=campaign.assets[-1].asset_id, passed=True)],
                "messages": [],
            }, [])]
        merged = campaign
        metrics = []
        for result, run_metrics in results:
            merged = _append(merged, result)
            metrics.extend(run_metrics)
        extras: dict[str, Any] = {"run_metrics": metrics}
        if options.enable_checkpoints:
            extras["checkpoints"] = [CampaignCheckpoint(
                node="review_panel", director_round=merged.budget.director_rounds,
                generation_count=merged.budget.generations,
                asset_ids=[asset.asset_id for asset in merged.assets],
            )]
        return {"campaign": persist(_append(merged, extras))}

    def evaluation_node(state: MultiAgentGraphState) -> dict[str, Any]:
        campaign = state["campaign"]
        asset = campaign.assets[-1]
        def evaluate() -> dict[str, Any]:
            observation = registry.get("evaluate_image").execute({
                "image_path": asset.file_path,
                "campaign_text": campaign.request.prompt,
                "attempt": campaign.budget.generations,
                "expected_texts": campaign.brief.required_elements if campaign.brief else [],
            })
            return {"shared_evaluations": [SharedEvaluation(
                asset_id=asset.asset_id, observation=observation,
            )]}
        return {"campaign": execute_node(
            campaign, node="evaluation", role=AgentRole.QUALITY,
            operation=evaluate, estimated_cost_usd=0.006,
        )}

    builder = StateGraph(MultiAgentGraphState)
    builder.add_node("director", director_node)
    builder.add_node("brief", brief_node)
    builder.add_node("strategy_panel", strategy_panel_node)
    builder.add_node("layout", layout_node)
    builder.add_node("generation", generation_node)
    builder.add_node("review_panel", review_panel_node)
    builder.add_node("evaluation", evaluation_node)
    builder.add_edge(START, "director")
    builder.add_conditional_edges("director", lambda state: state["route"], {
        "brief": "brief", "strategy_panel": "strategy_panel", "layout": "layout",
        "generation": "generation", "evaluation": "evaluation", "review_panel": "review_panel", "end": END,
    })
    for node in ("brief", "strategy_panel", "layout", "generation", "evaluation", "review_panel"):
        builder.add_edge(node, "director")
    return builder.compile()


def _route_name(roles: list[AgentRole]) -> str:
    role_set = set(roles)
    if not roles:
        return "end"
    if role_set == {AgentRole.BRIEF}:
        return "brief"
    if role_set == {AgentRole.BRAND_STRATEGY, AgentRole.PERFORMANCE_STRATEGY, AgentRole.MERCHANDISING_STRATEGY}:
        return "strategy_panel"
    if role_set == {AgentRole.LAYOUT}:
        return "layout"
    if role_set == {AgentRole.GENERATION}:
        return "generation"
    if role_set == {AgentRole.QUALITY, AgentRole.COMPLIANCE}:
        return "review_panel"
    raise ValueError(f"unsupported director route: {sorted(x.value for x in roles)}")


def run_multi_agent_campaign(
    campaign: MultiAgentCampaign,
    registry: ToolRegistry,
    *,
    model: StructuredModel | None = None,
    quality_threshold: float = 0.75,
    options: MultiAgentOptions | None = None,
) -> MultiAgentCampaign:
    graph = build_multi_agent_graph(
        registry, model=model, quality_threshold=quality_threshold, options=options
    )
    result = graph.invoke(
        {"campaign": campaign, "route": "director", "director_reason": "start"},
        config={"recursion_limit": campaign.budget.max_director_rounds * 3 + 10},
    )
    return result["campaign"]
