from __future__ import annotations

from ..contracts import (
    AgentRole, CampaignStatus, ConflictResolution, DirectorDecision, MultiAgentCampaign,
    RepairOperation, RevisionPlan, RevisionTask,
)


class CreativeDirectorAgent:
    """Deterministic safety-first supervisor; an LLM planner can replace policy selection."""

    role = AgentRole.DIRECTOR
    PRIORITY = {
        "legal": 100, "platform": 95, "brand": 90, "text": 85,
        "product": 80, "user": 70, "conversion": 60, "aesthetic": 50,
    }

    def decide(self, campaign: MultiAgentCampaign) -> DirectorDecision:
        budget = campaign.budget
        if budget.exhausted:
            return DirectorDecision(
                next_agents=[], reason="execution budget exhausted", terminal=True,
                requested_status=CampaignStatus.ABORTED,
            )
        if campaign.status == CampaignStatus.WAITING_FOR_USER:
            return DirectorDecision(
                next_agents=[], reason="decision-critical brief fields are missing", terminal=True,
                requested_status=CampaignStatus.WAITING_FOR_USER,
            )
        if campaign.brief is None:
            return DirectorDecision(next_agents=[AgentRole.BRIEF], reason="build a validated creative brief")
        if not campaign.brief.ready:
            return DirectorDecision(
                next_agents=[], reason="brief requires user clarification", terminal=True,
                requested_status=CampaignStatus.WAITING_FOR_USER,
            )
        if not campaign.strategies:
            return DirectorDecision(
                next_agents=[AgentRole.BRAND_STRATEGY, AgentRole.PERFORMANCE_STRATEGY, AgentRole.MERCHANDISING_STRATEGY],
                execution_mode="parallel", reason="explore three complementary creative strategies",
                success_criteria=["three schema-valid strategies", "all hard constraints preserved"],
            )
        if campaign.selected_strategy_id is None:
            return DirectorDecision(next_agents=[AgentRole.DIRECTOR], reason="rank and select a strategy")
        if not campaign.layouts:
            return DirectorDecision(next_agents=[AgentRole.LAYOUT], reason="translate strategy into executable layouts")
        if campaign.revision_plan and any(t.target_agent == AgentRole.LAYOUT for t in campaign.revision_plan.tasks):
            return DirectorDecision(next_agents=[AgentRole.LAYOUT], reason="apply evidence-backed layout or typography repair")
        if not campaign.assets or self._revision_requires_generation(campaign.revision_plan):
            return DirectorDecision(next_agents=[AgentRole.GENERATION], reason="generate or revise the visual asset")
        quality = campaign.quality_reviews[-1] if campaign.quality_reviews else None
        compliance = campaign.compliance_reviews[-1] if campaign.compliance_reviews else None
        latest_asset = campaign.assets[-1].asset_id
        if campaign.evaluation_for(latest_asset) is None:
            return DirectorDecision(
                next_agents=[AgentRole.QUALITY, AgentRole.COMPLIANCE], execution_mode="parallel",
                reason="evaluate the latest asset once, then fan out the shared observation",
            )
        if quality is None or compliance is None or quality.asset_id != latest_asset or compliance.asset_id != latest_asset:
            return DirectorDecision(
                next_agents=[AgentRole.QUALITY, AgentRole.COMPLIANCE], execution_mode="parallel",
                reason="run independent aesthetic and compliance reviews",
            )
        if quality.passed and compliance.passed:
            if not self._all_formats_passed(campaign):
                return DirectorDecision(
                    next_agents=[AgentRole.GENERATION],
                    reason="current format passed; continue with the next requested format",
                )
            status = CampaignStatus.WAITING_FOR_REVIEW if campaign.request.review_required else CampaignStatus.COMPLETED
            return DirectorDecision(
                next_agents=[], reason="quality and hard compliance gates passed", terminal=True,
                requested_status=status,
            )
        return DirectorDecision(
            next_agents=[self._revision_target(campaign)], reason="route evidence-backed revision",
            success_criteria=["blocking findings resolved", "already-passed constraints preserved"],
        )

    def select_strategy(self, campaign: MultiAgentCampaign) -> str:
        if not campaign.strategies:
            raise ValueError("strategy selection requires candidates")
        # Prefer confidence, then fewer declared risks. Replace with pairwise LLM ranking if configured.
        selected = max(campaign.strategies, key=lambda x: (x.confidence, -len(x.risks)))
        return selected.strategy_id

    def build_revision_plan(self, campaign: MultiAgentCampaign) -> RevisionPlan:
        findings = []
        if campaign.quality_reviews:
            findings.extend(campaign.quality_reviews[-1].findings)
        if campaign.compliance_reviews:
            findings.extend(campaign.compliance_reviews[-1].findings)
        findings.sort(key=lambda x: ({"blocking": 4, "high": 3, "medium": 2, "low": 1}[x.severity], -x.score), reverse=True)
        tasks = [
            RevisionTask(
                target_agent=f.target_agent, action=f.proposed_action,
                operation=self._repair_operation(f.dimension, f.target_agent),
                parameters=self._repair_parameters(f.dimension, f.evidence),
                source_finding_ids=[f.finding_id],
                preserve=["all constraints not mentioned by this finding"], priority=index + 1,
            )
            for index, f in enumerate(findings[:3])
        ]
        return RevisionPlan(reason="top evidence-backed findings", tasks=tasks)

    def resolve_conflict(self, campaign: MultiAgentCampaign) -> ConflictResolution | None:
        if not campaign.assets or not campaign.quality_reviews or not campaign.compliance_reviews:
            return None
        quality = campaign.quality_reviews[-1]
        compliance = campaign.compliance_reviews[-1]
        if quality.asset_id != compliance.asset_id or quality.passed == compliance.passed:
            return None
        if not compliance.passed:
            return ConflictResolution(
                asset_id=quality.asset_id,
                parties=[AgentRole.QUALITY, AgentRole.COMPLIANCE],
                issue="aesthetic approval conflicts with a hard compliance failure",
                priority_basis="legal/platform and brand hard constraints outrank aesthetic preference",
                resolution="reject delivery and execute the highest-priority compliance repair",
                winning_constraint="compliance_hard_gate",
            )
        return ConflictResolution(
            asset_id=quality.asset_id,
            parties=[AgentRole.QUALITY, AgentRole.COMPLIANCE],
            issue="compliant asset remains below the visual quality threshold",
            priority_basis="compliance is preserved while soft visual quality is improved",
            resolution="allow a bounded quality revision that must preserve every passed hard rule",
            winning_constraint="preserve_compliance_then_improve_quality",
        )

    @staticmethod
    def _repair_operation(dimension: str, target: AgentRole) -> RepairOperation:
        if dimension == "safe_area":
            return RepairOperation.MOVE_INSIDE_SAFE_AREA
        if dimension in {"text_accuracy", "text_uniqueness", "text_cleanliness", "text_duplication", "brand_rule"}:
            return RepairOperation.RERENDER_TYPOGRAPHY
        if target == AgentRole.LAYOUT:
            return RepairOperation.ADJUST_VISUAL_HIERARCHY
        return RepairOperation.REGENERATE_ASSET

    @staticmethod
    def _repair_parameters(dimension: str, evidence: str) -> dict[str, str]:
        if dimension == "safe_area" and ":" in evidence:
            return {"component_id": evidence.split(":", 1)[0]}
        return {"dimension": dimension}

    @staticmethod
    def _revision_requires_generation(plan: RevisionPlan | None) -> bool:
        return bool(plan and any(t.target_agent == AgentRole.GENERATION for t in plan.tasks))

    @staticmethod
    def _revision_target(campaign: MultiAgentCampaign) -> AgentRole:
        compliance = campaign.compliance_reviews[-1]
        findings = list(compliance.findings)
        if campaign.quality_reviews:
            findings += campaign.quality_reviews[-1].findings
        if not findings:
            return AgentRole.GENERATION
        findings.sort(key=lambda x: {"blocking": 4, "high": 3, "medium": 2, "low": 1}[x.severity], reverse=True)
        return findings[0].target_agent

    @staticmethod
    def _all_formats_passed(campaign: MultiAgentCampaign) -> bool:
        for layout in campaign.layouts:
            assets = [x for x in campaign.assets if x.output_format == layout.output_format]
            if not assets:
                return False
            asset_id = assets[-1].asset_id
            quality = next((x for x in reversed(campaign.quality_reviews) if x.asset_id == asset_id), None)
            compliance = next((x for x in reversed(campaign.compliance_reviews) if x.asset_id == asset_id), None)
            if not quality or not compliance or not quality.passed or not compliance.passed:
                return False
        return True
