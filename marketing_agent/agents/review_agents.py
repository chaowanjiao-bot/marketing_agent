from __future__ import annotations

from typing import Any

from ..contracts import (
    AgentRole, ComplianceReview, QualityReview, ReviewFinding,
)
from ..schemas import ObservationStatus
from ..tools import ToolRegistry
from .base import SpecialistAgent


TEXT_DIMENSIONS = {"text_accuracy", "text_uniqueness", "text_cleanliness", "text_duplication"}


class QualityCriticAgent(SpecialistAgent):
    role = AgentRole.QUALITY
    allowed_tools = frozenset()

    def __init__(self, registry: ToolRegistry, threshold: float = 0.75) -> None:
        super().__init__(None); self.registry = registry; self.threshold = threshold

    def act(self, campaign) -> dict[str, Any]:
        if not campaign.assets:
            raise ValueError("quality review requires an asset")
        asset = campaign.assets[-1]
        shared = campaign.evaluation_for(asset.asset_id)
        if shared is None:
            raise ValueError("quality review requires a shared evaluation")
        observation = shared.observation
        score = float(observation.metrics.get("marketing_alignment", 0.0))
        findings = []
        for issue in observation.issues:
            if issue in TEXT_DIMENSIONS:
                continue
            findings.append(ReviewFinding(
                dimension=issue, score=float(observation.metrics.get(issue, 0.0)),
                severity="high" if float(observation.metrics.get(issue, 0.0)) < .5 else "medium",
                evidence=f"evaluator reported low {issue} for asset {asset.asset_id}",
                proposed_action=self._action(issue), target_agent=self._target(issue),
            ))
        review = QualityReview(
            asset_id=asset.asset_id,
            overall_score=score,
            passed=score >= self.threshold and not any(x.severity in {"high", "blocking"} for x in findings),
            findings=findings,
        )
        msg = self.message(campaign, AgentRole.DIRECTOR, {"asset_id": asset.asset_id, "passed": review.passed})
        return {"quality_reviews": [review], "messages": [msg]}

    @staticmethod
    def _target(dimension: str) -> AgentRole:
        return AgentRole.LAYOUT if dimension in {"composition", "spatial", "focus"} else AgentRole.GENERATION

    @staticmethod
    def _action(dimension: str) -> str:
        return f"Improve only {dimension}; preserve all already-passed brand, product and typography constraints."


class ComplianceAgent(SpecialistAgent):
    role = AgentRole.COMPLIANCE
    allowed_tools = frozenset({"brand_rule_engine", "safe_area_validator"})

    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(None); self.registry = registry

    def act(self, campaign) -> dict[str, Any]:
        if not campaign.assets or campaign.brief is None:
            raise ValueError("compliance review requires asset and brief")
        asset = campaign.assets[-1]
        shared = campaign.evaluation_for(asset.asset_id)
        if shared is None:
            raise ValueError("compliance review requires a shared evaluation")
        observation = shared.observation
        findings = []
        for dimension in TEXT_DIMENSIONS:
            value = float(observation.metrics.get(dimension, 1.0))
            if value < 1.0 or dimension in observation.issues:
                findings.append(ReviewFinding(
                    dimension=dimension, score=value, severity="blocking",
                    evidence=f"OCR compliance metric {dimension}={value:.3f}",
                    violated_rule="critical_copy_must_match_exactly",
                    proposed_action="Re-render critical text as a deterministic typography layer.",
                    target_agent=AgentRole.LAYOUT,
                ))
        # Brand phrase checks are deterministic and complement the visual evaluator.
        recognized = [str(x) for x in observation.outputs.get("recognized_texts", [])]
        brand_check = self.registry.get("brand_rule_engine").execute({
            "recognized_texts": recognized,
            "required_phrases": campaign.brief.required_elements,
            "forbidden_phrases": campaign.brief.forbidden_elements,
            "expected_logo_count": 1 if campaign.request.brand_profile else None,
            "detected_logo_count": observation.outputs.get("detected_logo_count"),
            "primary_color_delta_e": observation.outputs.get("primary_color_delta_e"),
        })
        for issue in brand_check.issues:
            findings.append(ReviewFinding(
                dimension="brand_rule", score=0.0, severity="blocking",
                evidence=issue, violated_rule=issue,
                proposed_action="Repair the deterministic typography layer according to the brand rule.",
                target_agent=AgentRole.LAYOUT,
            ))
        current_layout = next(
            (x for x in campaign.layouts if x.output_format == asset.output_format), None
        )
        if current_layout is not None:
            safe_area = self.registry.get("safe_area_validator").execute({
                "components": [x.model_dump() for x in current_layout.components],
                "safe_margin": current_layout.safe_margin,
            })
            for issue in safe_area.issues:
                findings.append(ReviewFinding(
                    dimension="safe_area", score=0.0, severity="blocking",
                    evidence=issue, violated_rule="channel_safe_area",
                    proposed_action="Move or resize the named layer inside the channel safe area.",
                    target_agent=AgentRole.LAYOUT,
                ))
        hard_veto = any(x.severity == "blocking" for x in findings)
        review = ComplianceReview(
            asset_id=asset.asset_id, passed=not findings, hard_veto=hard_veto,
            recognized_texts=recognized, findings=findings,
        )
        msg = self.message(
            campaign, AgentRole.DIRECTOR,
            {"asset_id": asset.asset_id, "passed": review.passed, "hard_veto": review.hard_veto},
            evidence=[x.evidence for x in findings],
        )
        return {"compliance_reviews": [review], "messages": [msg]}
