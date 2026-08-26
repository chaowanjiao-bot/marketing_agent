from __future__ import annotations

from typing import Any

from ..brief import MarketingInputInterpreter
from ..copy_agent import MarketingCopyAgent
from ..contracts import AgentRole, CampaignStatus, CreativeBrief
from .base import SpecialistAgent


class BriefAgent(SpecialistAgent):
    role = AgentRole.BRIEF
    allowed_tools = frozenset()

    SYSTEM_PROMPT = """You are a marketing brief specialist. Convert a user request into a
strict CreativeBrief. Never invent price, dates, legal claims, brand rules or product facts.
Mark missing decision-critical fields as ambiguities and ask concise clarification questions."""

    def act(self, campaign) -> dict[str, Any]:
        request = campaign.request
        if self.model is not None:
            brief = self.model.invoke(
                system=self.SYSTEM_PROMPT,
                user=request.model_dump_json(exclude={"memory_context", "experience_strategies"}),
                response_model=CreativeBrief,
            )
        else:
            fallback = MarketingInputInterpreter().interpret(request.prompt, request.creativity)
            brand = request.brand_profile
            copy = MarketingCopyAgent().create(fallback, brand) if request.generate_copy else None
            brief = CreativeBrief(
                product_name=fallback.subject,
                campaign_type=fallback.scene or "marketing_creative",
                channels=[fallback.scene or "活动海报"],
                headline=copy.headline if copy else None,
                subheadline=copy.subheadline if copy else None,
                call_to_action=copy.call_to_action if copy else None,
                brand_tone=list(brand.tone) if brand else ([fallback.style] if fallback.style else []),
                brand_colors=list(brand.primary_colors) if brand else ([fallback.primary_color] if fallback.primary_color else []),
                required_elements=copy.required_text if copy else (list(brand.required_phrases) if brand else []),
                forbidden_elements=list(brand.forbidden_phrases) if brand else [],
                compliance_rules=list(brand.visual_rules) if brand else [],
                output_formats=request.output_formats,
                ambiguities=fallback.ambiguities,
                clarification_questions=fallback.clarification_questions,
            )
        status = CampaignStatus.RUNNING if brief.ready else CampaignStatus.WAITING_FOR_USER
        msg = self.message(
            campaign, AgentRole.DIRECTOR, {"brief_ready": brief.ready},
            evidence=["CreativeBrief schema validation passed"], confidence=0.9 if self.model else 0.65,
        )
        return {"brief": brief, "status": status, "messages": [msg]}
