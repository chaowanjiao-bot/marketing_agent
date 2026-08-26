from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..contracts import AgentRole, CreativeStrategy
from .base import SpecialistAgent, StructuredModel


class StrategyDraft(BaseModel):
    name: str
    target_insight: str
    core_message: str
    visual_concept: str
    composition: str
    color_and_lighting: str
    call_to_action: str
    expected_strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class StrategyAgent(SpecialistAgent):
    allowed_tools = frozenset()

    def __init__(self, role: AgentRole, lens: str, model: StructuredModel | None = None) -> None:
        super().__init__(model)
        self.role = role
        self.lens = lens

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior marketing strategist. Propose one distinctive, executable visual "
            f"strategy through this lens: {self.lens}. Respect all hard constraints. Return only "
            "a validated CreativeStrategy and state concrete risks."
        )

    def act(self, campaign) -> dict[str, Any]:
        if campaign.brief is None or not campaign.brief.ready:
            raise ValueError("strategy requires a ready brief")
        if self.model is not None:
            feedback = ""
            if campaign.human_feedback_route and self.role in campaign.human_feedback_route.targets:
                feedback = "\nHuman revision feedback:\n" + campaign.human_feedback_route.feedback
            draft = self.model.invoke(
                system=self.system_prompt,
                user=campaign.brief.model_dump_json() + feedback,
                response_model=StrategyDraft,
            )
            strategy = CreativeStrategy(author=self.role, **draft.model_dump())
        else:
            product = campaign.brief.product_name or "产品"
            role_names = {
                AgentRole.BRAND_STRATEGY: ("品牌叙事", "保持克制一致的品牌资产", "品牌辨识度"),
                AgentRole.PERFORMANCE_STRATEGY: ("转化驱动", "突出卖点与行动号召", "信息转化效率"),
                AgentRole.MERCHANDISING_STRATEGY: ("商品陈列", "强化材质、细节和使用场景", "商品可感知价值"),
            }
            name, message, strength = role_names[self.role]
            strategy = CreativeStrategy(
                author=self.role, name=name, target_insight=f"用户需要快速理解{product}的核心价值",
                core_message=message, visual_concept=f"以{product}为唯一视觉中心的高完成度商业画面",
                composition="主体居中或三分构图，文字与商品使用独立安全区",
                color_and_lighting="遵循品牌色，使用干净棚拍光和克制轮廓光",
                call_to_action=campaign.brief.call_to_action or "了解更多",
                expected_strengths=[strength], risks=["需由合规 Agent 检查关键文案"], confidence=0.65,
            )
            if campaign.human_feedback_route and self.role in campaign.human_feedback_route.targets:
                strategy = strategy.model_copy(update={
                    "core_message": strategy.core_message + "；人工反馈：" + campaign.human_feedback_route.feedback
                })
        msg = self.message(campaign, AgentRole.DIRECTOR, {"strategy_id": strategy.strategy_id})
        return {"strategies": [strategy], "messages": [msg]}


def build_strategy_team(model: StructuredModel | None = None) -> list[StrategyAgent]:
    return [
        StrategyAgent(AgentRole.BRAND_STRATEGY, "brand consistency and long-term equity", model),
        StrategyAgent(AgentRole.PERFORMANCE_STRATEGY, "conversion, offer clarity and CTA", model),
        StrategyAgent(AgentRole.MERCHANDISING_STRATEGY, "product visibility, material and hierarchy", model),
    ]
