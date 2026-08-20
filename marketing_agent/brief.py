from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class CreativityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketingBrief(BaseModel):
    raw_request: str = Field(min_length=3)
    subject: str | None = None
    scene: str | None = None
    primary_color: str | None = None
    style: str | None = None
    composition: str | None = None
    lighting: str | None = None
    hard_constraints: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    creativity: CreativityLevel = CreativityLevel.MEDIUM


class MarketingInputInterpreter:
    """Deterministic fallback; an MLLM may replace extraction later."""

    COLORS = ("红色", "蓝色", "绿色", "黑色", "白色", "金色", "银色", "象牙白")
    STYLES = ("高端", "极简", "科技", "复古", "活泼", "奢华", "自然")
    SCENES = ("电商主图", "产品手册", "卖场海报", "活动海报", "社交媒体")
    CONSTRAINTS = ("不要logo", "无logo", "纯色底", "白底", "产品位置不要变")

    def interpret(
        self, request: str, creativity: CreativityLevel = CreativityLevel.MEDIUM
    ) -> MarketingBrief:
        text = request.strip()
        subject = self._subject(text)
        scene = next((value for value in self.SCENES if value in text), None)
        color = next((value for value in self.COLORS if value in text), None)
        style = next((value for value in self.STYLES if value in text), None)
        constraints = [value for value in self.CONSTRAINTS if value in text]
        ambiguities: list[str] = []
        questions: list[str] = []
        if subject is None:
            ambiguities.append("marketing_subject")
            questions.append("需要推广的具体产品或主体是什么？")
        if scene is None and creativity == CreativityLevel.LOW:
            ambiguities.append("campaign_scene")
            questions.append("这张素材用于电商主图、手册还是活动海报？")
        if style is None and creativity == CreativityLevel.LOW:
            ambiguities.append("visual_style")
            questions.append("希望采用什么视觉风格和品牌调性？")
        if creativity == CreativityLevel.HIGH:
            scene = scene or "活动海报"
            style = style or "高端"
            color = color or "象牙白"
        return MarketingBrief(
            raw_request=text,
            subject=subject,
            scene=scene,
            primary_color=color,
            style=style,
            hard_constraints=constraints,
            ambiguities=ambiguities,
            clarification_questions=questions,
            creativity=creativity,
        )

    @staticmethod
    def _subject(text: str) -> str | None:
        candidates = ("精华", "口红", "香水", "手机", "咖啡", "运动鞋", "产品")
        return next((value for value in candidates if value in text), None)
