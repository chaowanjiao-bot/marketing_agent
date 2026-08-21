from __future__ import annotations

import re

from pydantic import BaseModel, Field

from .brand import BrandProfile
from .brief import MarketingBrief


class MarketingCopy(BaseModel):
    headline: str = Field(min_length=1, max_length=40)
    subheadline: str = Field(default="", max_length=80)
    call_to_action: str = Field(default="", max_length=20)
    brand_signature: str = Field(default="", max_length=40)
    required_text: list[str] = Field(default_factory=list)
    source: str = "deterministic_fallback"

    def prompt_context(self) -> str:
        fields = [f"主标题（逐字准确，仅出现一次）：{self.headline}"]
        if self.subheadline:
            fields.append(f"副标题（逐字准确，仅出现一次）：{self.subheadline}")
        if self.call_to_action:
            fields.append(f"行动号召：{self.call_to_action}")
        if self.brand_signature:
            fields.append(f"品牌落款（仅出现一次）：{self.brand_signature}")
        return "\n".join(fields)


class MarketingCopyAgent:
    """Structured deterministic fallback with an interface suitable for a future LLM provider."""

    QUOTED = re.compile(r"[《\"“]([^》\"”]{2,40})[》\"”]")

    def create(self, brief: MarketingBrief, brand: BrandProfile | None = None) -> MarketingCopy:
        quoted = self.QUOTED.findall(brief.raw_request)
        subject = brief.subject or "产品"
        headline = quoted[0] if quoted else f"发现{subject}之美"
        subheadline = quoted[1] if len(quoted) > 1 else self._subheadline(brief)
        signature = brand.name if brand else ""
        required = list(dict.fromkeys([headline, subheadline, signature] + (
            brand.required_phrases if brand else []
        )))
        required = [value for value in required if value]
        copy = MarketingCopy(
            headline=headline,
            subheadline=subheadline,
            call_to_action="即刻探索",
            brand_signature=signature,
            required_text=required,
        )
        if brand:
            violations = brand.violations(" ".join(required))
            if violations:
                raise ValueError("generated copy contains forbidden phrases: " + ", ".join(violations))
        return copy

    @staticmethod
    def _subheadline(brief: MarketingBrief) -> str:
        tone = brief.style or "精致"
        scene = brief.scene or "日常"
        return f"以{tone}质感，点亮{scene}灵感"

