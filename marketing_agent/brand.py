from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, field_validator


class BrandAssetType(str, Enum):
    LOGO = "logo"
    PRODUCT = "product"
    FONT = "font"
    STYLE_REFERENCE = "style_reference"


class BrandAssetRef(BaseModel):
    asset_id: str = Field(pattern=r"^brand_[a-zA-Z0-9_-]{3,64}$")
    asset_type: BrandAssetType
    uri: str = Field(min_length=1)
    usage: str = ""


class BrandProfile(BaseModel):
    brand_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{2,64}$")
    name: str = Field(min_length=1, max_length=80)
    tone: list[str] = Field(default_factory=list, max_length=8)
    primary_colors: list[str] = Field(default_factory=list, max_length=8)
    secondary_colors: list[str] = Field(default_factory=list, max_length=8)
    fonts: list[str] = Field(default_factory=list, max_length=6)
    required_phrases: list[str] = Field(default_factory=list, max_length=10)
    forbidden_phrases: list[str] = Field(default_factory=list, max_length=30)
    visual_rules: list[str] = Field(default_factory=list, max_length=20)
    assets: list[BrandAssetRef] = Field(default_factory=list, max_length=30)

    @field_validator(
        "tone", "primary_colors", "secondary_colors", "fonts", "required_phrases",
        "forbidden_phrases", "visual_rules",
    )
    @classmethod
    def clean_values(cls, values: list[str]) -> list[str]:
        cleaned = [" ".join(value.split()) for value in values if value.strip()]
        return list(dict.fromkeys(cleaned))

    def violations(self, text: str) -> list[str]:
        lower = text.casefold()
        return [phrase for phrase in self.forbidden_phrases if phrase.casefold() in lower]

    def prompt_context(self) -> str:
        sections: list[str] = [f"品牌：{self.name}"]
        if self.tone:
            sections.append("品牌调性：" + "、".join(self.tone))
        if self.primary_colors:
            sections.append("主色：" + "、".join(self.primary_colors))
        if self.secondary_colors:
            sections.append("辅色：" + "、".join(self.secondary_colors))
        if self.fonts:
            sections.append("字体偏好：" + "、".join(self.fonts))
        if self.required_phrases:
            sections.append("必须准确出现：" + "；".join(self.required_phrases))
        if self.forbidden_phrases:
            sections.append("禁止出现：" + "；".join(self.forbidden_phrases))
        if self.visual_rules:
            sections.append("视觉规范：" + "；".join(self.visual_rules))
        if self.assets:
            references = [
                f"{asset.asset_type.value}:{asset.asset_id}({asset.usage or '按品牌规范使用'})"
                for asset in self.assets
            ]
            sections.append("品牌素材引用：" + "；".join(references))
        return "\n".join(sections)


class BrandProfileProvider(Protocol):
    def get(self, brand_id: str) -> BrandProfile | None: ...


class BrandCatalog:
    """In-memory capability boundary; persistence may be added without changing callers."""

    def __init__(self, profiles: list[BrandProfile] | None = None) -> None:
        self._profiles: dict[str, BrandProfile] = {}
        for profile in profiles or []:
            self.register(profile)

    def register(self, profile: BrandProfile) -> None:
        if profile.brand_id in self._profiles:
            raise ValueError(f"duplicate brand profile: {profile.brand_id}")
        self._profiles[profile.brand_id] = profile

    def get(self, brand_id: str) -> BrandProfile | None:
        return self._profiles.get(brand_id)

