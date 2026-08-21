import pytest

from marketing_agent.brand import BrandAssetRef, BrandAssetType, BrandCatalog, BrandProfile
from marketing_agent.brief import MarketingInputInterpreter
from marketing_agent.copy_agent import MarketingCopyAgent


def profile() -> BrandProfile:
    return BrandProfile(
        brand_id="lumiere", name="LUMIÈRE", tone=["高端", "克制"],
        primary_colors=["象牙白", "香槟金"], required_phrases=["焕亮新生"],
        forbidden_phrases=["全网最低"], visual_rules=["品牌名只能出现一次"],
        assets=[BrandAssetRef(asset_id="brand_lumiere_logo", asset_type=BrandAssetType.LOGO,
                              uri="asset://logo", usage="底部安全区")],
    )


def test_brand_profile_compiles_generation_context() -> None:
    context = profile().prompt_context()
    assert "品牌：LUMIÈRE" in context
    assert "主色：象牙白、香槟金" in context
    assert "logo:brand_lumiere_logo" in context
    assert "禁止出现：全网最低" in context


def test_catalog_rejects_duplicate_profile() -> None:
    catalog = BrandCatalog([profile()])
    assert catalog.get("lumiere") is not None
    with pytest.raises(ValueError, match="duplicate"):
        catalog.register(profile())


def test_copy_agent_returns_stable_structured_copy() -> None:
    brief = MarketingInputInterpreter().interpret("生成一张高端精华活动海报")
    copy = MarketingCopyAgent().create(brief, profile())
    assert copy.headline == "发现精华之美"
    assert copy.brand_signature == "LUMIÈRE"
    assert "焕亮新生" in copy.required_text
    assert "主标题（逐字准确，仅出现一次）" in copy.prompt_context()


def test_explicit_quoted_copy_is_preserved() -> None:
    brief = MarketingInputInterpreter().interpret(
        "为香水生成海报，标题《此刻生香》，副标题《留住每一次心动》"
    )
    copy = MarketingCopyAgent().create(brief)
    assert copy.headline == "此刻生香"
    assert copy.subheadline == "留住每一次心动"


def test_forbidden_phrase_is_detected() -> None:
    brand = profile().model_copy(update={"required_phrases": ["全网最低"]})
    brief = MarketingInputInterpreter().interpret("生成一张精华海报")
    with pytest.raises(ValueError, match="forbidden"):
        MarketingCopyAgent().create(brief, brand)
