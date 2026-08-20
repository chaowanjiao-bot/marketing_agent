from marketing_agent.brief import (
    CreativityLevel,
    MarketingInputInterpreter,
)


def test_low_creativity_asks_instead_of_inventing() -> None:
    brief = MarketingInputInterpreter().interpret(
        "生成一张口红营销素材，不要logo", CreativityLevel.LOW
    )
    assert brief.subject == "口红"
    assert brief.hard_constraints == ["不要logo"]
    assert "campaign_scene" in brief.ambiguities
    assert len(brief.clarification_questions) == 2


def test_high_creativity_fills_safe_marketing_defaults() -> None:
    brief = MarketingInputInterpreter().interpret(
        "生成一张香水营销素材", CreativityLevel.HIGH
    )
    assert brief.scene == "活动海报"
    assert brief.style == "高端"
    assert brief.primary_color == "象牙白"
    assert not brief.ambiguities
