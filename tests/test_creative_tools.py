from marketing_agent.creative_tools import BrandRuleValidatorTool, SafeAreaValidatorTool
from marketing_agent.schemas import ObservationStatus
from marketing_agent.tools import build_default_registry


def test_safe_area_validator_reports_named_component():
    result = SafeAreaValidatorTool().execute({
        "safe_margin": .05,
        "components": [{
            "component_id": "logo", "kind": "logo",
            "x": .96, "y": .1, "width": .03, "height": .03,
        }],
    })
    assert result.status == ObservationStatus.PARTIAL
    assert result.issues == ["logo:outside_safe_area"]


def test_brand_rules_are_deterministic():
    result = BrandRuleValidatorTool().execute({
        "recognized_texts": ["焕亮新生", "全网最低"],
        "required_phrases": ["焕亮新生", "LUMIERE"],
        "forbidden_phrases": ["全网最低"],
    })
    assert "missing_required:LUMIERE" in result.issues
    assert "forbidden_phrase:全网最低" in result.issues


def test_restricted_registry_enforces_agent_capabilities():
    view = build_default_registry().restricted({"safe_area_validator"})
    assert view.get("safe_area_validator").name == "safe_area_validator"
    try:
        view.get("generate_image")
        assert False, "expected capability violation"
    except PermissionError:
        pass
