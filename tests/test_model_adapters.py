import pytest

from marketing_agent.model_adapters import structured_model_from_env


def test_structured_model_configuration_is_all_or_nothing(monkeypatch):
    monkeypatch.setenv("STRUCTURED_MODEL_BASE_URL", "http://localhost:8001/v1")
    monkeypatch.delenv("STRUCTURED_MODEL_NAME", raising=False)
    monkeypatch.delenv("STRUCTURED_MODEL_API_KEY", raising=False)
    with pytest.raises(ValueError):
        structured_model_from_env()


def test_structured_model_is_optional(monkeypatch):
    for name in ("STRUCTURED_MODEL_BASE_URL", "STRUCTURED_MODEL_NAME", "STRUCTURED_MODEL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert structured_model_from_env() is None
