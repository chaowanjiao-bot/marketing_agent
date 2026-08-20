from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_api_uses_project_python_and_production_by_default() -> None:
    script = (PROJECT_ROOT / "scripts/run_api.sh").read_text(encoding="utf-8")
    assert "runtime/venv/gpu/bin/python" in script
    assert 'AGENT_TOOLSET="${AGENT_TOOLSET:-production}"' in script
    assert 'exec "$PYTHON_BIN" -m uvicorn' in script


def test_environment_template_covers_model_workers() -> None:
    template = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    required = {
        "MARKETING_AGENT_ROOT",
        "TASK_ROOT",
        "QWEN_IMAGE_MODEL_PATH",
        "POWERPAINT_MODEL_PATH",
        "GROUNDING_DINO_MODEL_PATH",
        "SAM2_CHECKPOINT_PATH",
        "POWERPAINT_PYTHON",
        "VQASCORE_PYTHON",
        "OCR_PYTHON",
    }
    configured = {
        line.split("=", 1)[0]
        for line in template.splitlines()
        if line and not line.startswith("#") and "=" in line
    }
    assert required <= configured
