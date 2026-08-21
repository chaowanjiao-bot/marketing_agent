from pathlib import Path
import subprocess


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


def test_production_process_deployment_contract() -> None:
    deploy = PROJECT_ROOT / "deploy"
    for script in (deploy / "manage.sh", deploy / "render_systemd.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    template = (deploy / "production.env.example").read_text(encoding="utf-8")
    required = {
        "TASK_EXECUTION_MODE=external", "AUTH_ENABLED=true",
        "TASK_DATABASE_PATH=", "TASK_QUEUE_PATH=", "AUTH_DATABASE_PATH=",
        "APP_PYTHON=", "CUDA_VISIBLE_DEVICES=",
    }
    assert all(value in template for value in required)
    assert "MODEL_WORKER_MODE=oneshot" in template
    assert "password" not in template.casefold()
    assert "private_key" not in template.casefold()


def test_systemd_and_nginx_templates_keep_installation_explicit() -> None:
    deploy = PROJECT_ROOT / "deploy"
    api = (deploy / "marketing-agent-api.service.in").read_text()
    worker = (deploy / "marketing-agent-worker.service.in").read_text()
    nginx = (deploy / "nginx.conf.example").read_text()
    assert "__PROJECT_DIR__" in api and "run-api" in api
    assert "__PROJECT_DIR__" in worker and "run-worker" in worker
    assert "proxy_pass http://127.0.0.1:8000" in nginx
    assert "client_max_body_size 21m" in nginx
