from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..vision_contracts import EvaluationResult


RESULT_PREFIX = "__MARKETING_AGENT_JSON__"


class ModelWorkerError(RuntimeError):
    pass


class JsonModelWorker:
    def __init__(self, interpreter: Path, project_root: Path, timeout: int = 900) -> None:
        self.interpreter = interpreter
        self.project_root = project_root
        self.timeout = timeout

    def call(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.interpreter.is_file():
            raise FileNotFoundError(f"isolated model environment missing: {self.interpreter}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "mvp")
        completed = subprocess.run(
            [str(self.interpreter), "-m", "marketing_agent.model_worker"],
            input=json.dumps({"action": action, "arguments": arguments}),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.project_root,
            env=env,
        )
        if completed.returncode != 0:
            raise ModelWorkerError(completed.stderr[-2000:] or "model worker failed")
        payload_text = ""
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(RESULT_PREFIX):
                payload_text = line[len(RESULT_PREFIX) :]
                break
        if not payload_text:
            payload_text = completed.stdout.strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            stdout_tail = completed.stdout[-1000:].replace("\n", "\\n")
            stderr_tail = completed.stderr[-1000:].replace("\n", "\\n")
            raise ModelWorkerError(
                "model worker returned invalid JSON; "
                f"stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}"
            ) from exc
        if not payload.get("ok"):
            raise ModelWorkerError(str(payload.get("error", "model worker failed")))
        return dict(payload["result"])


class SubprocessPowerPaintEditor:
    def __init__(self, project_root: Path) -> None:
        interpreter = Path(
            os.environ.get(
                "POWERPAINT_PYTHON",
                project_root / "runtime/venv/powerpaint/bin/python",
            )
        )
        self.worker = JsonModelWorker(interpreter, project_root)

    def edit(self, **arguments: Any) -> dict[str, Any]:
        return self.worker.call("powerpaint_edit", arguments)


class SubprocessVQAScoreEvaluator:
    def __init__(self, project_root: Path) -> None:
        interpreter = Path(
            os.environ.get(
                "VQASCORE_PYTHON", project_root / "runtime/venv/vqascore/bin/python"
            )
        )
        self.worker = JsonModelWorker(interpreter, project_root)

    def evaluate(self, **arguments: Any) -> EvaluationResult:
        result = self.worker.call("vqascore_evaluate", arguments)
        return EvaluationResult(**result)


class SubprocessOCRTextEvaluator:
    def __init__(self, project_root: Path) -> None:
        interpreter = Path(
            os.environ.get(
                "OCR_PYTHON", project_root / "runtime/venv/vqascore/bin/python"
            )
        )
        self.worker = JsonModelWorker(interpreter, project_root)

    def evaluate(self, **arguments: Any) -> dict[str, Any]:
        return self.worker.call("ocr_text_evaluate", arguments)
