from __future__ import annotations

import json
import os
import subprocess
import select
import time
from pathlib import Path
from threading import Lock
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

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        paths = []
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        paths.extend([str(self.project_root), str(self.project_root / "mvp")])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    @staticmethod
    def _parse_payload(payload_text: str) -> dict[str, Any]:
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise ModelWorkerError("model worker returned invalid JSON") from exc
        if not payload.get("ok"):
            raise ModelWorkerError(str(payload.get("error", "model worker failed")))
        return dict(payload["result"])

    def call(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.interpreter.is_file():
            raise FileNotFoundError(f"isolated model environment missing: {self.interpreter}")
        completed = subprocess.run(
            [str(self.interpreter), "-m", "marketing_agent.model_worker"],
            input=json.dumps({"action": action, "arguments": arguments}),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.project_root,
            env=self._environment(),
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
        return self._parse_payload(payload_text)


class PersistentJsonModelWorker(JsonModelWorker):
    """One request at a time over JSONL; loaded model objects remain in the child process."""

    def __init__(self, interpreter: Path, project_root: Path, timeout: int = 900) -> None:
        super().__init__(interpreter, project_root, timeout)
        self.process: subprocess.Popen[str] | None = None
        self.lock = Lock()
        self.restart_count = 0

    def _start(self) -> subprocess.Popen[str]:
        if not self.interpreter.is_file():
            raise FileNotFoundError(f"isolated model environment missing: {self.interpreter}")
        self.process = subprocess.Popen(
            [str(self.interpreter), "-m", "marketing_agent.model_worker", "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=self.project_root, env=self._environment(),
        )
        return self.process

    def _stop(self) -> None:
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def call(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            for retry in range(2):
                process = self.process
                if process is None or process.poll() is not None:
                    process = self._start()
                    if retry:
                        self.restart_count += 1
                assert process.stdin is not None and process.stdout is not None
                try:
                    process.stdin.write(json.dumps({"action": action, "arguments": arguments}) + "\n")
                    process.stdin.flush()
                    deadline = time.monotonic() + self.timeout
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"model worker timed out after {self.timeout}s")
                        ready, _, _ = select.select([process.stdout], [], [], remaining)
                        if not ready:
                            raise TimeoutError(f"model worker timed out after {self.timeout}s")
                        line = process.stdout.readline()
                        if not line:
                            raise BrokenPipeError("model worker exited")
                        if line.startswith(RESULT_PREFIX):
                            return self._parse_payload(line[len(RESULT_PREFIX):].strip())
                except (BrokenPipeError, OSError):
                    self._stop()
                    if retry:
                        raise ModelWorkerError("persistent model worker repeatedly exited")
                except TimeoutError:
                    self._stop()
                    raise
        raise ModelWorkerError("persistent model worker failed")

    def close(self) -> None:
        with self.lock:
            self._stop()


def build_model_worker(interpreter: Path, project_root: Path) -> JsonModelWorker:
    timeout = int(os.environ.get(
        "MODEL_WORKER_TIMEOUT", os.environ.get("MODEL_WORKER_TIMEOUT_SECONDS", "900")
    ))
    if os.environ.get("MODEL_WORKER_MODE", "persistent").lower() == "persistent":
        return PersistentJsonModelWorker(interpreter, project_root, timeout)
    return JsonModelWorker(interpreter, project_root, timeout)


class SubprocessPowerPaintEditor:
    def __init__(self, project_root: Path, worker: JsonModelWorker | None = None) -> None:
        interpreter = Path(
            os.environ.get(
                "POWERPAINT_PYTHON",
                project_root / "runtime/venv/powerpaint/bin/python",
            )
        )
        self.worker = worker or build_model_worker(interpreter, project_root)

    def edit(self, **arguments: Any) -> dict[str, Any]:
        return self.worker.call("powerpaint_edit", arguments)

    def close(self) -> None:
        close = getattr(self.worker, "close", None)
        if callable(close):
            close()


class SubprocessVQAScoreEvaluator:
    def __init__(self, project_root: Path, worker: JsonModelWorker | None = None) -> None:
        interpreter = Path(
            os.environ.get(
                "VQASCORE_PYTHON", project_root / "runtime/venv/vqascore/bin/python"
            )
        )
        self.worker = worker or build_model_worker(interpreter, project_root)

    def evaluate(self, **arguments: Any) -> EvaluationResult:
        result = self.worker.call("vqascore_evaluate", arguments)
        return EvaluationResult(**result)

    def close(self) -> None:
        close = getattr(self.worker, "close", None)
        if callable(close):
            close()


class SubprocessOCRTextEvaluator:
    def __init__(self, project_root: Path, worker: JsonModelWorker | None = None) -> None:
        interpreter = Path(
            os.environ.get(
                "OCR_PYTHON", project_root / "runtime/venv/vqascore/bin/python"
            )
        )
        self.worker = worker or build_model_worker(interpreter, project_root)

    def evaluate(self, **arguments: Any) -> dict[str, Any]:
        return self.worker.call("ocr_text_evaluate", arguments)

    def close(self) -> None:
        close = getattr(self.worker, "close", None)
        if callable(close):
            close()
