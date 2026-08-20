import json
from pathlib import Path

import pytest

from marketing_agent.adapters.subprocess_adapters import JsonModelWorker, ModelWorkerError


def _worker(tmp_path: Path, body: str) -> JsonModelWorker:
    script = tmp_path / "worker.py"
    script.write_text(body, encoding="utf-8")
    return JsonModelWorker(Path("/usr/bin/python3"), tmp_path, timeout=2), script


def test_json_worker_round_trip(tmp_path: Path) -> None:
    worker, script = _worker(
        tmp_path,
        "import json,sys; p=json.load(sys.stdin); print(json.dumps({'ok': True, 'result': p}))",
    )
    worker.interpreter = script
    script.chmod(0o755)
    script.write_text("#!/usr/bin/python3\n" + script.read_text(), encoding="utf-8")
    result = worker.call("test", {"value": 7})
    assert result["action"] == "test"


def test_json_worker_rejects_missing_environment(tmp_path: Path) -> None:
    worker = JsonModelWorker(tmp_path / "missing-python", tmp_path)
    with pytest.raises(FileNotFoundError):
        worker.call("test", {})
