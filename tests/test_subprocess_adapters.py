import json
from pathlib import Path

import pytest

from marketing_agent.adapters.subprocess_adapters import (
    JsonModelWorker, PersistentJsonModelWorker,
)


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


def test_persistent_worker_reuses_same_process_for_multiple_calls(tmp_path: Path) -> None:
    script = tmp_path / "persistent_worker.py"
    script.write_text(
        "#!/usr/bin/python3\n"
        "import json,os,sys\n"
        "prefix='__MARKETING_AGENT_JSON__'\n"
        "count=0\n"
        "for line in sys.stdin:\n"
        " count+=1; payload=json.loads(line); print(prefix+json.dumps({"
        "'ok':True,'result':{'count':count,'pid':os.getpid(),'payload':payload}}),flush=True)\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    worker = PersistentJsonModelWorker(script, tmp_path, timeout=2)
    first = worker.call("one", {"value": 1})
    second = worker.call("two", {"value": 2})
    assert first["pid"] == second["pid"]
    assert [first["count"], second["count"]] == [1, 2]
    assert second["payload"]["action"] == "two"
    worker.close()
    assert worker.process is None
