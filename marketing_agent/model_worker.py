from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


RESULT_PREFIX = "__MARKETING_AGENT_JSON__"


def execute(
    payload: dict[str, object], runtimes: dict[str, Any] | None = None
) -> dict[str, object]:
    root = Path.cwd()
    runtimes = runtimes if runtimes is not None else {}
    action = str(payload.get("action", ""))
    arguments = dict(payload.get("arguments", {}))
    if action == "powerpaint_edit":
        from .adapters.powerpaint import PowerPaintEditor

        if action not in runtimes:
            runtimes[action] = PowerPaintEditor(root)
        runtime = runtimes[action]
        result = runtime.edit(**arguments)
    elif action == "vqascore_evaluate":
        from .adapters.vqa_score import VQAScoreEvaluator

        if action not in runtimes:
            runtimes[action] = VQAScoreEvaluator(root)
        runtime = runtimes[action]
        result = runtime.evaluate(**arguments).__dict__
    elif action == "ocr_text_evaluate":
        from .adapters.ocr_text import QwenVLOCREvaluator

        if action not in runtimes:
            runtimes[action] = QwenVLOCREvaluator(root)
        runtime = runtimes[action]
        result = runtime.evaluate(**arguments)
    else:
        raise ValueError(f"unsupported worker action: {action}")
    return {"ok": True, "result": result}


def serve() -> int:
    runtimes: dict[str, Any] = {}
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            response = execute(payload, runtimes)
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(RESULT_PREFIX + json.dumps(response, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    if "--serve" in sys.argv:
        return serve()
    try:
        payload = json.load(sys.stdin)
        print(RESULT_PREFIX + json.dumps(execute(payload), ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
