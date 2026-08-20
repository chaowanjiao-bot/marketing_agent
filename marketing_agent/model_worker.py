from __future__ import annotations

import json
import sys
from pathlib import Path


RESULT_PREFIX = "__MARKETING_AGENT_JSON__"


def execute(payload: dict[str, object]) -> dict[str, object]:
    root = Path.cwd()
    action = str(payload.get("action", ""))
    arguments = dict(payload.get("arguments", {}))
    if action == "powerpaint_edit":
        from .adapters.powerpaint import PowerPaintEditor

        result = PowerPaintEditor(root).edit(**arguments)
    elif action == "vqascore_evaluate":
        from .adapters.vqa_score import VQAScoreEvaluator

        result = VQAScoreEvaluator(root).evaluate(**arguments).__dict__
    elif action == "ocr_text_evaluate":
        from .adapters.ocr_text import QwenVLOCREvaluator

        result = QwenVLOCREvaluator(root).evaluate(**arguments)
    else:
        raise ValueError(f"unsupported worker action: {action}")
    return {"ok": True, "result": result}


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        print(RESULT_PREFIX + json.dumps(execute(payload), ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
