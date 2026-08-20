from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from marketing_agent.graph import run_task
from marketing_agent.real_tools import build_production_registry, build_qwen_registry
from marketing_agent.schemas import TaskRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument(
        "--production", action="store_true", help="use real generation, editing, and VQA tools"
    )
    parser.add_argument(
        "--quality-threshold", type=float, default=0.7, help="minimum VQA score required to finish"
    )
    args = parser.parse_args()
    registry = (
        build_production_registry(evaluation_threshold=args.quality_threshold)
        if args.production
        else build_qwen_registry()
    )
    result = run_task(
        TaskRequest(prompt=args.prompt, max_iterations=args.max_iterations),
        registry=registry,
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
