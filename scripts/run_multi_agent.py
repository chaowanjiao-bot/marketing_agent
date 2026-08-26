from __future__ import annotations

import argparse
import json

from marketing_agent.contracts import MultiAgentCampaign
from marketing_agent.real_tools import build_production_registry
from marketing_agent.model_adapters import structured_model_from_env
from marketing_agent.schemas import OutputFormat, TaskRequest
from marketing_agent.tools import build_default_registry
from marketing_agent.workflows import run_multi_agent_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Supervisor-Specialist marketing campaign")
    parser.add_argument("prompt")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--quality-threshold", type=float, default=0.75)
    parser.add_argument("--max-generations", type=int, default=6)
    parser.add_argument("--review-required", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["1:1"], choices=["1:1", "4:5", "9:16", "16:9"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = TaskRequest(
        prompt=args.prompt,
        output_formats=[OutputFormat(value) for value in args.formats],
        review_required=args.review_required,
    )
    campaign = MultiAgentCampaign(request=request)
    campaign = campaign.model_copy(update={
        "budget": campaign.budget.model_copy(update={"max_generations": args.max_generations})
    })
    registry = build_production_registry(args.quality_threshold) if args.production else build_default_registry()
    model = structured_model_from_env()
    try:
        result = run_multi_agent_campaign(
            campaign, registry, model=model,
            quality_threshold=args.quality_threshold
        )
    finally:
        if model is not None:
            model.close()
        registry.close()
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
