from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from marketing_agent.contracts import MultiAgentCampaign
from marketing_agent.schemas import OutputFormat, TaskRequest
from marketing_agent.tools import build_default_registry
from marketing_agent.workflows import MultiAgentOptions, campaign_to_final_result, run_multi_agent_campaign
from marketing_agent.candidates import run_campaign_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Compare direct, single-agent and multi-agent systems")
    parser.add_argument("--cases", default="benchmarks/multi_agent_acceptance_cases.json")
    parser.add_argument("--system", choices=["direct", "single_agent", "multi_agent", "all"], default="all")
    parser.add_argument("--report", default="runtime/multi_agent_benchmark.json")
    parser.add_argument("--ablation", choices=["none","no_parallel_strategy","no_compliance","no_structured_repair"], default="none")
    return parser.parse_args()


def direct_generation(request, registry):
    generated = registry.get("generate_image").execute({
        "prompt": request.prompt, "seed": request.seed, "attempt": 1,
        "width": 1024, "height": 1024, "output_format": request.output_formats[0].value,
    })
    return registry.get("evaluate_image").execute({
        "image_path": generated.outputs["file_path"], "campaign_text": request.prompt,
        "attempt": 1, "expected_texts": [],
    })


def run_case(case, system, ablation="none"):
    registry = build_default_registry()
    request = TaskRequest(
        prompt=case["prompt"],
        orchestration_mode="multi_agent" if system == "multi_agent" else "single_agent",
        output_formats=[OutputFormat(x) for x in case["formats"]],
    )
    started = time.perf_counter()
    if system == "direct":
        observation = direct_generation(request, registry)
        row = {"status":"completed", "score":observation.metrics.get("marketing_alignment"), "formats":1, "generations":1}
    elif system == "single_agent":
        result = run_campaign_batch(request, registry)
        row = {"status":result.status, "score":result.best_score, "formats":len(result.format_summaries), "generations":len(result.assets)}
    else:
        options = MultiAgentOptions(
            parallel_strategies=ablation != "no_parallel_strategy",
            enable_compliance_agent=ablation != "no_compliance",
            enable_structured_layout_repair=ablation != "no_structured_repair",
        )
        campaign = run_multi_agent_campaign(MultiAgentCampaign(request=request), registry, options=options)
        result = campaign_to_final_result(campaign)
        row = {
            "status":result.status, "score":result.best_score,
            "formats":len(result.format_summaries), "generations":campaign.budget.generations,
            "director_rounds":campaign.budget.director_rounds,
            "conflicts":len(campaign.conflict_resolutions),
        }
    row.update(case_id=case["case_id"], system=system, ablation=ablation, elapsed_seconds=time.perf_counter()-started)
    registry.close()
    return row


def aggregate(rows):
    by_system = {}
    for system in sorted({x["system"] for x in rows}):
        items = [x for x in rows if x["system"] == system]
        by_system[system] = {
            "cases":len(items),
            "completion_rate":sum(x["status"] in {"completed","waiting_for_review"} for x in items)/len(items),
            "average_score":sum(float(x.get("score") or 0) for x in items)/len(items),
            "average_generations":sum(x["generations"] for x in items)/len(items),
            "average_elapsed_seconds":sum(x["elapsed_seconds"] for x in items)/len(items),
        }
    return by_system


def main():
    args=parse_args(); cases=json.loads(Path(args.cases).read_text(encoding="utf-8"))
    systems=["direct","single_agent","multi_agent"] if args.system=="all" else [args.system]
    rows=[run_case(case,system,args.ablation) for case in cases for system in systems]
    report={"rows":rows,"aggregate":aggregate(rows),"note":"Use production tools and human gold labels for publishable resume metrics."}
    target=Path(args.report); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report["aggregate"],ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
