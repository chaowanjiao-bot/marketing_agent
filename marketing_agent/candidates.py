from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .graph import run_task
from .formats import get_format_spec
from .schemas import CandidateSummary, FinalResult, FormatSummary, TaskRequest
from .tools import ToolRegistry


CANDIDATE_SEED_STRIDE = 1_000_003


def candidate_seed(base_seed: int, candidate_index: int) -> int:
    return base_seed + candidate_index * CANDIDATE_SEED_STRIDE


def _run_one(
    request: TaskRequest, registry: ToolRegistry, candidate_index: int
) -> FinalResult:
    candidate_request = request.model_copy(update={
        "candidate_count": 1,
        "parallel_candidates": False,
        "seed": candidate_seed(request.seed, candidate_index),
    })
    return run_task(candidate_request, registry=registry)


def _selection_key(result: FinalResult) -> tuple[int, float]:
    compliant = result.best_compliant_asset_id is not None
    return (int(compliant), float(result.best_score if result.best_score is not None else -1.0))


def run_candidate_batch(request: TaskRequest, registry: ToolRegistry) -> FinalResult:
    if request.candidate_count == 1:
        return run_task(request, registry=registry)

    indices = list(range(request.candidate_count))
    if request.parallel_candidates:
        with ThreadPoolExecutor(
            max_workers=request.candidate_count, thread_name_prefix="candidate"
        ) as pool:
            results = list(pool.map(lambda index: _run_one(request, registry, index), indices))
    else:
        results = [_run_one(request, registry, index) for index in indices]

    selected_index = max(indices, key=lambda index: _selection_key(results[index]))
    selected = results[selected_index]
    summaries = [
        CandidateSummary(
            candidate_index=index,
            seed=candidate_seed(request.seed, index),
            status=result.status,
            terminal_reason=result.terminal_reason,
            best_asset_id=result.best_asset_id,
            best_score=result.best_score,
            compliant=result.best_compliant_asset_id is not None,
            asset_count=len(result.assets),
            selected=index == selected_index,
            output_format=request.output_formats[0],
        )
        for index, result in enumerate(results)
    ]
    all_assets = [asset for result in results for asset in result.assets]
    all_observations = [observation for result in results for observation in result.observations]
    batch_trace = [{
        "event": "candidate_batch_selected",
        "candidate_count": request.candidate_count,
        "parallel": request.parallel_candidates,
        "selected_candidate_index": selected_index,
        "selection_policy": "text_compliance_then_marketing_score",
    }]
    for index, result in enumerate(results):
        batch_trace.append({
            "event": "candidate_completed",
            "candidate_index": index,
            "seed": candidate_seed(request.seed, index),
            "best_asset_id": result.best_asset_id,
            "best_score": result.best_score,
            "compliant": result.best_compliant_asset_id is not None,
        })
    return selected.model_copy(update={
        "assets": all_assets,
        "observations": all_observations,
        "trace": batch_trace + selected.trace,
        "candidate_summaries": summaries,
        "selected_candidate_index": selected_index,
    })


def run_campaign_batch(request: TaskRequest, registry: ToolRegistry) -> FinalResult:
    format_results: list[FinalResult] = []
    for output_format in request.output_formats:
        format_request = request.model_copy(update={"output_formats": [output_format]})
        format_results.append(run_candidate_batch(format_request, registry))
    if len(format_results) == 1:
        result = format_results[0]
        spec = get_format_spec(request.output_formats[0])
        return result.model_copy(update={"format_summaries": [FormatSummary(
            output_format=spec.name, width=spec.width, height=spec.height,
            best_asset_id=result.best_asset_id, best_score=result.best_score,
            compliant=result.best_compliant_asset_id is not None,
            selected_candidate_index=result.selected_candidate_index,
        )]})

    primary = format_results[0]
    summaries: list[FormatSummary] = []
    for output_format, result in zip(request.output_formats, format_results):
        spec = get_format_spec(output_format)
        summaries.append(FormatSummary(
            output_format=output_format, width=spec.width, height=spec.height,
            best_asset_id=result.best_asset_id, best_score=result.best_score,
            compliant=result.best_compliant_asset_id is not None,
            selected_candidate_index=result.selected_candidate_index,
        ))
    return primary.model_copy(update={
        "assets": [asset for result in format_results for asset in result.assets],
        "observations": [item for result in format_results for item in result.observations],
        "trace": [{"event": "format_batch_completed", "formats": [
            output_format.value for output_format in request.output_formats
        ]}] + [item for result in format_results for item in result.trace],
        "candidate_summaries": [
            item for result in format_results for item in result.candidate_summaries
        ],
        "format_summaries": summaries,
        "primary_output_format": request.output_formats[0],
    })
