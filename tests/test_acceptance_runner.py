import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_acceptance.py"
SPEC = importlib.util.spec_from_file_location("run_acceptance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summarize_result_counts_repairs_and_requires_compliance() -> None:
    result = {
        "status": "completed", "terminal_reason": "quality_gate_passed",
        "best_compliant_asset_id": "asset_ok", "best_score": 0.82,
        "assets": [{}, {}],
        "observations": [
            {"tool_name": "generate_image", "issues": []},
            {"tool_name": "evaluate_image", "issues": ["unexpected_text"]},
            {"tool_name": "generate_image", "issues": []},
        ],
    }
    row = MODULE.summarize_result({"id": "text"}, "task_1", 12.345, result)
    assert row["passed"] is True
    assert row["repair_count"] == 1
    assert row["issues_seen"] == ["unexpected_text"]


def test_aggregate_reports_pass_rate_and_average() -> None:
    rows = [
        {"passed": True, "elapsed_seconds": 10, "repair_count": 1},
        {"passed": False, "elapsed_seconds": 20, "repair_count": 0},
    ]
    summary = MODULE.aggregate(rows)
    assert summary == {
        "case_count": 2, "passed": 1, "failed": 1, "pass_rate": 0.5,
        "total_elapsed_seconds": 30.0, "average_elapsed_seconds": 15.0,
        "total_repairs": 1,
    }
