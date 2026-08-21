#!/usr/bin/env python3
"""Run reproducible production acceptance cases through the public HTTP API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener


TERMINAL_STATUSES = {"completed", "aborted", "failed", "cancelled", "waiting_for_review"}


def summarize_result(case: dict, task_id: str, elapsed: float, result: dict) -> dict:
    observations = result.get("observations", [])
    generation_count = sum(item.get("tool_name") == "generate_image" for item in observations)
    issues = sorted({issue for item in observations for issue in item.get("issues", [])})
    compliant = bool(result.get("best_compliant_asset_id"))
    passed = (
        result.get("status") == "completed"
        and result.get("terminal_reason") == "quality_gate_passed"
        and compliant
    )
    return {
        "case_id": case["id"],
        "task_id": task_id,
        "passed": passed,
        "status": result.get("status"),
        "terminal_reason": result.get("terminal_reason"),
        "elapsed_seconds": round(elapsed, 2),
        "generation_count": generation_count,
        "repair_count": max(0, generation_count - 1),
        "best_score": result.get("best_score"),
        "compliant": compliant,
        "issues_seen": issues,
        "asset_count": len(result.get("assets", [])),
    }


def aggregate(rows: list[dict]) -> dict:
    passed = sum(bool(row["passed"]) for row in rows)
    elapsed = sum(float(row["elapsed_seconds"]) for row in rows)
    return {
        "case_count": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "pass_rate": round(passed / len(rows), 4) if rows else 0.0,
        "total_elapsed_seconds": round(elapsed, 2),
        "average_elapsed_seconds": round(elapsed / len(rows), 2) if rows else 0.0,
        "total_repairs": sum(int(row["repair_count"]) for row in rows),
    }


class ApiClient:
    def __init__(self, base_url: str, timeout: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        request = Request(
            self.base_url + path, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def authenticate(self, email: str, password: str) -> str:
        try:
            response = self.request("POST", "/auth/login", {"email": email, "password": password})
        except HTTPError as exc:
            if exc.code not in {401, 404}:
                raise
            response = self.request("POST", "/auth/register", {
                "email": email, "password": password, "display_name": "自动验收",
            })
        project_id = response.get("default_project_id")
        if not project_id:
            projects = self.request("GET", "/projects").get("projects", [])
            if not projects:
                raise RuntimeError("authenticated account has no project")
            project_id = projects[0]["project_id"]
        return str(project_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8023")
    parser.add_argument("--cases", type=Path, default=Path("benchmarks/acceptance_cases.json"))
    parser.add_argument("--report", type=Path, default=Path("runtime/acceptance_report.json"))
    parser.add_argument("--tag", help="run only cases carrying this tag")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--task-timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    email = os.environ.get("ACCEPTANCE_EMAIL")
    password = os.environ.get("ACCEPTANCE_PASSWORD")
    if not email or not password:
        print("ACCEPTANCE_EMAIL and ACCEPTANCE_PASSWORD are required", file=sys.stderr)
        return 2
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    if args.tag:
        cases = [case for case in cases if args.tag in case.get("tags", [])]
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected]
    if args.max_cases is not None:
        cases = cases[:args.max_cases]
    if not cases:
        print("no acceptance cases selected", file=sys.stderr)
        return 2

    client = ApiClient(args.base_url, timeout=30)
    try:
        health = client.request("GET", "/health")
        if not health.get("production_ready"):
            raise RuntimeError("production tools are not ready")
        project_id = client.authenticate(email, password)
        rows = []
        for index, case in enumerate(cases, 1):
            payload = {
                "prompt": case["prompt"], "project_id": project_id,
                "candidate_count": 1, "output_formats": ["1:1"],
                "generate_copy": False, "seed": case["seed"],
                "max_iterations": case.get("max_iterations", 4),
            }
            started = time.monotonic()
            created = client.request("POST", "/tasks", payload)
            task_id = created["task_id"]
            print(f"[{index}/{len(cases)}] {case['id']} -> {task_id}", flush=True)
            while True:
                status = client.request("GET", f"/tasks/{task_id}")
                if status["status"] in TERMINAL_STATUSES:
                    break
                if time.monotonic() - started > args.task_timeout:
                    raise TimeoutError(f"{task_id} exceeded {args.task_timeout}s")
                time.sleep(args.poll_seconds)
            result = client.request("GET", f"/tasks/{task_id}/result")
            row = summarize_result(case, task_id, time.monotonic() - started, result)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    except (HTTPError, URLError, OSError, RuntimeError, TimeoutError) as exc:
        print(f"acceptance run failed: {exc}", file=sys.stderr)
        return 1

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "summary": aggregate(rows),
        "cases": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
