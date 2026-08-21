import time
from pathlib import Path

from fastapi.testclient import TestClient

from marketing_agent.api import create_app
from marketing_agent.dashboard import DashboardService
from marketing_agent.graph import run_task
from marketing_agent.schemas import TaskRequest
from marketing_agent.task_store import TaskStore


def test_empty_dashboard_has_zero_metrics_and_safe_html(tmp_path: Path) -> None:
    service = DashboardService(TaskStore(tmp_path / "tasks"))
    summary = service.summary()
    html = service.render_index()
    assert summary["total_tasks"] == 0
    assert summary["average_best_score"] is None
    assert "实验与决策看板" in html
    assert "暂无任务" in html


def test_dashboard_escapes_user_prompt(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    request = TaskRequest(prompt="生成产品海报 <script>alert(1)</script>", max_iterations=8)
    task_id = store.create(request)
    store.save_result(task_id, run_task(request))
    html = DashboardService(store).render_index()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_dashboard_api_lists_completed_task_and_detail(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        created = client.post("/tasks", json={
            "prompt": "生成高端香水活动海报", "candidate_count": 2,
            "output_formats": ["1:1", "4:5"], "max_iterations": 8,
        }).json()
        task_id = created["task_id"]
        for _ in range(200):
            status = client.get(f"/tasks/{task_id}").json()["status"]
            if status in {"completed", "failed"}:
                break
            time.sleep(0.01)
        summary = client.get("/dashboard/api/summary")
        index = client.get("/dashboard")
        detail = client.get(f"/dashboard/tasks/{task_id}")
    assert summary.status_code == 200
    assert summary.json()["total_tasks"] == 1
    assert summary.json()["status_counts"]["completed"] == 1
    assert index.status_code == 200 and task_id in index.text
    assert detail.status_code == 200
    assert "候选比较" in detail.text and "各画幅最佳结果" in detail.text


def test_asset_endpoint_rejects_unknown_and_outside_paths(tmp_path: Path) -> None:
    task_root = tmp_path / "tasks"
    store = TaskStore(task_root)
    request = TaskRequest(prompt="生成产品海报", max_iterations=8)
    task_id = store.create(request)
    result = run_task(request)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"image")
    result.assets[0].file_path = str(outside)
    store.save_result(task_id, result)
    with TestClient(create_app(task_root=task_root)) as client:
        response = client.get(f"/tasks/{task_id}/assets/{result.assets[0].asset_id}")
        traversal = client.get(f"/tasks/{task_id}/assets/not-an-asset")
    assert response.status_code == 404
    assert traversal.status_code == 404
