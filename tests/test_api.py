import time
from pathlib import Path

from fastapi.testclient import TestClient

from marketing_agent.api import create_app
from marketing_agent.case_memory import CaseMemory


def test_health_lists_registered_tools(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["tools"] == ["edit_image", "evaluate_image", "generate_image"]


def test_create_and_read_task_result(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        created = client.post(
            "/tasks",
            json={"prompt": "生成一张高端香水活动海报", "max_iterations": 8},
        )
        assert created.status_code == 202
        assert created.json()["status"] == "queued"
        task_id = created.json()["task_id"]

        for _ in range(100):
            status = client.get(f"/tasks/{task_id}").json()["status"]
            if status in {"completed", "failed"}:
                break
            time.sleep(0.01)

        assert status == "completed"
        result = client.get(f"/tasks/{task_id}/result")
        assert result.status_code == 200
        assert result.json()["terminal_reason"] == "quality_gate_passed"


def test_upload_png(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        response = client.post(
            "/assets",
            files={"file": ("reference.png", b"\x89PNG\r\n\x1a\nmock", "image/png")},
        )
    assert response.status_code == 201
    payload = response.json()
    assert payload["content_type"] == "image/png"
    assert payload["size"] == 12
    assert Path(payload["path"]).is_file()


def test_upload_rejects_spoofed_image(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        response = client.post(
            "/assets",
            files={"file": ("fake.png", b"not an image", "image/png")},
        )
    assert response.status_code == 400


def test_uploaded_asset_drives_async_image_edit(tmp_path: Path) -> None:
    task_root = tmp_path / "tasks"
    with TestClient(create_app(task_root=task_root)) as client:
        uploaded = client.post(
            "/assets",
            files={"file": ("reference.png", b"\x89PNG\r\n\x1a\nmock", "image/png")},
        ).json()
        created = client.post(
            "/tasks",
            json={
                "prompt": "把左侧瓶子改成红色节日包装",
                "input_asset_id": uploaded["asset_id"],
                "target_expression": "left bottle",
                "max_iterations": 8,
            },
        )
        assert created.status_code == 202
        task_id = created.json()["task_id"]
        for _ in range(100):
            status = client.get(f"/tasks/{task_id}").json()["status"]
            if status in {"completed", "failed"}:
                break
            time.sleep(0.01)
        result = client.get(f"/tasks/{task_id}/result").json()

    assert status == "completed"
    assert result["goal"]["task_type"] == "image_edit"
    assert result["assets"][0]["tool_name"] == "edit_image"
    request_data = (task_root / task_id / "request.json").read_text(encoding="utf-8")
    assert uploaded["path"] in request_data


def test_task_rejects_unknown_asset_and_raw_server_path(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        missing = client.post(
            "/tasks",
            json={"prompt": "编辑产品图", "input_asset_id": "upload_000000000000"},
        )
        raw_path = client.post(
            "/tasks", json={"prompt": "编辑产品图", "input_image": "/etc/passwd"}
        )
    assert missing.status_code == 404
    assert raw_path.status_code == 400


def test_missing_task_is_404(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        assert client.get("/tasks/task_missing").status_code == 404


def test_seed_and_search_case_memory(tmp_path: Path) -> None:
    memory = CaseMemory(tmp_path / "explicit_test_memory.sqlite3")
    with TestClient(create_app(task_root=tmp_path / "tasks", memory=memory)) as client:
        created = client.post("/memory/cases", json={
            "prompt": "高端护肤精华广告", "enhanced_prompt": "香槟金棚拍",
            "score": 0.92, "compliant": True,
        })
        searched = client.get("/memory/search", params={"query": "护肤精华海报"})
    assert created.status_code == 201
    assert searched.status_code == 200
    assert searched.json()["cases"][0]["case_id"] == created.json()["case_id"]


def test_rag_is_disabled_without_explicit_configuration(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        health = client.get("/health")
        search = client.get("/memory/search", params={"query": "护肤海报"})
    assert health.json()["memory_enabled"] is False
    assert search.status_code == 503
    assert not (tmp_path / "memory").exists()


def test_api_persists_multi_candidate_selection(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        created = client.post("/tasks", json={
            "prompt": "生成三版高端香水活动海报",
            "candidate_count": 3,
            "max_iterations": 8,
        })
        assert created.status_code == 202
        task_id = created.json()["task_id"]
        for _ in range(200):
            status = client.get(f"/tasks/{task_id}").json()["status"]
            if status in {"completed", "failed"}:
                break
            time.sleep(0.01)
        result = client.get(f"/tasks/{task_id}/result").json()
    assert status == "completed"
    assert len(result["candidate_summaries"]) == 3
    assert sum(item["selected"] for item in result["candidate_summaries"]) == 1
    assert result["selected_candidate_index"] in {0, 1, 2}
    assert len(result["assets"]) == 6
