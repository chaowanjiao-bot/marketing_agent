import time
from pathlib import Path

from fastapi.testclient import TestClient

from marketing_agent.api import create_app
from marketing_agent.case_memory import CaseMemory
from marketing_agent.experience import ExperienceMemory
from marketing_agent.provenance import ProvenanceService


def test_health_lists_registered_tools(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["tools"] == ["edit_image", "evaluate_image", "generate_image"]


def test_user_web_app_and_root_redirect(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        root = client.get("/", follow_redirects=False)
        page = client.get("/app")
        script = client.get("/app/static/app.js")

    assert root.status_code == 307 and root.headers["location"] == "/app"
    assert page.status_code == 200
    assert "创建营销任务" in page.text
    assert script.status_code == 200
    assert "TASK" not in script.text


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


def test_list_tasks_and_read_status_events(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        created = client.post("/tasks", json={
            "prompt": "Create a launch poster", "max_iterations": 8,
        }).json()
        task_id = created["task_id"]
        for _ in range(100):
            if client.get(f"/tasks/{task_id}").json()["status"] in {"completed", "failed"}:
                break
            time.sleep(0.01)
        listing = client.get("/tasks", params={"limit": 10})
        events = client.get(f"/tasks/{task_id}/events")

    assert listing.status_code == 200
    assert listing.json()["tasks"][0]["task_id"] == task_id
    assert events.status_code == 200
    assert events.json()["events"][0]["status"] == "created"
    assert events.json()["events"][-1]["status"] == "completed"


def test_external_execution_mode_only_enqueues(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TASK_EXECUTION_MODE", "external")
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        created = client.post("/tasks", json={"prompt": "Queued campaign"}).json()
        status = client.get(f"/tasks/{created['task_id']}").json()
        health = client.get("/health").json()

    assert status["status"] == "queued"
    assert health["execution_mode"] == "external"
    assert health["queue"]["queued"] == 1


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


def test_experience_memory_is_opt_in(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        disabled = client.get("/experience/strategies")
        health = client.get("/health")
    assert disabled.status_code == 503
    assert health.json()["experience_memory_enabled"] is False

    experience = ExperienceMemory()
    with TestClient(create_app(
        task_root=tmp_path / "enabled_tasks", experience=experience,
    )) as client:
        enabled = client.get("/experience/strategies")
    assert enabled.status_code == 200
    assert enabled.json() == {"count": 0, "strategies": {}}


def test_health_reports_c2pa_mode(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        disabled = client.get("/health").json()
    assert disabled["c2pa_enabled"] is False
    service = ProvenanceService(tmp_path / "signed_tasks", manifest_only=True)
    with TestClient(create_app(
        task_root=tmp_path / "signed_tasks", provenance=service,
    )) as client:
        enabled = client.get("/health").json()
    assert enabled["c2pa_enabled"] is True
    assert enabled["c2pa_mode"] == "manifest_only"


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


def test_api_returns_one_selected_asset_per_output_format(tmp_path: Path) -> None:
    formats = ["1:1", "4:5", "9:16", "16:9"]
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        created = client.post("/tasks", json={
            "prompt": "生成全渠道香水活动海报",
            "output_formats": formats,
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
    assert [item["output_format"] for item in result["format_summaries"]] == formats
    assert all(item["best_asset_id"] for item in result["format_summaries"])
    assert {(asset["width"], asset["height"]) for asset in result["assets"]} == {
        (1024, 1024), (1024, 1280), (768, 1360), (1360, 768),
    }


def test_api_human_review_revision_and_approval(tmp_path: Path) -> None:
    with TestClient(create_app(task_root=tmp_path / "tasks")) as client:
        created = client.post("/tasks", json={
            "prompt": "生成高端口红活动海报", "review_required": True,
            "max_iterations": 8,
        })
        task_id = created.json()["task_id"]
        for _ in range(200):
            status = client.get(f"/tasks/{task_id}").json()["status"]
            if status in {"waiting_for_review", "failed"}:
                break
            time.sleep(0.01)
        assert status == "waiting_for_review"
        revised = client.post(f"/tasks/{task_id}/review", json={
            "decision": "revise", "feedback": "口红主体放大，减少背景装饰",
            "reviewer": "creative_lead",
        })
        assert revised.status_code == 202
        for _ in range(200):
            status = client.get(f"/tasks/{task_id}").json()["status"]
            if status in {"waiting_for_review", "failed"}:
                break
            time.sleep(0.01)
        approved = client.post(f"/tasks/{task_id}/review", json={
            "decision": "approve", "reviewer": "creative_lead",
        })
        result = client.get(f"/tasks/{task_id}/result").json()
    assert approved.status_code == 202
    assert result["status"] == "completed"
    assert result["review_status"] == "approved"
    assert [item["decision"] for item in result["review_history"]] == ["revise", "approve"]
