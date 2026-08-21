from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .asset_store import AssetStore
from .case_memory import CaseMemory
from .executor import TaskExecutor
from .experience import ExperienceMemory, JsonlExperienceStore
from .provenance import ProvenanceService, build_provenance_service
from .dashboard import DashboardService
from .readiness import production_readiness
from .schemas import ReviewDecision, TaskRequest
from .task_store import TaskStore
from .task_queue import DurableTaskQueue
from .tools import ToolRegistry, build_default_registry


class SeedCaseRequest(BaseModel):
    prompt: str = Field(min_length=3)
    enhanced_prompt: str = ""
    asset_path: str | None = None
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    compliant: bool = False
    metadata: dict[str, object] = Field(default_factory=dict)


class HumanReviewRequest(BaseModel):
    decision: ReviewDecision
    feedback: str = Field(default="", max_length=2000)
    reviewer: str = Field(default="human", min_length=1, max_length=100)


def create_app(
    *, registry: ToolRegistry | None = None, task_root: Path | None = None,
    memory: CaseMemory | None = None,
    experience: ExperienceMemory | None = None,
    provenance: ProvenanceService | None = None,
) -> FastAPI:
    toolset = "custom" if registry is not None else os.environ.get("AGENT_TOOLSET", "mock")
    if registry is not None:
        tools = registry
    elif toolset == "mock":
        tools = build_default_registry()
    elif toolset == "production":
        from .real_tools import build_production_registry

        tools = build_production_registry()
    else:
        raise ValueError("AGENT_TOOLSET must be mock or production")
    root = task_root or Path(os.environ.get("TASK_ROOT", "runtime/tasks"))
    store = TaskStore(root, metadata_path=Path(os.environ.get(
        "TASK_DATABASE_PATH", root.parent / "task_metadata.sqlite3"
    )))
    assets = AssetStore(root.parent / "uploads")
    # RAG is opt-in. Merely starting the API must not create a database.
    if memory is None and os.environ.get("RAG_ENABLED", "false").lower() in {"1", "true", "yes"}:
        memory = CaseMemory(Path(os.environ.get(
            "RAG_DATABASE_PATH", root.parent / "memory" / "cases.sqlite3"
        )))
    if experience is None and os.environ.get(
        "EXPERIENCE_MEMORY_ENABLED", "false"
    ).lower() in {"1", "true", "yes"}:
        experience = ExperienceMemory(JsonlExperienceStore(Path(os.environ.get(
            "EXPERIENCE_MEMORY_PATH", root.parent / "memory" / "experience.jsonl"
        ))))
    if provenance is None:
        provenance = build_provenance_service(root)
    execution_mode = os.environ.get("TASK_EXECUTION_MODE", "inline").lower()
    if execution_mode not in {"inline", "external"}:
        raise ValueError("TASK_EXECUTION_MODE must be inline or external")
    queue = DurableTaskQueue(Path(os.environ.get(
        "TASK_QUEUE_PATH", root.parent / "task_queue.sqlite3"
    ))) if execution_mode == "external" else None
    executor = TaskExecutor(
        store, tools, memory=memory, experience=experience, provenance=provenance,
        queue=queue,
    )
    dashboard = DashboardService(store)
    app = FastAPI(title="Marketing Creative Agent", version="0.2.0")

    @app.on_event("shutdown")
    def shutdown_executor() -> None:
        executor.shutdown()
        tools.close()

    @app.get("/health")
    def health() -> dict[str, object]:
        project_root = Path(os.environ.get("MARKETING_AGENT_ROOT", Path.cwd()))
        components = production_readiness(project_root)
        return {
            "status": "ok",
            "tools": list(tools.names),
            "qwen_ready": components["qwen_image"],
            "production_ready": all(components.values()),
            "components": components,
            "toolset": toolset,
            "memory_enabled": memory is not None,
            "experience_memory_enabled": experience is not None,
            "experience_count": experience.count() if experience else 0,
            "c2pa_enabled": provenance is not None,
            "c2pa_mode": (
                "manifest_only" if provenance and provenance.manifest_only
                else "signed" if provenance else "disabled"
            ),
            "runtime": tools.runtime_status(),
            "task_database": str(store.metadata.path),
            "execution_mode": execution_mode,
            "queue": queue.stats() if queue else None,
        }

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard_index() -> str:
        return dashboard.render_index()

    @app.get("/dashboard/api/summary")
    def dashboard_summary() -> dict[str, object]:
        return dashboard.summary()

    @app.get("/dashboard/tasks/{task_id}", response_class=HTMLResponse)
    def dashboard_task(task_id: str) -> str:
        try:
            return dashboard.render_task(task_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.post("/assets", status_code=201)
    async def upload_asset(file: UploadFile = File(...)) -> dict[str, str | int]:
        data = await file.read(assets.max_bytes + 1)
        try:
            return assets.save(content_type=file.content_type or "", data=data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/memory/cases", status_code=201)
    def add_seed_case(case: SeedCaseRequest) -> dict[str, str]:
        if memory is None:
            raise HTTPException(status_code=503, detail="RAG memory is not configured")
        case_id = memory.add(source="seed", **case.model_dump())
        return {"case_id": case_id, "status": "active"}

    @app.get("/memory/search")
    def search_memory(query: str, limit: int = 3) -> dict[str, object]:
        if memory is None:
            raise HTTPException(status_code=503, detail="RAG memory is not configured")
        try:
            cases = memory.search(query, limit=limit)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"query": query, "count": len(cases), "cases": cases}

    @app.get("/experience/strategies")
    def experience_strategies() -> dict[str, object]:
        if experience is None:
            raise HTTPException(status_code=503, detail="experience memory is not configured")
        return {"count": experience.count(), "strategies": experience.strategies()}

    @app.post("/tasks", status_code=202)
    def create_task(request: TaskRequest) -> dict[str, object]:
        if request.input_image:
            raise HTTPException(
                status_code=400, detail="use input_asset_id instead of input_image"
            )
        if request.input_asset_id:
            try:
                input_path = assets.resolve(request.input_asset_id)
            except (KeyError, ValueError) as exc:
                raise HTTPException(status_code=404, detail="input asset not found") from exc
            request = request.model_copy(update={"input_image": str(input_path)})
        task_id = store.create(request)
        executor.submit(task_id, request)
        return {"task_id": task_id, "status": "queued"}

    @app.get("/tasks")
    def list_tasks(status: str | None = None, limit: int = 50) -> dict[str, object]:
        if limit < 1 or limit > 200:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
        tasks = store.list_tasks(status=status, limit=limit)
        return {"count": len(tasks), "tasks": tasks}

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, object]:
        try:
            return {"task_id": task_id, **store.status(task_id)}
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc

    @app.get("/tasks/{task_id}/events")
    def get_task_events(task_id: str) -> dict[str, object]:
        try:
            events = store.events(task_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        return {"task_id": task_id, "count": len(events), "events": events}
    @app.delete("/tasks/{task_id}", status_code=202)
    def cancel_task(task_id: str) -> dict[str, str]:
        try:
            cancelled = executor.cancel(task_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        if not cancelled:
            raise HTTPException(status_code=409, detail="task is already running or finished")
        return {"task_id": task_id, "status": "cancelled"}


    @app.get("/tasks/{task_id}/result")
    def get_result(task_id: str) -> dict[str, object]:
        try:
            result = store.result(task_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        if result is None:
            raise HTTPException(status_code=409, detail="task is not finished")
        return result

    @app.get("/tasks/{task_id}/assets/{asset_id}")
    def get_task_asset(task_id: str, asset_id: str) -> FileResponse:
        try:
            path = store.asset_path(task_id, asset_id)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="asset not found") from exc
        return FileResponse(path)

    @app.post("/tasks/{task_id}/review", status_code=202)
    def review_task(task_id: str, review: HumanReviewRequest) -> dict[str, object]:
        try:
            return executor.review(
                task_id, review.decision, feedback=review.feedback,
                reviewer=review.reviewer,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return app


app = create_app()
