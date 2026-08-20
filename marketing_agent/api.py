from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from .asset_store import AssetStore
from .executor import TaskExecutor
from .readiness import production_readiness
from .schemas import TaskRequest
from .task_store import TaskStore
from .tools import ToolRegistry, build_default_registry


def create_app(
    *, registry: ToolRegistry | None = None, task_root: Path | None = None
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
    store = TaskStore(root)
    assets = AssetStore(root.parent / "uploads")
    executor = TaskExecutor(store, tools)
    app = FastAPI(title="Marketing Creative Agent", version="0.2.0")

    @app.on_event("shutdown")
    def shutdown_executor() -> None:
        executor.shutdown()

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
        }

    @app.post("/assets", status_code=201)
    async def upload_asset(file: UploadFile = File(...)) -> dict[str, str | int]:
        data = await file.read(assets.max_bytes + 1)
        try:
            return assets.save(content_type=file.content_type or "", data=data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str) -> dict[str, object]:
        try:
            return {"task_id": task_id, **store.status(task_id)}
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail="task not found") from exc
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

    return app


app = create_app()
