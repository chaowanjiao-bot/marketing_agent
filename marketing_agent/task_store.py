from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from .brief import MarketingInputInterpreter
from .schemas import FinalResult, TaskRequest
from .task_repository import SqliteTaskRepository


class TaskStore:
    def __init__(self, root: Path, metadata_path: Path | None = None) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = SqliteTaskRepository(
            metadata_path or self.root.parent / "task_metadata.sqlite3"
        )

    def create(self, request: TaskRequest, *, owner_id: str = "anonymous") -> str:
        task_id = f"task_{uuid4().hex[:12]}"
        task_dir = self.root / task_id
        for name in (
            "inputs", "masks", "generations", "evaluations", "final", "provenance"
        ):
            (task_dir / name).mkdir(parents=True, exist_ok=True)
        request_payload = request.model_dump(mode="json")
        self._write_json(task_dir / "request.json", request_payload)
        brief = MarketingInputInterpreter().interpret(request.prompt, request.creativity)
        self._write_json(task_dir / "brief.json", brief.model_dump(mode="json"))
        self._write_json(task_dir / "status.json", {"status": "created"})
        self.metadata.create(
            task_id, request_payload, owner_id=owner_id, project_id=request.project_id
        )
        return task_id

    def set_status(self, task_id: str, status: str, **details: object) -> None:
        payload: dict[str, object] = {"status": status}
        payload.update(details)
        self._write_json(self.path(task_id) / "status.json", payload)
        self.metadata.update_status(
            task_id, status, str(details.get("phase")) if details.get("phase") else None,
            dict(details),
        )

    def materialize_outputs(self, task_id: str, result: FinalResult) -> FinalResult:
        task_dir = self.path(task_id)
        assets = []
        materialized_by_id: dict[str, Path] = {}
        for index, asset in enumerate(result.assets, start=1):
            source = Path(asset.file_path)
            if source.is_file():
                target = task_dir / "generations" / (
                    f"{index:02d}_{asset.asset_id}{source.suffix or '.png'}"
                )
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                asset = asset.model_copy(update={"file_path": str(target)})
                materialized_by_id[asset.asset_id] = target
            assets.append(asset)
        selected_output = materialized_by_id.get(result.best_asset_id or "")
        if selected_output is not None:
            shutil.copy2(
                selected_output, task_dir / "final" / f"final{selected_output.suffix}"
            )
        for observation in result.observations:
            mask_path = observation.outputs.get("mask_path")
            if mask_path and Path(str(mask_path)).is_file():
                source = Path(str(mask_path))
                target = task_dir / "masks" / f"{observation.tool_name}{source.suffix}"
                shutil.copy2(source, target)
                observation.outputs["mask_path"] = str(target)
        return result.model_copy(update={"assets": assets})

    def save_result(self, task_id: str, result: FinalResult) -> None:
        task_dir = self.path(task_id)
        self._write_json(task_dir / "result.json", result.model_dump(mode="json"))
        self.set_status(
            task_id, result.status, phase="finished",
            terminal_reason=result.terminal_reason,
        )
        trace_path = task_dir / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for event in result.trace:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def status(self, task_id: str) -> dict[str, object]:
        return self._read_json(self.path(task_id) / "status.json")

    def request(self, task_id: str) -> TaskRequest:
        return TaskRequest.model_validate(
            self._read_json(self.path(task_id) / "request.json")
        )

    def update_request(self, task_id: str, request: TaskRequest) -> None:
        payload = request.model_dump(mode="json")
        self._write_json(self.path(task_id) / "request.json", payload)
        self.metadata.update_request(task_id, payload)

    def result_model(self, task_id: str) -> FinalResult | None:
        payload = self.result(task_id)
        return FinalResult.model_validate(payload) if payload is not None else None

    def archive_result(self, task_id: str, review_round: int) -> Path:
        task_dir = self.path(task_id)
        source = task_dir / "result.json"
        if not source.is_file():
            raise KeyError("task result is missing")
        target = task_dir / f"result.review_{review_round}.json"
        if target.exists():
            raise ValueError("review round is already archived")
        shutil.copy2(source, target)
        return target

    def update_final_asset(self, task_id: str, result: FinalResult) -> None:
        selected = next(
            (asset for asset in result.assets if asset.asset_id == result.best_asset_id), None
        )
        if selected is None:
            return
        source = Path(selected.file_path)
        if source.is_file():
            target = self.path(task_id) / "final" / f"final{source.suffix}"
            shutil.copy2(source, target)

    def result(self, task_id: str) -> dict[str, object] | None:
        path = self.path(task_id) / "result.json"
        return self._read_json(path) if path.is_file() else None

    def path(self, task_id: str) -> Path:
        if not task_id.startswith("task_") or "/" in task_id or ".." in task_id:
            raise ValueError("invalid task id")
        path = self.root / task_id
        if not path.is_dir():
            raise KeyError(task_id)
        return path

    def task_ids(self) -> list[str]:
        tasks = [
            path for path in self.root.iterdir()
            if path.is_dir() and path.name.startswith("task_")
            and "/" not in path.name and ".." not in path.name
        ]
        return [path.name for path in sorted(
            tasks, key=lambda item: item.stat().st_mtime, reverse=True
        )]

    def list_tasks(
        self, *, status: str | None = None, limit: int = 50,
        owner_id: str | None = None, project_id: str | None = None,
    ) -> list[dict[str, object]]:
        return self.metadata.list(
            status=status, limit=limit, owner_id=owner_id, project_id=project_id
        )

    def authorize(self, task_id: str, owner_id: str) -> None:
        self.path(task_id)
        if self.metadata.owner(task_id) != owner_id:
            raise KeyError(task_id)

    def events(self, task_id: str) -> list[dict[str, object]]:
        self.path(task_id)
        return self.metadata.events(task_id)

    def asset_path(self, task_id: str, asset_id: str) -> Path:
        if not asset_id.startswith("asset_") or "/" in asset_id or ".." in asset_id:
            raise ValueError("invalid asset id")
        result = self.result(task_id)
        if result is None:
            raise KeyError(asset_id)
        asset = next(
            (item for item in result.get("assets", []) if item.get("asset_id") == asset_id),
            None,
        )
        if asset is None:
            raise KeyError(asset_id)
        path = Path(str(asset["file_path"])).resolve()
        task_path = self.path(task_id).resolve()
        if not path.is_file() or not path.is_relative_to(task_path):
            raise KeyError(asset_id)
        return path

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))
