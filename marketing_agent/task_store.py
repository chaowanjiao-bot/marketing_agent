from __future__ import annotations

import json
import shutil
from pathlib import Path
from uuid import uuid4

from .brief import MarketingInputInterpreter
from .schemas import FinalResult, TaskRequest


class TaskStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, request: TaskRequest) -> str:
        task_id = f"task_{uuid4().hex[:12]}"
        task_dir = self.root / task_id
        for name in ("inputs", "masks", "generations", "evaluations", "final"):
            (task_dir / name).mkdir(parents=True, exist_ok=True)
        self._write_json(task_dir / "request.json", request.model_dump(mode="json"))
        brief = MarketingInputInterpreter().interpret(request.prompt, request.creativity)
        self._write_json(task_dir / "brief.json", brief.model_dump(mode="json"))
        self._write_json(task_dir / "status.json", {"status": "created"})
        return task_id

    def set_status(self, task_id: str, status: str, **details: object) -> None:
        payload: dict[str, object] = {"status": status}
        payload.update(details)
        self._write_json(self.path(task_id) / "status.json", payload)

    def materialize_outputs(self, task_id: str, result: FinalResult) -> FinalResult:
        task_dir = self.path(task_id)
        assets = []
        latest_output: Path | None = None
        for index, asset in enumerate(result.assets, start=1):
            source = Path(asset.file_path)
            if source.is_file():
                target = task_dir / "generations" / (
                    f"{index:02d}_{asset.asset_id}{source.suffix or '.png'}"
                )
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                asset = asset.model_copy(update={"file_path": str(target)})
                latest_output = target
            assets.append(asset)
        if latest_output is not None:
            shutil.copy2(
                latest_output, task_dir / "final" / f"final{latest_output.suffix}"
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
        self._write_json(
            task_dir / "status.json",
            {"status": result.status, "terminal_reason": result.terminal_reason},
        )
        trace_path = task_dir / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as handle:
            for event in result.trace:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def status(self, task_id: str) -> dict[str, object]:
        return self._read_json(self.path(task_id) / "status.json")

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
