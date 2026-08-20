from pathlib import Path

import pytest

from marketing_agent.graph import run_task
from marketing_agent.schemas import TaskRequest
from marketing_agent.task_store import TaskStore


def test_task_store_persists_reproducible_artifacts(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    request = TaskRequest(prompt="Create a premium perfume poster", max_iterations=8)
    task_id = store.create(request)
    store.save_result(task_id, run_task(request))
    task_dir = store.path(task_id)
    assert (task_dir / "request.json").is_file()
    assert (task_dir / "brief.json").is_file()
    assert (task_dir / "trace.jsonl").read_text().count("\n") > 0
    assert store.status(task_id)["status"] == "completed"


def test_task_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = TaskStore(tmp_path)
    with pytest.raises(ValueError):
        store.path("../secrets")


def test_materialize_outputs_copies_assets_into_task_directory(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / "tasks")
    request = TaskRequest(prompt="Create a product poster", max_iterations=8)
    task_id = store.create(request)
    result = run_task(request)
    source = tmp_path / "generated.png"
    mask = tmp_path / "mask.png"
    source.write_bytes(b"image")
    mask.write_bytes(b"mask")
    result.assets[0].file_path = str(source)
    result.observations[0].outputs["mask_path"] = str(mask)

    materialized = store.materialize_outputs(task_id, result)

    assert Path(materialized.assets[0].file_path).parent.name == "generations"
    assert Path(materialized.observations[0].outputs["mask_path"]).parent.name == "masks"
    assert (store.path(task_id) / "final/final.png").is_file()
