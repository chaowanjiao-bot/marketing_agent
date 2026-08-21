import json
import time
from pathlib import Path

from marketing_agent.candidates import run_campaign_batch
from marketing_agent.executor import TaskExecutor
from marketing_agent.provenance import (
    COMPOSITED_AI_MEDIA, TRAINED_ALGORITHMIC_MEDIA, C2paSigner,
    ProvenanceManifestBuilder, ProvenanceService,
)
from marketing_agent.schemas import (
    AssetVersion, Observation, ObservationStatus, ReviewDecision, TaskRequest,
)
from marketing_agent.task_store import TaskStore
from marketing_agent.tools import AgentTool, MockEditTool, ToolRegistry, build_default_registry


def asset(tmp_path: Path, *, tool_name: str = "generate_image") -> AssetVersion:
    path = tmp_path / "result.png"
    path.write_bytes(b"image")
    return AssetVersion(
        tool_name=tool_name, file_path=str(path), prompt="秘密产品提示词",
        seed=42,
    )


def action_data(manifest: dict) -> list[dict]:
    return manifest["assertions"][0]["data"]["actions"]


def test_manifest_marks_generated_media_and_hashes_prompt(tmp_path: Path) -> None:
    item = asset(tmp_path)
    manifest = ProvenanceManifestBuilder().build(
        task_id="task_123", asset=item, score=0.91, brand_id="lumiere",
    )
    actions = action_data(manifest)
    assert actions[0]["action"] == "c2pa.created"
    assert actions[0]["digitalSourceType"] == TRAINED_ALGORITHMIC_MEDIA
    serialized = json.dumps(manifest, ensure_ascii=False)
    assert "秘密产品提示词" not in serialized
    assert len(manifest["assertions"][1]["data"]["prompt_sha256"]) == 64
    assert "private_key" not in serialized and "sign_cert" not in serialized


def test_manifest_marks_ai_assisted_edit(tmp_path: Path) -> None:
    manifest = ProvenanceManifestBuilder().build(
        task_id="task_123", asset=asset(tmp_path, tool_name="edit_image"),
        score=0.8, brand_id=None,
    )
    actions = action_data(manifest)
    assert [item["action"] for item in actions] == ["c2pa.opened", "c2pa.edited"]
    assert actions[1]["digitalSourceType"] == COMPOSITED_AI_MEDIA


def test_manifest_only_processes_best_asset_per_format(tmp_path: Path) -> None:
    task_root = tmp_path / "tasks"
    task_id = "task_123"
    (task_root / task_id / "provenance").mkdir(parents=True)
    result = run_campaign_batch(TaskRequest(
        prompt="生成全渠道香水海报", output_formats=["1:1", "4:5"],
        candidate_count=2, max_iterations=4,
    ), build_default_registry())
    for index, item in enumerate(result.assets):
        path = task_root / task_id / f"asset_{index}.png"
        path.write_bytes(b"image")
        item.file_path = str(path)
    processed = ProvenanceService(task_root, manifest_only=True).process(task_id, result)
    assert processed.content_credentials_status == "manifest_only"
    assert len(processed.provenance) == 2
    assert all(Path(item.manifest_path).is_file() for item in processed.provenance)
    assert all(item.signed_asset_path is None for item in processed.provenance)


def test_signer_uses_temporary_secret_manifest_and_verifies(tmp_path: Path) -> None:
    tool = tmp_path / "fake_c2patool.py"
    tool.write_text(
        "#!/usr/bin/python3\n"
        "import pathlib,shutil,sys\n"
        "args=sys.argv[1:]\n"
        "if '--info' in args: print('{\"validation_status\":\"valid\"}'); raise SystemExit(0)\n"
        "source=pathlib.Path(args[0]); output=pathlib.Path(args[args.index('--output')+1])\n"
        "shutil.copy2(source,output)\n",
        encoding="utf-8",
    )
    tool.chmod(0o755)
    certificate = tmp_path / "cert.pem"
    private_key = tmp_path / "private.pem"
    certificate.write_text("certificate", encoding="utf-8")
    private_key.write_text("private", encoding="utf-8")
    source = tmp_path / "source image.png"
    output = tmp_path / "signed image.png"
    source.write_bytes(b"image")
    signer = C2paSigner(tool, certificate, private_key)
    report = signer.sign(source, output, {"assertions": []})
    assert report["verified"] is True
    assert output.read_bytes() == b"image"
    assert not list(tmp_path.glob("**/manifest.json"))


class FileGenerator(AgentTool):
    name = "generate_image"

    def __init__(self, output: Path) -> None:
        self.output = output

    def execute(self, arguments):
        self.output.write_bytes(b"image")
        return Observation(
            tool_name=self.name, status=ObservationStatus.SUCCESS,
            outputs={
                "file_path": str(self.output), "prompt": arguments["prompt"],
                "seed": arguments["seed"], "width": arguments["width"],
                "height": arguments["height"], "output_format": arguments["output_format"],
            },
        )


class PassingEvaluator(AgentTool):
    name = "evaluate_image"

    def execute(self, arguments):
        return Observation(
            tool_name=self.name, status=ObservationStatus.SUCCESS,
            metrics={"marketing_alignment": 0.9, "text_accuracy": 1.0,
                     "text_uniqueness": 1.0},
        )


def test_human_review_signing_waits_until_approval(tmp_path: Path) -> None:
    task_root = tmp_path / "tasks"
    store = TaskStore(task_root)
    tools = ToolRegistry()
    tools.register(FileGenerator(tmp_path / "generated.png"))
    tools.register(PassingEvaluator())
    tools.register(MockEditTool())
    service = ProvenanceService(task_root, manifest_only=True)
    executor = TaskExecutor(store, tools, provenance=service)
    request = TaskRequest(
        prompt="生成高端香水海报", review_required=True, max_iterations=4,
    )
    task_id = store.create(request)
    executor.submit(task_id, request)
    for _ in range(200):
        if store.status(task_id)["status"] in {"waiting_for_review", "failed"}:
            break
        time.sleep(0.01)
    preview = store.result(task_id)
    assert preview["status"] == "waiting_for_review"
    assert preview["provenance"] == []
    executor.review(task_id, ReviewDecision.APPROVE)
    approved = store.result(task_id)
    assert approved["content_credentials_status"] == "manifest_only"
    assert len(approved["provenance"]) == 1
    executor.shutdown()
