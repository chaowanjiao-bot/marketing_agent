from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .schemas import AssetVersion, FinalResult, ProvenanceRecord


TRAINED_ALGORITHMIC_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
)
COMPOSITED_AI_MEDIA = (
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositedWithTrainedAlgorithmicMedia"
)


class ProvenanceManifestBuilder:
    def build(
        self, *, task_id: str, asset: AssetVersion, score: float | None,
        brand_id: str | None,
    ) -> dict[str, Any]:
        generated = asset.tool_name == "generate_image"
        actions: list[dict[str, Any]] = []
        if not generated:
            actions.append({
                "action": "c2pa.opened",
                "softwareAgent": "Marketing Creative Agent",
            })
        actions.append({
            "action": "c2pa.created" if generated else "c2pa.edited",
            "digitalSourceType": TRAINED_ALGORITHMIC_MEDIA if generated else COMPOSITED_AI_MEDIA,
            "softwareAgent": "Marketing Creative Agent/0.4.0",
            "description": (
                "AI-generated marketing image" if generated
                else "AI-assisted localized image edit"
            ),
        })
        mime = mimetypes.guess_type(asset.file_path)[0] or "image/png"
        return {
            "claim_generator": "Marketing Creative Agent/0.4.0",
            "claim_generator_info": [{
                "name": "Marketing Creative Agent", "version": "0.4.0",
            }],
            "title": Path(asset.file_path).name,
            "format": mime,
            "assertions": [
                {"label": "c2pa.actions", "data": {"actions": actions}},
                {"label": "org.marketingagent.generation", "data": {
                    "task_id": task_id,
                    "asset_id": asset.asset_id,
                    "model": "Qwen-Image" if generated else "PowerPaint-v2",
                    "seed": asset.seed,
                    "output_format": asset.output_format.value,
                    "width": asset.width,
                    "height": asset.height,
                    "best_score": score,
                    "brand_id": brand_id,
                    "prompt_sha256": hashlib.sha256(
                        asset.prompt.encode("utf-8")
                    ).hexdigest(),
                }},
            ],
        }


class C2paSigner:
    def __init__(
        self, tool_path: Path, certificate_path: Path, private_key_path: Path,
        *, timeout: int = 120,
    ) -> None:
        self.tool_path = tool_path
        self.certificate_path = certificate_path
        self.private_key_path = private_key_path
        self.timeout = timeout

    def validate(self) -> None:
        if not self.tool_path.is_file() or not os.access(self.tool_path, os.X_OK):
            raise FileNotFoundError("c2patool executable is missing")
        if not self.certificate_path.is_file():
            raise FileNotFoundError("C2PA signing certificate is missing")
        if not self.private_key_path.is_file():
            raise FileNotFoundError("C2PA private key is missing")

    def sign(
        self, source: Path, output: Path, manifest: dict[str, Any]
    ) -> dict[str, Any]:
        self.validate()
        output.parent.mkdir(parents=True, exist_ok=True)
        signing_manifest = dict(manifest)
        signing_manifest.update({
            "alg": os.environ.get("C2PA_SIGNING_ALGORITHM", "es256"),
            "sign_cert": str(self.certificate_path.resolve()),
            "private_key": str(self.private_key_path.resolve()),
        })
        with tempfile.TemporaryDirectory(prefix="marketing-c2pa-") as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(
                json.dumps(signing_manifest, ensure_ascii=False), encoding="utf-8"
            )
            completed = subprocess.run(
                [str(self.tool_path), str(source), "--manifest", str(manifest_path),
                 "--output", str(output), "--force"],
                capture_output=True, text=True, timeout=self.timeout,
            )
        if completed.returncode != 0 or not output.is_file():
            raise RuntimeError("C2PA signing failed: " + completed.stderr[-1000:])
        verified = subprocess.run(
            [str(self.tool_path), str(output), "--info"],
            capture_output=True, text=True, timeout=self.timeout,
        )
        if verified.returncode != 0:
            output.unlink(missing_ok=True)
            raise RuntimeError("C2PA verification failed: " + verified.stderr[-1000:])
        return {"verified": True, "verification_output": verified.stdout[-2000:]}


class ProvenanceService:
    def __init__(
        self, task_root: Path, *, signer: C2paSigner | None = None,
        manifest_only: bool = False,
    ) -> None:
        self.task_root = task_root
        self.signer = signer
        self.manifest_only = manifest_only
        self.builder = ProvenanceManifestBuilder()

    def process(self, task_id: str, result: FinalResult) -> FinalResult:
        selected_ids = {
            item.best_asset_id for item in result.format_summaries if item.best_asset_id
        } or ({result.best_asset_id} if result.best_asset_id else set())
        score_by_asset = {
            item.best_asset_id: item.best_score for item in result.format_summaries
            if item.best_asset_id
        }
        records: list[ProvenanceRecord] = []
        updated_assets: list[AssetVersion] = []
        provenance_dir = self.task_root / task_id / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        for asset in result.assets:
            if asset.asset_id not in selected_ids:
                updated_assets.append(asset)
                continue
            source = Path(asset.file_path)
            if not source.is_file():
                raise FileNotFoundError(f"selected asset is missing: {source}")
            manifest = self.builder.build(
                task_id=task_id, asset=asset,
                score=score_by_asset.get(asset.asset_id, result.best_score),
                brand_id=result.brand_id,
            )
            manifest_path = provenance_dir / f"{asset.asset_id}.manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            signed_path: Path | None = None
            status = "manifest_only"
            if self.signer is not None and not self.manifest_only:
                signed_path = source.with_name(
                    f"{source.stem}_c2pa{source.suffix}"
                )
                self.signer.sign(source, signed_path, manifest)
                asset = asset.model_copy(update={"file_path": str(signed_path)})
                status = "signed_and_verified"
            records.append(ProvenanceRecord(
                asset_id=asset.asset_id, status=status,
                manifest_path=str(manifest_path),
                signed_asset_path=str(signed_path) if signed_path else None,
                prompt_sha256=manifest["assertions"][1]["data"]["prompt_sha256"],
            ))
            updated_assets.append(asset)
        return result.model_copy(update={
            "assets": updated_assets, "provenance": records,
            "content_credentials_status": (
                "signed_and_verified" if records and all(
                    item.status == "signed_and_verified" for item in records
                ) else "manifest_only" if records else "not_enabled"
            ),
        })


def build_provenance_service(task_root: Path) -> ProvenanceService | None:
    enabled = os.environ.get("C2PA_ENABLED", "false").lower() in {"1", "true", "yes"}
    manifest_only = os.environ.get("C2PA_MANIFEST_ONLY", "false").lower() in {
        "1", "true", "yes"
    }
    if not enabled and not manifest_only:
        return None
    if manifest_only:
        return ProvenanceService(task_root, manifest_only=True)
    signer = C2paSigner(
        Path(os.environ.get("C2PATOOL_PATH", "c2patool")),
        Path(os.environ.get("C2PA_SIGN_CERT_PATH", "")),
        Path(os.environ.get("C2PA_PRIVATE_KEY_PATH", "")),
        timeout=int(os.environ.get("C2PA_TIMEOUT_SECONDS", "120")),
    )
    signer.validate()
    return ProvenanceService(task_root, signer=signer)
