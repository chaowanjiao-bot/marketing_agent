from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    project_root: Path
    qwen_model_path: Path
    output_dir: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    local_files_only: bool = True

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        root = Path(os.environ.get("MARKETING_AGENT_ROOT", Path.cwd())).resolve()
        return cls(
            project_root=root,
            qwen_model_path=Path(
                os.environ.get("QWEN_IMAGE_MODEL_PATH", root / "runtime/models/Qwen-Image")
            ).resolve(),
            output_dir=Path(
                os.environ.get("MODEL_OUTPUT_DIR", root / "runtime/outputs")
            ).resolve(),
            device=os.environ.get("MODEL_DEVICE", "cuda:0"),
        )

    def validate_qwen_image(self) -> None:
        required = self.qwen_model_path / "model_index.json"
        transformer_index = (
            self.qwen_model_path
            / "transformer/diffusion_pytorch_model.safetensors.index.json"
        )
        has_partial_files = any(self.qwen_model_path.rglob("*.incomplete"))
        if not required.is_file() or not transformer_index.is_file() or has_partial_files:
            raise FileNotFoundError(
                "Qwen-Image download is incomplete; check "
                "runtime/logs/qwen_image_download.log"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
