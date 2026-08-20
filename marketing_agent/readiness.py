from __future__ import annotations

from pathlib import Path


def _complete(root: Path, required: list[str]) -> bool:
    return all((root / item).is_file() for item in required) and not any(
        root.rglob("*.incomplete")
    )


def production_readiness(project_root: Path) -> dict[str, bool]:
    models = project_root / "runtime/models"
    qwen = models / "Qwen-Image"
    grounding = models / "grounding-dino-base"
    powerpaint = models / "PowerPaint-v2"
    sam_checkpoint = models / "sam2/sam2.1_hiera_large.pt"
    return {
        "qwen_image": _complete(
            qwen,
            [
                "model_index.json",
                "transformer/diffusion_pytorch_model.safetensors.index.json",
            ],
        ),
        "sam2": sam_checkpoint.is_file()
        and sam_checkpoint.stat().st_size > 800 * 1024 * 1024,
        "grounding_dino": _complete(
            grounding, ["config.json", "model.safetensors"]
        ),
        "powerpaint": _complete(
            powerpaint,
            [
                "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors",
                "PowerPaint_Brushnet/pytorch_model.bin",
                "realisticVisionV60B1_v51VAE/model_index.json",
                "realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model-002.safetensors",
            ],
        ),
        "vqascore_code": (project_root / "runtime/repositories/t2v_metrics").is_dir(),
        "powerpaint_env": (
            project_root / "runtime/venv/powerpaint/bin/python"
        ).is_file(),
        "vqascore_env": (project_root / "runtime/venv/vqascore/bin/python").is_file(),
    }

def production_ready(project_root: Path) -> bool:
    return all(production_readiness(project_root).values())
