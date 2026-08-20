from pathlib import Path

from marketing_agent.readiness import production_readiness, production_ready


def test_readiness_requires_all_key_files(tmp_path: Path) -> None:
    state = production_readiness(tmp_path)
    assert not production_ready(tmp_path)
    assert state == {
        "qwen_image": False,
        "sam2": False,
        "grounding_dino": False,
        "powerpaint": False,
        "vqascore_code": False,
        "powerpaint_env": False,
        "vqascore_env": False,
    }


def test_incomplete_marker_keeps_component_unready(tmp_path: Path) -> None:
    qwen = tmp_path / "runtime/models/Qwen-Image"
    (qwen / "transformer").mkdir(parents=True)
    (qwen / "model_index.json").touch()
    (qwen / "transformer/diffusion_pytorch_model.safetensors.index.json").touch()
    (qwen / "weight.incomplete").touch()
    assert not production_readiness(tmp_path)["qwen_image"]


def test_powerpaint_accepts_official_v2_weight_layout(tmp_path: Path) -> None:
    powerpaint = tmp_path / "runtime/models/PowerPaint-v2"
    required = [
        "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors",
        "PowerPaint_Brushnet/pytorch_model.bin",
        "realisticVisionV60B1_v51VAE/model_index.json",
        "realisticVisionV60B1_v51VAE/unet/diffusion_pytorch_model-002.safetensors",
    ]
    for relative_path in required:
        path = powerpaint / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    assert production_readiness(tmp_path)["powerpaint"]
