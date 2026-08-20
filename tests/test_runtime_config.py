from pathlib import Path

import pytest

from marketing_agent.runtime_config import RuntimeConfig


def make_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        project_root=tmp_path,
        qwen_model_path=tmp_path / "runtime/models/Qwen-Image",
        output_dir=tmp_path / "runtime/outputs",
    )


def test_qwen_validation_rejects_partial_download(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.qwen_model_path.mkdir(parents=True)
    (config.qwen_model_path / "model_index.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="download is incomplete"):
        config.validate_qwen_image()


def test_qwen_validation_accepts_required_layout(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.qwen_model_path.mkdir(parents=True)
    (config.qwen_model_path / "model_index.json").write_text("{}")
    (config.qwen_model_path / "transformer").mkdir()
    (
        config.qwen_model_path / "transformer/diffusion_pytorch_model.safetensors.index.json"
    ).write_text("{}")
    config.validate_qwen_image()
    assert config.output_dir.is_dir()
