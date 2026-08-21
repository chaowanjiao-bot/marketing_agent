from __future__ import annotations

import gc
import os
import time
from typing import Any

from ..runtime_config import RuntimeConfig


class QwenImageGenerator:
    """Lazy local-only adapter for the official Qwen/Qwen-Image model."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._pipeline: Any | None = None

    def load(self) -> None:
        if self._pipeline is not None:
            return
        self.config.validate_qwen_image()
        import torch
        from diffusers import QwenImagePipeline

        self._pipeline = QwenImagePipeline.from_pretrained(
            str(self.config.qwen_model_path),
            torch_dtype=getattr(torch, self.config.dtype),
            local_files_only=self.config.local_files_only,
        )
        self._pipeline.to(self.config.device)

    def unload(self) -> None:
        """Release the large generation pipeline before evaluator models load."""
        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _unload_after_generate() -> bool:
        return os.environ.get("QWEN_UNLOAD_AFTER_GENERATE", "false").lower() in {
            "1", "true", "yes", "on",
        }

    def generate(
        self,
        *,
        prompt: str,
        seed: int,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = "low quality, blurry, distorted product, watermark",
        output_name: str = "qwen_image_output.png",
    ) -> dict[str, Any]:
        self.load()
        import torch

        started = time.monotonic()
        generator = torch.Generator(device=self.config.device).manual_seed(seed)
        try:
            image = self._pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                generator=generator,
            ).images[0]
            output_path = self.config.output_dir / output_name
            image.save(output_path)
            return {
                "file_path": str(output_path),
                "prompt": prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "latency_seconds": round(time.monotonic() - started, 3),
                "backend": "Qwen/Qwen-Image",
            }
        finally:
            if self._unload_after_generate():
                self.unload()
