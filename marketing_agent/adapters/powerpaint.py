from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any


class PowerPaintEditor:
    def __init__(self, project_root: Path, device: str = "cuda:0") -> None:
        self.root = project_root
        self.device = device
        self.repo = self.root / "runtime/repositories/PowerPaint"
        self.checkpoint = self.root / "runtime/models/PowerPaint-v2"
        self.output_dir = self.root / "runtime/outputs/edits"
        self._controller: Any | None = None

    def load(self) -> None:
        if self._controller is not None:
            return
        if not self.repo.is_dir() or not self.checkpoint.is_dir():
            raise FileNotFoundError("PowerPaint-v2 code or checkpoint is incomplete")
        import torch

        sys.path.insert(0, str(self.repo))
        from app import PowerPaintController

        self._controller = PowerPaintController(
            weight_dtype=torch.float16,
            checkpoint_dir=str(self.checkpoint),
            local_files_only=True,
            version="ppt-v2",
        )

    def edit(
        self, *, image_path: str, mask_path: str, prompt: str, seed: int
    ) -> dict[str, object]:
        self.load()
        from PIL import Image, ImageFilter

        started = time.monotonic()
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        outputs, _ = self._controller.predict(
            {"image": image, "mask": mask},
            prompt,
            1.0,
            30,
            7.5,
            seed,
            "low quality, blurry, distorted product, watermark",
            "text-guided",
            1.0,
            1.0,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generated = outputs[0].convert("RGB").resize(image.size, Image.Resampling.LANCZOS)
        resized_mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        feather_radius = round(min(image.size) * 3 / 1024)
        if feather_radius:
            resized_mask = resized_mask.filter(ImageFilter.GaussianBlur(feather_radius))
        composited = Image.composite(generated, image, resized_mask)
        output = self.output_dir / "powerpaint_edit.png"
        composited.save(output)
        return {
            "file_path": str(output),
            "prompt": prompt,
            "seed": seed,
            "mask_path": mask_path,
            "backend": "PowerPaint-v2",
            "latency_seconds": round(time.monotonic() - started, 3),
        }
