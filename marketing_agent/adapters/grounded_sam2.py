from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ..vision_contracts import SegmentationResult


class GroundedSam2Segmenter:
    def __init__(self, project_root: Path, device: str = "cuda:0") -> None:
        self.root = project_root
        self.device = device
        self.repo = self.root / "runtime/repositories/Grounded-SAM-2"
        self.checkpoint = self.root / "runtime/models/sam2/sam2.1_hiera_large.pt"
        self.model_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.grounding_model_path = self.root / "runtime/models/grounding-dino-base"
        self.output_dir = self.root / "runtime/outputs/masks"
        self._sam_predictor: Any | None = None
        self._processor: Any | None = None
        self._grounder: Any | None = None

    def load(self) -> None:
        if self._sam_predictor is not None:
            return
        if not self.repo.is_dir() or not self.checkpoint.is_file():
            raise FileNotFoundError(
                "Grounded-SAM-2 code or SAM2 checkpoint is incomplete under runtime/"
            )
        sys.path.insert(0, str(self.repo))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        sam = build_sam2(self.model_config, str(self.checkpoint), device=self.device)
        self._sam_predictor = SAM2ImagePredictor(sam)
        if not (self.grounding_model_path / "config.json").is_file():
            raise FileNotFoundError("Grounding DINO model download is incomplete")
        self._processor = AutoProcessor.from_pretrained(
            self.grounding_model_path, local_files_only=True)
        self._grounder = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model_path, local_files_only=True
        ).to(self.device)

    def segment(self, *, image_path: str, expression: str) -> SegmentationResult:
        self.load()
        import numpy as np
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        query = expression.strip().lower().rstrip(".") + "."
        self._sam_predictor.set_image(np.asarray(image))
        inputs = self._processor(images=image, text=query, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self._grounder(**inputs)
        detected = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = detected["boxes"].detach().cpu().numpy()
        if len(boxes) == 0:
            raise ValueError(f"no region found for expression: {expression}")
        masks, _, _ = self._sam_predictor.predict(
            point_coords=None, point_labels=None, box=boxes, multimask_output=False
        )
        masks = np.asarray(masks).squeeze(1) if np.asarray(masks).ndim == 4 else masks
        combined = np.any(masks, axis=0).astype("uint8") * 255
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / "grounded_sam2_mask.png"
        Image.fromarray(combined).save(output)
        return SegmentationResult(
            mask_path=str(output),
            boxes=boxes.tolist(),
            labels=list(detected["labels"]),
            scores=detected["scores"].detach().cpu().tolist(),
        )
