# Day 1 model and environment audit

Date: 2026-08-18

## Server

- Ubuntu 22.04, Python 3.10.12.
- 8 x NVIDIA A800-SXM4-80GB.
- Docker, uv, rg and jq are not installed.
- Project storage is CephFS.

## Upstream agent skeleton

- Source: JoshuaC215/agent-service-toolkit.
- Locked commit: `d4147aca6e5e8149592fb2e524b5e528406ddee1`.
- The current upstream requires Python >=3.12,<3.15 and cannot run directly
  on the server's Python 3.10.
- Decision: retain upstream code and provenance for reference, while placing
  the four-day Python 3.10 implementation under `mvp/`.

## Correct target stack from the resume

- FLUX.1-dev: generation backend.
- PowerPaint: masked image-editing backend.
- Grounding-SAM2 / referring-expression segmentation: converts a natural
  language target or user annotation into the edit mask.
- GPT-4o-mini / MLLM: input interpretation, ambiguity resolution,
  clarification and creativity-level-controlled context completion.
- VQAScore: ten-dimension aesthetics and marketing-alignment evaluation;
  failed results return repair advice to the generation engine.

The runtime contracts are `generate_image -> FLUX.1-dev`,
`edit_image -> Grounding-SAM2 + PowerPaint`, and
`evaluate_image -> VQAScore`. GPU environments stay isolated behind typed
adapters to avoid dependency conflicts.

## Existing workspace models (excluded)

These were audited before the resume stack was confirmed. They are unrelated
experiments and must not be integrated into this project.

### Boogu-Image

- Apache-2.0.
- Supports both text-to-image and single-image editing.
- Local edit checkpoint exists at
  `Boogu-Image/models/Boogu-Image-0.1-Edit` (about 36 GB).
- Local Edit-Turbo directory is incomplete (about 955 MB), so it must not be
  treated as a ready checkpoint.
- The old `.venv` links to system Python and currently cannot import torch.
- Superseded decision: exclude it from this project's runtime; do not create
  an adapter.

### Step1X-Edit

- Apache-2.0.
- `inference.py` supports both `edit` and `t2i` task types.
- No complete local base checkpoint was found during the bounded audit.
- Existing repository is dirty and must not be modified.
- Superseded decision: exclude it from this project's runtime.

### Magic-Makeup

- A local LoRA weight exists, but the project is specialized for makeup
  transfer rather than general marketing generation/editing.
- Superseded decision: exclude it from this project's runtime.

## Environment decision

Agent dependencies are installed under the new project's `.packages/` and
loaded through `PYTHONPATH`. GPU model environments remain isolated from the
Agent runtime. This avoids Torch/CUDA/Diffusers conflicts.
