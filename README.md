# Marketing Creative Multi-Agent

A production-oriented, LangGraph-based system for generating, editing, reviewing, and delivering marketing creatives.

The system uses a deterministic Creative Director to coordinate specialized Brief, Brand Strategy, Performance Strategy, Merchandising Strategy, Layout, Generation, Quality Critic, and Compliance agents. They collaborate through typed messages, parallel proposals, independent review gates, evidence-backed revisions, and explicit execution budgets.

The original single-agent quality loop remains available as a baseline for direct-generation, single-agent, multi-agent, and ablation comparisons.

> Status: production-oriented prototype. Mock tools are included for development, while real model adapters target GPU deployment. Model weights are not included.

## Highlights

- Supervisor–Specialist multi-agent orchestration with LangGraph
- Parallel Brand, Performance, and Merchandising strategy proposals
- Independent Quality Critic and Compliance Agent with hard-veto rules
- Structured revision plans routed to the responsible agent
- Qwen-Image generation and PowerPaint local editing
- Grounding DINO + SAM2 localization and segmentation
- Qwen2.5-VL/VQAScore evaluation and OCR validation
- Deterministic typography, safe-area, layout, and brand-rule tools
- Multi-format delivery for `1:1`, `4:5`, `9:16`, and `16:9`
- Top-K candidate generation with reproducible seeds
- Human review, feedback routing, and approval history
- FastAPI, durable SQLite queue, independent GPU worker, and web workspace
- User/project isolation and optional C2PA provenance
- Bounded retries, atomic checkpoints, cost estimates, and observability
- Fixed acceptance datasets, system comparisons, and ablation experiments

## Architecture

```text
User / API
    |
Task Executor
    |
Creative Director
    |-- Brief Agent
    |-- Brand Strategy Agent ---------+
    |-- Performance Strategy Agent ---+  parallel proposal panel
    |-- Merchandising Strategy Agent -+
    |-- Layout Agent
    |-- Generation Agent -> image generation / local editing tools
    |-- Shared VQA + OCR evaluation
    |       |-- Quality Critic Agent
    |       `-- Compliance Agent
    |-- Structured revision / conflict resolution
    `-- Human review or delivery
```

The graph defines safe execution boundaries without prescribing one fixed sequence. The Director inspects campaign state and decides which agents run next, whether they execute in parallel, which findings deserve revision, and when the campaign should stop.

Legal rules, compliance vetoes, tool permissions, budgets, and termination conditions remain enforced in code even when a structured model assists specialist reasoning.

See [the architecture guide](docs/multi-agent-architecture.md) and [operations guide](docs/operations.md) for details.

## Workflow

1. Brief Agent converts the request into a schema-valid creative brief.
2. Three strategy agents independently propose brand, conversion, and merchandising directions.
3. Creative Director selects a strategy using confidence, strengths, and risk evidence.
4. Layout Agent creates format-specific normalized layouts and typography reservations.
5. Generation Agent creates immutable asset versions through registered tools.
6. The image is evaluated once; Quality Critic and Compliance Agent reuse the observation.
7. Director resolves conflicts and creates typed repair operations for Layout or Generation.
8. Each output format continues independently until it passes both gates or exhausts its budget.
9. The campaign completes, waits for human review, requests clarification, or stops with an explicit reason.

Structured repair operations include:

- `move_inside_safe_area`
- `rerender_typography`
- `adjust_visual_hierarchy`
- `regenerate_asset`
- `local_image_edit`

Each repair identifies its target agent, source findings, priority, parameters, and elements that must be preserved. Director decisions and review conflicts are retained in the final trace.

## Model Stack

| Capability | Backend |
|---|---|
| Text-to-image | Qwen-Image |
| Target localization | Grounding DINO |
| Segmentation | SAM2 |
| Local image editing | PowerPaint v2 |
| Visual quality | Qwen2.5-VL / VQAScore |
| OCR and text validation | Qwen2.5-VL |
| Structured reasoning | Optional OpenAI-compatible endpoint |

GPU models are isolated behind typed JSON subprocess adapters so incompatible Torch, CUDA, and Diffusers environments do not need to coexist in the API environment.

## Repository Layout

```text
marketing_agent/
  agents/          Specialist agents and Creative Director
  contracts/       Typed messages and campaign state
  workflows/       LangGraph orchestration and result adapters
  adapters/        Generation, editing, VQA, and OCR adapters
  web/             User-facing workspace
  api.py           FastAPI application
  executor.py      Inline/external task execution
  tools.py         Tool registry and capability restrictions
benchmarks/        Fixed acceptance cases
deploy/            Production configuration and service templates
docs/              Architecture, acceptance, and operations notes
scripts/           CLI, benchmark, acceptance, and setup commands
tests/             Unit, workflow, API, security, and deployment tests
```

## Requirements and Installation

- Python 3.10
- CUDA 12.x for production tools
- NVIDIA GPU; 80 GB VRAM is recommended for the complete local stack

```bash
python -m venv runtime/venv/gpu
source runtime/venv/gpu/bin/activate
pip install -r requirements/base.txt
cp .env.example .env
bash scripts/setup_model_envs.sh all
```

Expected model directories:

```text
runtime/models/Qwen-Image/
runtime/models/Qwen2.5-VL-3B-Instruct/
runtime/models/PowerPaint-v2/
runtime/models/grounding-dino-base/
runtime/models/sam2/
```

Weights, generated assets, datasets, environments, credentials, and runtime reports are excluded from Git.

## Quick Start

Run with development tools:

```bash
python scripts/run_multi_agent.py \
  "Create a premium skincare launch poster titled Radiant Renewal" \
  --formats 1:1 4:5 \
  --review-required
```

Run with production adapters:

```bash
python scripts/run_multi_agent.py \
  "Create a premium skincare launch poster" \
  --production \
  --quality-threshold 0.80
```

Production multi-agent mode requires `TYPOGRAPHY_FONT_PATH`. A real structured Brief/Strategy model can be configured with:

```bash
STRUCTURED_MODEL_BASE_URL=https://your-endpoint/v1
STRUCTURED_MODEL_NAME=your-model
STRUCTURED_MODEL_API_KEY=your-key
```

Without these values, deterministic specialist fallbacks are used.

## API

```bash
bash scripts/run_api.sh
```

Submit a multi-agent task:

```json
{
  "orchestration_mode": "multi_agent",
  "prompt": "Create a premium serum campaign poster titled Radiant Renewal",
  "output_formats": ["1:1", "4:5"],
  "candidate_count": 2,
  "review_required": true
}
```

Key endpoints:

```text
POST /tasks
GET  /tasks
GET  /tasks/{task_id}
GET  /tasks/{task_id}/events
GET  /tasks/{task_id}/result
POST /tasks/{task_id}/review
GET  /tasks/{task_id}/observability
GET  /health
```

The user workspace is available at `/app`; the experiment dashboard is available at `/dashboard` when authentication policy allows it.

## Multi-Format, Candidates, and Human Review

`candidate_count` requests 1–8 reproducible candidates. Selection prioritizes text and compliance requirements before marketing score.

`output_formats` accepts `1:1`, `4:5`, `9:16`, and `16:9`. Each format receives an independent layout, safe-area policy, review cycle, and `FormatSummary`.

Set `review_required=true` to stop at the delivery gate. Feedback is classified and routed to the responsible agents:

```bash
curl -X POST http://127.0.0.1:8000/tasks/TASK_ID/review \
  -H 'Content-Type: application/json' \
  -d '{"decision":"revise","feedback":"Increase the product size and move the headline upward"}'
```

## Reliability and Observability

Specialist nodes use bounded retries for transient timeout and connection failures. Every attempt records its agent, node, status, latency, error category, and estimated cost.

Campaign state can be saved atomically after key nodes and restored with its brief, strategies, layouts, assets, evaluations, reviews, and remaining budget.

```bash
MULTI_AGENT_MAX_NODE_ATTEMPTS=2
MULTI_AGENT_CHECKPOINTS=true
MULTI_AGENT_CHECKPOINT_DIR=runtime/checkpoints
```

Task metrics are available from `GET /tasks/{task_id}/observability`. Cost fields are capacity-planning estimates, not provider invoices.

## Evaluation and Ablations

Compare direct generation, the original single-agent loop, and the multi-agent system:

```bash
python scripts/run_multi_agent_benchmark.py \
  --system all \
  --report runtime/multi_agent_benchmark.json
```

Run ablations:

```bash
python scripts/run_multi_agent_benchmark.py --system multi_agent --ablation no_parallel_strategy
python scripts/run_multi_agent_benchmark.py --system multi_agent --ablation no_compliance
python scripts/run_multi_agent_benchmark.py --system multi_agent --ablation no_structured_repair
```

Reports include completion rate, average score, generation count, latency, Director rounds, and conflict count. Resume or publication claims should be reproduced with fixed model versions, fixed hardware, real tools, and human gold labels.

## Testing

```bash
PYTHONPATH="$PWD" python -m pytest -q tests
```

The suite covers contracts, routing, tools, model adapters, layout repair, human feedback, persistence, API behavior, authentication isolation, queues, provenance, observability, and deployment contracts.

This README deliberately does not claim a current passing-test count until the final suite is rerun in a compatible environment.

## Production Deployment

```bash
cp deploy/production.env.example deploy/production.env
deploy/manage.sh start
deploy/manage.sh status
```

The deployment separates FastAPI from the GPU worker, uses a durable SQLite WAL queue, limits GPU concurrency, and supports stale-job recovery. Systemd templates and an Nginx HTTPS example are included under `deploy/`.

For multi-host deployments, replace SQLite with Redis or another centralized broker while keeping executor and API contracts unchanged.

## Authentication and Provenance

```bash
AUTH_ENABLED=true
AUTH_DATABASE_PATH=runtime/accounts.sqlite3
AUTH_SECURE_COOKIE=true
```

Passwords use salted PBKDF2-SHA256. Session tokens are stored as hashes and delivered through HttpOnly, SameSite cookies. Task, event, result, asset, review, and observability endpoints enforce ownership.

The optional provenance pipeline creates prompt-hash manifests and can sign assets through `c2patool`. `manifest_only` and `signed_and_verified` are reported separately; test certificates must not be used in production.

## Current Limitations

- The full pipeline requires a compatible CUDA/GPU environment and separately downloaded weights.
- SQLite targets single-host deployment; multi-host workers require a centralized queue.
- Diffusion models cannot guarantee perfect text; OCR detects failures but cannot guarantee repair.
- PowerPaint targets image-editing and local-repair tasks; text-to-image failures may require regeneration.
- Checkpoint storage exists, while automatic node-level worker resume still requires lease integration.
- Reported node costs are estimates until connected to billing or measured GPU accounting.

## Security

- Never commit `.env` files, credentials, model weights, datasets, logs, or generated assets.
- Run the test suite and inspect `git status --short` before release.
- This repository does not redistribute model weights; review each model license before use.

## License

Apache-2.0. See [LICENSE](LICENSE).
