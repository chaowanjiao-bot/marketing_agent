# Day 1 acceptance record

Date: 2026-08-18

## Delivered

- Python 3.10 compatible isolated dependencies under `.packages/`.
- Pydantic domain schemas for goals, constraints, decisions, observations,
  assets, budget and final results.
- LangGraph state machine with generation, editing, evaluation, replanning,
  clarification and deterministic termination.
- Typed mock tool registry and two-level asset lineage.
- Resume-aligned adapter plan: FLUX.1-dev for generation, PowerPaint for
  editing, Grounding-SAM2/referring-expression segmentation for masks,
  GPT-4o-mini/MLLM for interpretation, and VQAScore for evaluation.
- Boogu-Image, Step1X-Edit and Magic-Makeup are explicitly excluded from this
  project's runtime.
- CLI demo and nine automated tests.

## Test evidence

Command:

```bash
PYTHONPATH="$PWD/.packages:$PWD/mvp" \
  python3 -m pytest -c mvp/pytest.ini mvp/tests
```

Expected result: `9 passed`.

## Scenario evidence

1. Text-to-image: generate -> evaluate(partial) -> generate ->
   evaluate(success) -> finish.
2. Image edit: edit -> evaluate(partial) -> edit -> evaluate(success) ->
   finish.
3. Ambiguous request: ask_user -> waiting_for_user.
4. Low iteration budget: abort with a deterministic terminal reason.

## Day 1 gate

PASS. The Agent behavior can be tested without a GPU. Real FLUX.1-dev,
PowerPaint and Grounding-SAM2 adapter smoke tests begin on Day 2.
