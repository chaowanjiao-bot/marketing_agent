# Multi-Agent Marketing Creative Architecture

## Scope

The multi-agent workflow is an additive replacement for the original single
decision loop. Existing image generation, segmentation, editing, VQA/OCR,
GPU scheduling, asset storage and API modules remain reusable.

## Roles

- `CreativeDirectorAgent`: safety-first supervisor, routing, strategy selection,
  conflict resolution, revision planning and termination.
- `BriefAgent`: converts a request into a schema-valid `CreativeBrief`; it may
  ask for user clarification but must not invent product facts or legal claims.
- Strategy panel: brand, performance and merchandising specialists create
  independent proposals in parallel.
- `LayoutAgent`: produces normalized coordinates, safe areas and a generation
  prompt that reserves typography regions.
- `GenerationAgent`: calls only registered generation/editing tools and creates
  immutable `AssetVersion` records.
- `QualityCriticAgent`: reviews aesthetic and marketing alignment.
- `ComplianceAgent`: owns hard vetoes for critical text and brand rules.

## Collaboration contract

Agents communicate through `AgentMessage`. Every message has a correlation ID,
sender, receiver, message type, structured payload, evidence and confidence.
Large images never enter the message history; agents exchange asset IDs and
registered paths instead.

## Safety and deterministic policy

The Director is intentionally deterministic by default. A structured LLM may
replace specialist reasoning, but the following rules stay outside model
control:

1. Legal/platform rules outrank brand, copy, conversion and aesthetic goals.
2. Compliance hard vetoes cannot be averaged away by an aesthetic score.
3. Only the top evidence-backed findings enter a revision plan.
4. Generation and supervisor rounds have explicit budgets.
5. Critical copy should be rendered as a deterministic typography layer.
6. Human review is a terminal delivery gate when requested.

## Graph

```text
director -> brief -> director
director -> strategy panel (3 agents in parallel) -> director
director -> layout -> director
director -> generation -> director
director -> review panel (quality + compliance in parallel) -> director
director -> layout/generation revision or terminal state
```

## Provider integration

`StructuredModel` is a provider-neutral protocol. An OpenAI, Qwen, local vLLM
or other adapter must implement schema-constrained `invoke`. The deterministic
fallback keeps development and contract testing independent from model access.

## Production integration

The workflow can be invoked by `scripts/run_multi_agent.py` or selected through
`TaskRequest.orchestration_mode="multi_agent"`. The existing API persists that
request, external workers restore it from the durable queue, and the executor
adapts the completed campaign into the existing `FinalResult` contract. Agent
messages and Director decisions are retained in the result trace.

Each asset is evaluated once. The shared Observation is then consumed in
parallel by Quality Critic and Compliance Agent, avoiding duplicate VQA/OCR
model execution. Requested output formats are processed independently until
every format passes both gates or the generation budget is exhausted.

## Structured revision and human feedback

`RevisionPlan` contains typed operations and parameters rather than prose only.
Layout repairs can move named layers into safe areas, adjust visual hierarchy,
or force deterministic typography re-rendering. Human review feedback is
classified into strategy, layout, generation and compliance dimensions and
sent as correlated `revision_request` messages to the responsible Agents.

Every Director round is persisted as a `DirectorRecord`. When Quality Critic
and Compliance Agent disagree, a `ConflictResolution` records the parties,
priority basis, winning constraint and resolution. Both records are exported
to the existing task trace for dashboard and audit use.

## Evaluation protocol

`benchmarks/multi_agent_acceptance_cases.json` is the fixed functional suite.
`scripts/run_multi_agent_benchmark.py` compares direct generation, the original
single Agent and the multi-agent system. Supported ablations remove parallel
strategy exploration, Compliance Agent, or structured layout repair. Production
claims require real tools, fixed model versions, fixed hardware and human gold
labels; mock reports are development checks only.
