# Multi-Agent Operations Guide

## Reliability policy

Specialist nodes use a bounded retry policy. Only transient timeout and
connection failures are retried; schema, policy and programming errors fail
fast. Every attempt produces an `AgentRunMetric`, so a successful retry does
not hide the original failure. Retry budgets are deliberately small to avoid
runaway model cost.

Full `MultiAgentCampaign` state can be written atomically with
`CampaignCheckpointStore`. Restoring that model and passing it to
`run_multi_agent_campaign` resumes from retained brief, strategies, layouts,
assets, reviews and budgets. Production workers should save after every
`CampaignCheckpoint` and use a single-writer lease around each campaign.

## Observability

The final result trace contains:

- `agent_run_metric`: node, role, attempt, status, latency and estimated cost;
- `campaign_checkpoint`: recovery boundary and retained asset IDs;
- `campaign_run_report`: aggregate retries, cost, duration and gate outcome.

Task-level JSON is exposed at `GET /tasks/{task_id}/observability` and preserves
the same authorization boundary as task results. Cost values are configuration
estimates for capacity planning, not provider invoices.

## Suggested production alerts

- failed-attempt ratio above 10% over 15 minutes;
- p95 generation latency above the configured GPU service objective;
- compliance pass rate dropping more than 20% from the seven-day baseline;
- checkpoint age exceeding the worker visibility timeout;
- estimated campaign cost exceeding the per-task budget.
