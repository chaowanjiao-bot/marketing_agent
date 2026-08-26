from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, TypeVar

from .contracts import AgentRole, AgentRunMetric

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 2
    retryable_errors: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)

    def __post_init__(self) -> None:
        if self.max_attempts < 1 or self.max_attempts > 5:
            raise ValueError("max_attempts must be between 1 and 5")


class AgentExecutionError(RuntimeError):
    def __init__(self, node: str, cause: Exception, metrics: list[AgentRunMetric]) -> None:
        super().__init__(f"{node} failed after {len(metrics)} attempt(s): {cause}")
        self.node = node
        self.cause = cause
        self.metrics = metrics


def execute_with_retry(
    *, node: str, agent: AgentRole, operation: Callable[[], T],
    policy: RetryPolicy, estimated_cost_usd: float = 0.0,
) -> tuple[T, list[AgentRunMetric]]:
    metrics: list[AgentRunMetric] = []
    for attempt in range(1, policy.max_attempts + 1):
        started = perf_counter()
        try:
            result = operation()
        except Exception as exc:
            metrics.append(AgentRunMetric(
                agent=agent, node=node, attempt=attempt, status="failed",
                duration_ms=(perf_counter() - started) * 1000,
                error_type=type(exc).__name__, error_message=str(exc)[:500],
            ))
            if not isinstance(exc, policy.retryable_errors) or attempt == policy.max_attempts:
                raise AgentExecutionError(node, exc, metrics) from exc
        else:
            metrics.append(AgentRunMetric(
                agent=agent, node=node, attempt=attempt, status="succeeded",
                duration_ms=(perf_counter() - started) * 1000,
                estimated_cost_usd=estimated_cost_usd,
            ))
            return result, metrics
    raise AssertionError("retry loop exited unexpectedly")
