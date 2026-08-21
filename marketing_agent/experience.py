from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


class ExperienceRecord(BaseModel):
    experience_id: str = Field(default_factory=lambda: f"exp_{uuid4().hex[:12]}")
    issue: str = Field(min_length=1)
    action: str = Field(min_length=1)
    score_before: float
    score_after: float
    delta: float
    success: bool
    output_format: str = "1:1"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExperienceStore(Protocol):
    def add(self, record: ExperienceRecord) -> None: ...
    def records(self) -> list[ExperienceRecord]: ...


class InMemoryExperienceStore:
    def __init__(self) -> None:
        self.items: list[ExperienceRecord] = []
        self.lock = Lock()

    def add(self, record: ExperienceRecord) -> None:
        with self.lock:
            self.items.append(record)

    def records(self) -> list[ExperienceRecord]:
        with self.lock:
            return list(self.items)


class JsonlExperienceStore(InMemoryExperienceStore):
    """Opt-in persistence. Construction never creates a file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.items.append(ExperienceRecord.model_validate_json(line))

    def add(self, record: ExperienceRecord) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
            self.items.append(record)


class ExperienceMemory:
    def __init__(self, store: ExperienceStore | None = None) -> None:
        self.store = store or InMemoryExperienceStore()

    def learn_from_trace(
        self, trace: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None
    ) -> list[ExperienceRecord]:
        learned: list[ExperienceRecord] = []
        previous: dict[str, Any] | None = None
        for event in trace:
            if event.get("event") == "candidate_batch_selected":
                previous = None
                continue
            if event.get("event") != "observation" or event.get("tool") != "evaluate_image":
                continue
            metrics = dict(event.get("metrics") or {})
            score = metrics.get("marketing_alignment")
            if score is None:
                continue
            current = {
                "score": float(score), "issues": list(event.get("issues") or []),
                "actions": list(event.get("repair_actions") or []),
                "output_format": str(event.get("output_format") or "1:1"),
            }
            if previous is not None:
                delta = current["score"] - previous["score"]
                actions = previous["actions"]
                for index, issue in enumerate(previous["issues"]):
                    action = actions[index] if index < len(actions) else (
                        actions[-1] if actions else f"improve {issue}"
                    )
                    record = ExperienceRecord(
                        issue=issue, action=action,
                        score_before=previous["score"], score_after=current["score"],
                        delta=delta, success=delta >= 0.002,
                        output_format=previous["output_format"], metadata=metadata or {},
                    )
                    self.store.add(record)
                    learned.append(record)
            previous = current
        return learned

    def strategies(self, *, limit_per_issue: int = 2) -> dict[str, list[str]]:
        grouped: dict[tuple[str, str], list[ExperienceRecord]] = defaultdict(list)
        for record in self.store.records():
            grouped[(record.issue, record.action)].append(record)
        ranked: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for (issue, action), records in grouped.items():
            successes = sum(record.success for record in records)
            success_rate = (successes + 1) / (len(records) + 2)
            average_delta = sum(record.delta for record in records) / len(records)
            confidence = min(len(records) / 5, 1.0)
            rank_score = success_rate * 0.6 + average_delta * 4 * 0.4 * confidence
            if successes:
                ranked[issue].append((rank_score, action))
        return {
            issue: [action for _, action in sorted(items, reverse=True)[:limit_per_issue]]
            for issue, items in ranked.items()
        }

    def count(self) -> int:
        return len(self.store.records())
