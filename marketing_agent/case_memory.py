from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")
GENERIC_MARKERS = ("海报", "广告", "营销", "产品", "品牌", "电商", "社媒", "小红书")


def _tokens(text: str) -> list[str]:
    normalized = text.lower().strip()
    tokens = TOKEN_RE.findall(normalized)
    chinese = [normalized[index : index + 2] for index in range(len(normalized) - 1)
               if "\u4e00" <= normalized[index] <= "\u9fff"
               and "\u4e00" <= normalized[index + 1] <= "\u9fff"]
    return tokens + chinese


class HashEmbedding:
    """Dependency-free local embedding; replaceable by BGE without changing storage APIs."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in _tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions differ")
    return sum(a * b for a, b in zip(left, right))


@dataclass(frozen=True)
class RetrievalDecision:
    should_retrieve: bool
    reason: str
    query: str


class RetrievalGate:
    """Determines whether memory is useful before paying retrieval/model costs."""

    def decide(self, prompt: str, *, case_count: int, override: bool | None = None) -> RetrievalDecision:
        query = " ".join(prompt.split())
        if override is not None:
            return RetrievalDecision(override, "request_override", query)
        if case_count == 0:
            return RetrievalDecision(False, "memory_empty", query)
        if len(query) < 3:
            return RetrievalDecision(False, "query_too_short", query)
        if any(marker in query.lower() for marker in GENERIC_MARKERS):
            return RetrievalDecision(True, "marketing_task", query)
        # Image-generation requests normally benefit from style/layout precedents.
        return RetrievalDecision(True, "creative_precedent_useful", query)


class CaseMemory:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path, embedder: HashEmbedding | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashEmbedding()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL CHECK(source IN ('seed', 'generated')),
                    prompt TEXT NOT NULL,
                    enhanced_prompt TEXT NOT NULL,
                    asset_path TEXT,
                    score REAL,
                    compliant INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL
                )
                """
            )
            db.execute("CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_cases_score ON cases(score)")

    def count(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT COUNT(*) FROM cases WHERE status='active'").fetchone()[0])

    def add(
        self,
        *,
        prompt: str,
        enhanced_prompt: str = "",
        asset_path: str | None = None,
        score: float | None = None,
        compliant: bool = False,
        source: str = "seed",
        metadata: dict[str, Any] | None = None,
        case_id: str | None = None,
    ) -> str:
        if source not in {"seed", "generated"}:
            raise ValueError("source must be seed or generated")
        if len(prompt.strip()) < 3:
            raise ValueError("prompt is too short")
        case_id = case_id or f"case_{uuid4().hex[:12]}"
        document = "\n".join(part for part in (prompt.strip(), enhanced_prompt.strip()) if part)
        with self._connect() as db:
            db.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)",
                (
                    case_id, source, prompt.strip(), enhanced_prompt.strip(), asset_path,
                    score, int(compliant), json.dumps(metadata or {}, ensure_ascii=False),
                    json.dumps(self.embedder.embed(document)),
                    datetime.now(timezone.utc).isoformat(), self.SCHEMA_VERSION,
                ),
            )
        return case_id

    def search(self, query: str, *, limit: int = 3, min_score: float = 0.0) -> list[dict[str, Any]]:
        if limit < 1 or limit > 20:
            raise ValueError("limit must be between 1 and 20")
        query_vector = self.embedder.embed(query)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM cases WHERE status='active' AND (score IS NULL OR score>=?)",
                (min_score,),
            ).fetchall()
        results = []
        for row in rows:
            similarity = cosine(query_vector, json.loads(row["embedding_json"]))
            quality = float(row["score"] or 0.0)
            rank_score = 0.8 * similarity + 0.2 * quality
            results.append({
                "case_id": row["case_id"], "source": row["source"],
                "prompt": row["prompt"], "enhanced_prompt": row["enhanced_prompt"],
                "asset_path": row["asset_path"], "score": row["score"],
                "compliant": bool(row["compliant"]), "similarity": similarity,
                "rank_score": rank_score, "metadata": json.loads(row["metadata_json"]),
            })
        return sorted(results, key=lambda item: item["rank_score"], reverse=True)[:limit]

    def retrieve_if_needed(
        self, prompt: str, *, limit: int = 3, override: bool | None = None
    ) -> tuple[RetrievalDecision, list[dict[str, Any]]]:
        decision = RetrievalGate().decide(prompt, case_count=self.count(), override=override)
        return decision, self.search(decision.query, limit=limit) if decision.should_retrieve else []

    def seed(self, examples: Iterable[dict[str, Any]]) -> list[str]:
        return [self.add(source="seed", **example) for example in examples]


def format_retrieval_context(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return ""
    lines = ["参考以下历史优秀案例的构图、风格和约束，但不要照搬文案或品牌："]
    for index, case in enumerate(cases, 1):
        reference = case["enhanced_prompt"] or case["prompt"]
        lines.append(f"案例{index}：{reference}")
    return "\n".join(lines)
