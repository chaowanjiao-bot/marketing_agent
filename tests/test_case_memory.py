from pathlib import Path

from marketing_agent.case_memory import CaseMemory, HashEmbedding, RetrievalGate


def test_empty_memory_skips_retrieval(tmp_path: Path) -> None:
    memory = CaseMemory(tmp_path / "cases.sqlite3")
    decision, cases = memory.retrieve_if_needed("生成高端护肤品广告")
    assert decision.should_retrieve is False
    assert decision.reason == "memory_empty"
    assert cases == []


def test_retrieves_relevant_high_quality_case(tmp_path: Path) -> None:
    memory = CaseMemory(tmp_path / "cases.sqlite3")
    memory.add(prompt="高端护肤精华产品广告 海报", enhanced_prompt="香槟金 棚拍", score=0.91,
               compliant=True, asset_path="/assets/serum.png")
    memory.add(prompt="运动鞋户外广告", enhanced_prompt="山地奔跑", score=0.95, compliant=True)
    decision, cases = memory.retrieve_if_needed("生成高端护肤精华海报", limit=1)
    assert decision.should_retrieve is True
    assert cases[0]["asset_path"] == "/assets/serum.png"


def test_generated_case_is_persisted(tmp_path: Path) -> None:
    path = tmp_path / "cases.sqlite3"
    first = CaseMemory(path)
    case_id = first.add(prompt="美妆社媒图", source="generated", score=0.88,
                        compliant=True, metadata={"task_id": "task_1"})
    second = CaseMemory(path)
    result = second.search("美妆社媒", limit=1)[0]
    assert result["case_id"] == case_id
    assert result["source"] == "generated"
    assert result["metadata"]["task_id"] == "task_1"


def test_override_can_disable_memory() -> None:
    decision = RetrievalGate().decide("生成护肤广告", case_count=10, override=False)
    assert decision.should_retrieve is False
    assert decision.reason == "request_override"


def test_embedding_is_deterministic_and_normalized() -> None:
    embedder = HashEmbedding(64)
    first = embedder.embed("护肤产品广告")
    assert first == embedder.embed("护肤产品广告")
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
