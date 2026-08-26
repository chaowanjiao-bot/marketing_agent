from marketing_agent.executor import TaskExecutor
from marketing_agent.schemas import ReviewDecision, TaskRequest
from marketing_agent.task_store import TaskStore
from marketing_agent.tools import build_default_registry


def test_executor_persists_multi_agent_decisions_and_messages(tmp_path):
    store = TaskStore(tmp_path / "tasks")
    request = TaskRequest(
        prompt="生成精华活动海报《焕亮新生》",
        orchestration_mode="multi_agent",
    )
    task_id = store.create(request)
    executor = TaskExecutor(store, build_default_registry())
    executor.execute_now(task_id, request)
    result = store.result_model(task_id)
    assert result is not None
    assert any(item["event"] == "director_decision" for item in result.trace)
    assert any(item["event"] == "agent_message" for item in result.trace)
    assert store.request(task_id).orchestration_mode == "multi_agent"


def test_multi_agent_human_revision_preserves_mode_and_feedback(tmp_path, monkeypatch):
    store = TaskStore(tmp_path / "tasks")
    request = TaskRequest(
        prompt="生成精华活动海报《焕亮新生》",
        orchestration_mode="multi_agent",
        review_required=True,
    )
    task_id = store.create(request)
    executor = TaskExecutor(store, build_default_registry())
    executor.execute_now(task_id, request)
    captured = {}

    def capture_submit(value_task_id, value_request, history=None):
        captured.update(task_id=value_task_id, request=value_request, history=history)

    monkeypatch.setattr(executor, "submit", capture_submit)
    executor.review(
        task_id, ReviewDecision.REVISE,
        feedback="产品放大到画面高度55%，标题上移",
        reviewer="creative_lead",
    )
    assert captured["request"].orchestration_mode == "multi_agent"
    assert captured["request"].review_feedback == "产品放大到画面高度55%，标题上移"
    assert captured["request"].review_round == 1
