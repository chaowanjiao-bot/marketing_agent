import pytest
from pydantic import ValidationError

from marketing_agent import run_task
from marketing_agent.schemas import (
    BudgetState,
    Decision,
    DecisionType,
    TaskRequest,
    TaskType,
)
from marketing_agent.tools import MockGenerateTool, ToolRegistry


def test_task_request_rejects_short_prompt():
    with pytest.raises(ValidationError):
        TaskRequest(prompt="x")


def test_call_tool_requires_tool_name():
    with pytest.raises(ValidationError):
        Decision(type=DecisionType.CALL_TOOL, reason_summary="需要调用工具")


def test_non_tool_decision_rejects_tool_name():
    with pytest.raises(ValidationError):
        Decision(
            type=DecisionType.FINISH,
            reason_summary="任务已经完成",
            tool_name="generate_image",
        )


def test_registry_rejects_duplicate_tool():
    registry = ToolRegistry()
    registry.register(MockGenerateTool())
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(MockGenerateTool())


def test_budget_rejects_overspend():
    budget = BudgetState(max_cost=1.0)
    with pytest.raises(ValueError, match="budget exceeded"):
        budget.charge(1.1)


def test_text_to_image_routes_to_generate_and_replans():
    result = run_task(
        TaskRequest(prompt="生成一张七夕香水营销海报", max_iterations=6)
    )
    assert result.status == "completed"
    assert result.goal is not None
    assert result.goal.task_type == TaskType.TEXT_TO_IMAGE
    tools = [e.get("tool") for e in result.trace if e["event"] == "decision"]
    assert tools.count("generate_image") == 2
    assert tools.count("evaluate_image") == 2
    assert len(result.assets) == 2
    assert result.assets[1].parent_id == result.assets[0].asset_id


def test_image_edit_routes_to_edit():
    result = run_task(
        TaskRequest(
            prompt="保持产品不变，把背景替换成黑金风格",
            input_image="product.png",
            max_iterations=6,
        )
    )
    assert result.status == "completed"
    assert result.goal is not None
    assert result.goal.task_type == TaskType.IMAGE_EDIT
    decision_tools = [
        e.get("tool") for e in result.trace if e["event"] == "decision"
    ]
    assert "edit_image" in decision_tools
    assert "generate_image" not in decision_tools
    assert result.goal.hard_constraints[0].name == "product_identity_preserved"


def test_ambiguous_request_waits_for_user():
    result = run_task(TaskRequest(prompt="帮我改好看一点"))
    assert result.status == "waiting_for_user"
    assert result.goal is not None
    assert result.goal.task_type == TaskType.AMBIGUOUS
    assert result.goal.uncertainties


def test_max_iterations_aborts_safely():
    result = run_task(
        TaskRequest(prompt="生成一张高级香水海报", max_iterations=1)
    )
    assert result.status == "aborted"
    assert result.terminal_reason == "max_iterations_or_invalid_state"
