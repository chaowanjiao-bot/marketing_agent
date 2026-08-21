from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from .brief import CreativityLevel, MarketingBrief, MarketingInputInterpreter
from .copy_agent import MarketingCopy, MarketingCopyAgent

from .schemas import (
    AssetVersion,
    Constraint,
    Decision,
    DecisionType,
    FinalResult,
    GoalSpec,
    Observation,
    ObservationStatus,
    TaskRequest,
    TaskType,
)
from .tools import ToolRegistry, build_default_registry


class AgentState(TypedDict):
    request: TaskRequest
    goal: GoalSpec | None
    brief: MarketingBrief | None
    decision: Decision | None
    observations: list[Observation]
    assets: list[AssetVersion]
    trace: list[dict[str, Any]]
    phase: str
    iteration: int
    generation_attempt: int
    evaluation_attempt: int
    waiting_for_user: bool
    terminal_reason: str | None
    best_asset_id: str | None
    best_score: float | None
    best_aesthetic_asset_id: str | None
    best_aesthetic_score: float | None
    best_compliant_asset_id: str | None
    best_compliant_score: float | None
    no_improvement_count: int
    marketing_copy: MarketingCopy | None


AMBIGUOUS_PHRASES = ("改好看", "优化一下", "随便改", "make it better")
HARD_CONSTRAINT_MARKERS = {
    "产品位置不要变": "product_position_unchanged",
    "保持产品不变": "product_identity_preserved",
    "不要logo": "no_logo",
    "无logo": "no_logo",
    "纯白背景": "pure_white_background",
}

BASE_SEED = 42
SEED_STRIDE = 1009
MIN_MEANINGFUL_IMPROVEMENT = 0.002
MAX_NO_IMPROVEMENT = 2
REPAIR_INSTRUCTIONS = {
    "composition": "优化构图：产品瓶严格居中并占画面高度约45%，主标题、副标题、产品和品牌形成清晰的纵向层级，四周保留安全边距",
    "color": "统一暖象牙白与香槟金配色，减少脏灰和过饱和区域，保持产品、背景与文字之间有清晰对比",
    "lighting": "使用柔和电影级棚拍主光和克制轮廓光，瓶身标签清晰可见，高光不过曝",
    "focus": "强化唯一精华瓶作为视觉中心，背景光环和丝绸只作陪衬，不得抢夺主体注意力",
    "emotion": "增强奢华、温润、焕亮的新生氛围，同时保持画面克制而高级",
    "creativity": "增加精致但克制的水面倒影与金色光环层次，避免模板化和无关装饰",
    "subject": "确保画面只有一支完整的象牙白玻璃精华瓶，瓶体、滴管和标签结构准确",
    "scene": "明确呈现高端护肤品商业棚拍场景：圆形展台、丝绸、水面反射和柔和金色背景",
    "spatial": "产品位于圆台中心，标题置于顶部安全区，品牌位于底部安全区，元素不得重叠或贴边",
    "brand": "保持国际高端美妆广告风格，LUMIÈRE仅清晰出现一次，删除乱码、伪文字和重复品牌名",
    "text_accuracy": "逐字准确显示指定的中文标题、副标题和品牌名，不得缺字、错字或乱码",
    "text_duplication": "每个指定文本只允许出现一次，删除瓶身或底部重复的品牌名和重复标题",
}


def generation_seed(attempt: int) -> int:
    return BASE_SEED + max(attempt - 1, 0) * SEED_STRIDE


def repair_actions_for(observation: Observation) -> list[str]:
    concrete = [REPAIR_INSTRUCTIONS[name] for name in observation.issues if name in REPAIR_INSTRUCTIONS]
    return concrete or list(observation.recommended_actions) or ["提升主体清晰度、版式层级和品牌一致性"]


def initial_state(request: TaskRequest) -> AgentState:
    return {
        "request": request,
        "goal": None,
        "brief": None,
        "decision": None,
        "observations": [],
        "assets": [],
        "trace": [],
        "phase": "understand",
        "iteration": 0,
        "generation_attempt": 0,
        "evaluation_attempt": 0,
        "waiting_for_user": False,
        "terminal_reason": None,
        "best_asset_id": None,
        "best_score": None,
        "best_aesthetic_asset_id": None,
        "best_aesthetic_score": None,
        "best_compliant_asset_id": None,
        "best_compliant_score": None,
        "no_improvement_count": 0,
        "marketing_copy": None,
    }


def understand_goal(state: AgentState) -> dict[str, Any]:
    request = state["request"]
    brief = MarketingInputInterpreter().interpret(request.prompt, request.creativity)
    brand = request.brand_profile
    if brand:
        violations = brand.violations(request.prompt)
        if violations:
            raise ValueError("request contains forbidden brand phrases: " + ", ".join(violations))
    marketing_copy = MarketingCopyAgent().create(brief, brand) if request.generate_copy else None
    low_creativity_needs_clarification = (
        request.creativity == CreativityLevel.LOW and bool(brief.ambiguities)
    )
    lower = request.prompt.lower()
    if any(marker in lower for marker in AMBIGUOUS_PHRASES) or low_creativity_needs_clarification:
        task_type = TaskType.AMBIGUOUS
        uncertainties = brief.clarification_questions or ["缺少明确的修改目标、区域或风格"]
    elif request.input_image:
        task_type = TaskType.IMAGE_EDIT
        uncertainties = []
    else:
        task_type = TaskType.TEXT_TO_IMAGE
        uncertainties = []

    hard_constraints = [
        Constraint(name=name, description=phrase, hard=True)
        for phrase, name in HARD_CONSTRAINT_MARKERS.items()
        if phrase in lower
    ]
    goal = GoalSpec(
        business_goal=request.prompt,
        task_type=task_type,
        hard_constraints=hard_constraints,
        soft_preferences=[],
        uncertainties=uncertainties,
    )
    return {
        "goal": goal,
        "phase": "decide",
        "brief": brief,
        "marketing_copy": marketing_copy,
        "trace": state["trace"]
        + [{"event": "goal_understood", "task_type": task_type.value}],
    }


def generation_context(state: AgentState) -> str:
    sections = [state["goal"].business_goal] if state["goal"] else []
    if state["request"].brand_profile:
        sections.append(state["request"].brand_profile.prompt_context())
    if state["marketing_copy"]:
        sections.append(state["marketing_copy"].prompt_context())
    if state["request"].memory_context:
        sections.append(state["request"].memory_context)
    return "\n\n".join(section for section in sections if section)


def decide(state: AgentState) -> dict[str, Any]:
    goal = state["goal"]
    if goal is None:
        raise RuntimeError("goal is missing")

    if state["terminal_reason"] == "quality_plateau":
        decision = Decision(
            type=DecisionType.ABORT,
            reason_summary="连续两轮评分没有显著提升，提前停止并保留最佳版本",
        )
    elif state["iteration"] >= state["request"].max_iterations:
        decision = Decision(
            type=DecisionType.ABORT,
            reason_summary="达到最大迭代次数，安全终止",
        )
    elif goal.task_type == TaskType.AMBIGUOUS:
        decision = Decision(
            type=DecisionType.ASK_USER,
            reason_summary="关键信息不足，需要用户明确修改目标",
        )
    elif state["phase"] == "decide":
        tool_name = (
            "edit_image"
            if goal.task_type == TaskType.IMAGE_EDIT
            else "generate_image"
        )
        generation_prompt = generation_context(state)
        decision = Decision(
            type=DecisionType.CALL_TOOL,
            reason_summary=f"根据任务类型选择{tool_name}",
            tool_name=tool_name,
            arguments={
                "prompt": generation_prompt,
                "input_image": state["request"].input_image,
                "target_expression": state["request"].target_expression,
                "attempt": state["generation_attempt"] + 1,
                "seed": generation_seed(state["generation_attempt"] + 1),
            },
            success_criteria=["生成有效图片资产"],
        )
    elif state["phase"] == "evaluate":
        if not state["assets"]:
            raise RuntimeError("evaluation requires a generated asset")
        decision = Decision(
            type=DecisionType.CALL_TOOL,
            reason_summary="调用质量评估工具获取Observation",
            tool_name="evaluate_image",
            arguments={
                "attempt": state["evaluation_attempt"] + 1,
                "image_path": state["assets"][-1].file_path,
                "campaign_text": goal.business_goal,
            },
            success_criteria=["营销对齐分数达到0.7"],
        )
    elif state["phase"] == "replan":
        latest_observation = state["observations"][-1]
        repair_actions = repair_actions_for(latest_observation)
        repair_instruction = (
            "；改进建议：" + "；".join(repair_actions) if repair_actions else "；提升营销对齐质量"
        )
        repair_instruction += "；只修改上述低分项，保持其余已达标的主体、色彩、光照、文字和版式不变"
        tool_name = (
            "edit_image"
            if goal.task_type == TaskType.IMAGE_EDIT
            else "generate_image"
        )
        decision = Decision(
            type=DecisionType.CALL_TOOL,
            reason_summary="评估未通过，根据Observation重新执行修复",
            tool_name=tool_name,
            arguments={
                "prompt": generation_context(state) + repair_instruction,
                "input_image": state["request"].input_image,
                "attempt": state["generation_attempt"] + 1,
                "target_expression": state["request"].target_expression,
                "seed": generation_seed(state["generation_attempt"] + 1),
            },
            success_criteria=["修复评估器指出的问题"],
        )
    elif state["phase"] == "finish":
        decision = Decision(
            type=DecisionType.FINISH,
            reason_summary="质量评估通过，任务完成",
        )
    else:
        decision = Decision(
            type=DecisionType.ABORT,
            reason_summary=f"未知阶段：{state['phase']}",
        )

    return {
        "decision": decision,
        "iteration": state["iteration"] + 1,
        "trace": state["trace"]
        + [
            {
                "event": "decision",
                "type": decision.type.value,
                "tool": decision.tool_name,
                "reason": decision.reason_summary,
            }
        ],
    }


def make_execute_tool(registry: ToolRegistry):
    def execute_tool(state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        if decision is None or decision.tool_name is None:
            raise RuntimeError("tool decision is missing")
        observation = registry.get(decision.tool_name).execute(decision.arguments)
        updates: dict[str, Any] = {
            "observations": state["observations"] + [observation],
            "trace": state["trace"]
            + [
                {
                    "event": "observation",
                    "tool": observation.tool_name,
                    "status": observation.status.value,
                    "issues": observation.issues,
                }
            ],
        }
        if decision.tool_name in {"generate_image", "edit_image"}:
            parent = state["assets"][-1].asset_id if state["assets"] else None
            asset = AssetVersion(
                parent_id=parent,
                tool_name=decision.tool_name,
                file_path=str(observation.outputs["file_path"]),
                prompt=str(observation.outputs["prompt"]),
                seed=int(observation.outputs["seed"]),
            )
            updates.update(
                assets=state["assets"] + [asset],
                generation_attempt=state["generation_attempt"] + 1,
                phase="evaluate",
            )
        else:
            next_attempt = state["evaluation_attempt"] + 1
            updates["evaluation_attempt"] = next_attempt
            score = float(observation.metrics.get("marketing_alignment", 0.0))
            text_accuracy = observation.metrics.get("text_accuracy", 1.0)
            text_uniqueness = observation.metrics.get("text_uniqueness", 1.0)
            compliant = text_accuracy >= 1.0 and text_uniqueness >= 1.0
            previous_aesthetic = state["best_aesthetic_score"]
            is_aesthetic_best = previous_aesthetic is None or score > previous_aesthetic
            previous_compliant = state["best_compliant_score"]
            is_compliant_best = compliant and (
                previous_compliant is None or score > previous_compliant
            )
            became_compliant = compliant and previous_compliant is None
            comparison_best = previous_compliant if compliant else previous_aesthetic
            meaningful = became_compliant or comparison_best is None or (
                score > comparison_best + MIN_MEANINGFUL_IMPROVEMENT
            )
            no_improvement_count = 0 if meaningful else state["no_improvement_count"] + 1
            current_asset_id = state["assets"][-1].asset_id
            aesthetic_asset_id = (
                current_asset_id if is_aesthetic_best else state["best_aesthetic_asset_id"]
            )
            aesthetic_score = score if is_aesthetic_best else previous_aesthetic
            compliant_asset_id = (
                current_asset_id if is_compliant_best else state["best_compliant_asset_id"]
            )
            compliant_score = score if is_compliant_best else previous_compliant
            updates.update(
                best_aesthetic_asset_id=aesthetic_asset_id,
                best_aesthetic_score=aesthetic_score,
                best_compliant_asset_id=compliant_asset_id,
                best_compliant_score=compliant_score,
                best_asset_id=compliant_asset_id or aesthetic_asset_id,
                best_score=compliant_score if compliant_asset_id else aesthetic_score,
                no_improvement_count=no_improvement_count,
            )
            if observation.status == ObservationStatus.SUCCESS:
                updates["phase"] = "finish"
            elif no_improvement_count >= MAX_NO_IMPROVEMENT:
                updates["phase"] = "plateau"
                updates["terminal_reason"] = "quality_plateau"
            else:
                updates["phase"] = "replan"
        return updates

    return execute_tool


def ask_user(state: AgentState) -> dict[str, Any]:
    return {
        "waiting_for_user": True,
        "terminal_reason": "需要用户补充明确的修改目标",
        "trace": state["trace"] + [{"event": "waiting_for_user"}],
    }


def finalize(state: AgentState) -> dict[str, Any]:
    decision = state["decision"]
    if decision and decision.type == DecisionType.FINISH:
        reason = "quality_gate_passed"
    elif state["terminal_reason"]:
        reason = state["terminal_reason"]
    else:
        reason = "max_iterations_or_invalid_state"
    return {"terminal_reason": reason}


def route_decision(state: AgentState) -> str:
    decision = state["decision"]
    if decision is None:
        return "abort"
    return {
        DecisionType.CALL_TOOL: "tool",
        DecisionType.ASK_USER: "ask_user",
        DecisionType.REPLAN: "decide",
        DecisionType.FINISH: "finalize",
        DecisionType.ABORT: "finalize",
    }[decision.type]


def build_graph(registry: ToolRegistry | None = None):
    registry = registry or build_default_registry()
    builder = StateGraph(AgentState)
    builder.add_node("understand_goal", understand_goal)
    builder.add_node("decide", decide)
    builder.add_node("execute_tool", make_execute_tool(registry))
    builder.add_node("ask_user", ask_user)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "understand_goal")
    builder.add_edge("understand_goal", "decide")
    builder.add_conditional_edges(
        "decide",
        route_decision,
        {
            "tool": "execute_tool",
            "ask_user": "ask_user",
            "decide": "decide",
            "finalize": "finalize",
            "abort": "finalize",
        },
    )
    builder.add_edge("execute_tool", "decide")
    builder.add_edge("ask_user", END)
    builder.add_edge("finalize", END)
    return builder.compile()


def run_task(request: TaskRequest, registry: ToolRegistry | None = None) -> FinalResult:
    state = build_graph(registry).invoke(initial_state(request))
    status = "waiting_for_user" if state["waiting_for_user"] else "completed"
    if state["terminal_reason"] in {"max_iterations_or_invalid_state", "quality_plateau"}:
        status = "aborted"
    return FinalResult(
        status=status,
        terminal_reason=state["terminal_reason"] or "unknown",
        goal=state["goal"],
        assets=state["assets"],
        observations=state["observations"],
        trace=state["trace"],
        best_asset_id=state["best_asset_id"],
        best_score=state["best_score"],
        best_aesthetic_asset_id=state["best_aesthetic_asset_id"],
        best_aesthetic_score=state["best_aesthetic_score"],
        best_compliant_asset_id=state["best_compliant_asset_id"],
        best_compliant_score=state["best_compliant_score"],
        marketing_copy=state["marketing_copy"],
        brand_id=request.brand_profile.brand_id if request.brand_profile else None,
    )
