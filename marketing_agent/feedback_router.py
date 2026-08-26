from __future__ import annotations

from .contracts import AgentMessage, AgentRole, HumanFeedbackRoute, MessageType


def route_human_feedback(feedback: str) -> HumanFeedbackRoute | None:
    normalized = " ".join(feedback.split())
    if not normalized:
        return None
    targets: list[AgentRole] = []
    rules = [
        (("构图", "布局", "位置", "上移", "下移", "放大", "缩小", "字号", "安全区"), [AgentRole.LAYOUT]),
        (("文案", "标题", "副标题", "价格", "错字", "logo", "Logo", "品牌"), [AgentRole.COMPLIANCE, AgentRole.LAYOUT]),
        (("风格", "创意", "卖点", "受众", "调性", "策略"), [AgentRole.BRAND_STRATEGY, AgentRole.PERFORMANCE_STRATEGY, AgentRole.MERCHANDISING_STRATEGY]),
        (("产品", "背景", "光照", "材质", "替换", "重生成", "场景"), [AgentRole.GENERATION]),
    ]
    reasons = []
    for keywords, roles in rules:
        matched = [keyword for keyword in keywords if keyword in normalized]
        if matched:
            targets.extend(roles); reasons.append("/".join(matched))
    if not targets:
        targets = [AgentRole.DIRECTOR, AgentRole.GENERATION]
        reasons.append("unclassified creative feedback")
    return HumanFeedbackRoute(
        feedback=normalized,
        targets=list(dict.fromkeys(targets)),
        reason="matched feedback dimensions: " + ", ".join(reasons),
    )


def feedback_messages(campaign_id: str, route: HumanFeedbackRoute) -> list[AgentMessage]:
    return [AgentMessage(
        correlation_id=campaign_id,
        sender=AgentRole.HUMAN,
        receiver=target,
        message_type=MessageType.REVISION_REQUEST,
        payload={"feedback": route.feedback, "routing_reason": route.reason},
        evidence=[route.feedback],
        confidence=1.0,
    ) for target in route.targets]
