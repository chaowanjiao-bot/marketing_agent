from marketing_agent.contracts import AgentRole
from marketing_agent.feedback_router import feedback_messages, route_human_feedback


def test_layout_feedback_routes_to_layout_agent():
    route = route_human_feedback("产品放大到画面高度55%，标题上移")
    assert route is not None
    assert AgentRole.LAYOUT in route.targets


def test_copy_feedback_routes_to_compliance_and_layout():
    route = route_human_feedback("标题有错字，并且Logo只保留一个")
    assert route is not None
    assert {AgentRole.COMPLIANCE, AgentRole.LAYOUT}.issubset(set(route.targets))
    messages = feedback_messages("campaign_123", route)
    assert {message.receiver for message in messages} == set(route.targets)
