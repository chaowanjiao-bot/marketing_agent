import pytest
from pydantic import ValidationError

from marketing_agent.contracts import AgentMessage, AgentRole, MessageType
from marketing_agent.contracts.multi_agent import LayoutComponent


def test_agent_message_carries_correlation_and_evidence():
    message = AgentMessage(
        correlation_id="campaign_123456789abc",
        sender=AgentRole.QUALITY,
        receiver=AgentRole.DIRECTOR,
        message_type=MessageType.CRITIQUE,
        payload={"passed": False},
        evidence=["composition=0.42"],
        confidence=0.9,
    )
    assert message.receiver == AgentRole.DIRECTOR
    assert message.evidence == ["composition=0.42"]


def test_layout_component_must_remain_inside_canvas():
    with pytest.raises(ValidationError):
        LayoutComponent(
            component_id="logo", kind="logo", x=.9, y=.9,
            width=.2, height=.1, z_index=1,
        )
