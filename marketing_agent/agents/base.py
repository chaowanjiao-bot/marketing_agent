from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from ..contracts import AgentMessage, AgentRole, MultiAgentCampaign

T = TypeVar("T", bound=BaseModel)


class StructuredModel(Protocol):
    """Provider-neutral structured LLM/MLLM boundary."""

    def invoke(self, *, system: str, user: str, response_model: type[T]) -> T: ...


class SpecialistAgent(ABC):
    role: AgentRole
    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, model: StructuredModel | None = None) -> None:
        self.model = model

    @abstractmethod
    def act(self, campaign: MultiAgentCampaign) -> dict[str, Any]:
        raise NotImplementedError

    def message(
        self,
        campaign: MultiAgentCampaign,
        receiver: AgentRole,
        payload: dict[str, Any],
        *,
        evidence: list[str] | None = None,
        confidence: float = 1.0,
    ) -> AgentMessage:
        from ..contracts import MessageType

        return AgentMessage(
            correlation_id=campaign.campaign_id,
            sender=self.role,
            receiver=receiver,
            message_type=MessageType.RESULT,
            payload=payload,
            evidence=evidence or [],
            confidence=confidence,
        )
