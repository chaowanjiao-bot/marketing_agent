from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from .brief import CreativityLevel
from .brand import BrandProfile
from .copy_agent import MarketingCopy
from .formats import OutputFormat

from pydantic import BaseModel, Field, model_validator


class TaskType(str, Enum):
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_EDIT = "image_edit"
    AMBIGUOUS = "ambiguous"


class DecisionType(str, Enum):
    CALL_TOOL = "call_tool"
    ASK_USER = "ask_user"
    REPLAN = "replan"
    FINISH = "finish"
    ABORT = "abort"


class ObservationStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class ReviewStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    WAITING = "waiting_for_review"
    APPROVED = "approved"
    REVISION_REQUESTED = "revision_requested"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"


class ReviewRecord(BaseModel):
    round: int = Field(ge=0)
    decision: ReviewDecision
    feedback: str = ""
    reviewer: str = "human"
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Constraint(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    hard: bool = True


class GoalSpec(BaseModel):
    business_goal: str = Field(min_length=3)
    task_type: TaskType
    hard_constraints: list[Constraint] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    type: DecisionType
    reason_summary: str = Field(min_length=3)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    success_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tool_call(self) -> "Decision":
        if self.type == DecisionType.CALL_TOOL and not self.tool_name:
            raise ValueError("CALL_TOOL requires tool_name")
        if self.type != DecisionType.CALL_TOOL and self.tool_name is not None:
            raise ValueError("only CALL_TOOL may include tool_name")
        return self


class Observation(BaseModel):
    tool_name: str
    status: ObservationStatus
    outputs: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    cost: float = Field(default=0.0, ge=0.0)


class AssetVersion(BaseModel):
    asset_id: str = Field(default_factory=lambda: f"asset_{uuid4().hex[:10]}")
    parent_id: str | None = None
    tool_name: str
    file_path: str
    prompt: str
    seed: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    output_format: OutputFormat = OutputFormat.SQUARE
    width: int = Field(default=1024, gt=0)
    height: int = Field(default=1024, gt=0)


class BudgetState(BaseModel):
    max_iterations: int = Field(default=4, ge=1, le=20)
    spent: float = Field(default=0.0, ge=0.0)
    max_cost: float = Field(default=1.0, gt=0.0)

    def charge(self, amount: float) -> "BudgetState":
        if amount < 0:
            raise ValueError("charge must be non-negative")
        if self.spent + amount > self.max_cost:
            raise ValueError("budget exceeded")
        return self.model_copy(update={"spent": self.spent + amount})


class TaskRequest(BaseModel):
    prompt: str = Field(min_length=3)
    input_image: str | None = None
    target_expression: str | None = None
    creativity: CreativityLevel = CreativityLevel.MEDIUM
    max_iterations: int = Field(default=4, ge=1, le=20)
    input_asset_id: str | None = None
    use_memory: bool | None = None
    memory_top_k: int = Field(default=3, ge=1, le=10)
    memory_context: str = Field(default="", exclude=True)
    brand_profile: BrandProfile | None = None
    generate_copy: bool = True
    candidate_count: int = Field(default=1, ge=1, le=8)
    parallel_candidates: bool = False
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    output_formats: list[OutputFormat] = Field(
        default_factory=lambda: [OutputFormat.SQUARE], min_length=1, max_length=4
    )
    review_required: bool = False
    review_round: int = Field(default=0, ge=0, le=20)
    max_review_rounds: int = Field(default=3, ge=1, le=20)
    review_feedback: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def validate_output_formats(self) -> "TaskRequest":
        if len(set(self.output_formats)) != len(self.output_formats):
            raise ValueError("output_formats must be unique")
        if (self.input_image or self.input_asset_id) and self.output_formats != [OutputFormat.SQUARE]:
            raise ValueError("format variants currently support text-to-image tasks only")
        return self


class FinalResult(BaseModel):
    status: str
    terminal_reason: str
    goal: GoalSpec | None
    assets: list[AssetVersion]
    observations: list[Observation]
    trace: list[dict[str, Any]]
    best_asset_id: str | None = None
    best_score: float | None = None
    best_aesthetic_asset_id: str | None = None
    best_aesthetic_score: float | None = None
    best_compliant_asset_id: str | None = None
    best_compliant_score: float | None = None
    memory_used: bool = False
    retrieved_case_ids: list[str] = Field(default_factory=list)
    saved_case_id: str | None = None
    marketing_copy: MarketingCopy | None = None
    brand_id: str | None = None
    candidate_summaries: list["CandidateSummary"] = Field(default_factory=list)
    selected_candidate_index: int = 0
    format_summaries: list["FormatSummary"] = Field(default_factory=list)
    primary_output_format: OutputFormat = OutputFormat.SQUARE
    review_status: ReviewStatus = ReviewStatus.NOT_REQUIRED
    review_round: int = 0
    review_history: list[ReviewRecord] = Field(default_factory=list)


class CandidateSummary(BaseModel):
    candidate_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    status: str
    terminal_reason: str
    best_asset_id: str | None = None
    best_score: float | None = None
    compliant: bool = False
    asset_count: int = Field(ge=0)
    selected: bool = False
    output_format: OutputFormat = OutputFormat.SQUARE


class FormatSummary(BaseModel):
    output_format: OutputFormat
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    best_asset_id: str | None = None
    best_score: float | None = None
    compliant: bool = False
    selected_candidate_index: int = Field(default=0, ge=0)
