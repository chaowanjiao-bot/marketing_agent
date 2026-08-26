from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from ..schemas import AssetVersion, Observation, OutputFormat, TaskRequest


class AgentRole(str, Enum):
    DIRECTOR = "creative_director"
    BRIEF = "brief_agent"
    BRAND_STRATEGY = "brand_strategy_agent"
    PERFORMANCE_STRATEGY = "performance_strategy_agent"
    MERCHANDISING_STRATEGY = "merchandising_strategy_agent"
    LAYOUT = "layout_agent"
    GENERATION = "generation_agent"
    QUALITY = "quality_critic_agent"
    COMPLIANCE = "compliance_agent"
    HUMAN = "human_reviewer"


class MessageType(str, Enum):
    TASK = "task"
    RESULT = "result"
    CLARIFICATION = "clarification"
    CRITIQUE = "critique"
    REVISION_REQUEST = "revision_request"
    APPROVAL = "approval"


class CampaignStatus(str, Enum):
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    WAITING_FOR_REVIEW = "waiting_for_review"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class RepairOperation(str, Enum):
    MOVE_INSIDE_SAFE_AREA = "move_inside_safe_area"
    RERENDER_TYPOGRAPHY = "rerender_typography"
    ADJUST_VISUAL_HIERARCHY = "adjust_visual_hierarchy"
    REGENERATE_ASSET = "regenerate_asset"
    LOCAL_IMAGE_EDIT = "local_image_edit"


class AgentMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: f"msg_{uuid4().hex[:12]}")
    correlation_id: str
    sender: AgentRole
    receiver: AgentRole
    message_type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CreativeBrief(BaseModel):
    product_name: str | None = None
    campaign_type: str | None = None
    target_audience: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    headline: str | None = None
    subheadline: str | None = None
    call_to_action: str | None = None
    brand_tone: list[str] = Field(default_factory=list)
    brand_colors: list[str] = Field(default_factory=list)
    required_elements: list[str] = Field(default_factory=list)
    forbidden_elements: list[str] = Field(default_factory=list)
    compliance_rules: list[str] = Field(default_factory=list)
    output_formats: list[OutputFormat] = Field(default_factory=lambda: [OutputFormat.SQUARE])
    ambiguities: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.product_name and self.channels and not self.ambiguities)


class CreativeStrategy(BaseModel):
    strategy_id: str = Field(default_factory=lambda: f"strategy_{uuid4().hex[:10]}")
    author: AgentRole
    name: str
    target_insight: str
    core_message: str
    visual_concept: str
    composition: str
    color_and_lighting: str
    call_to_action: str
    expected_strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class LayoutComponent(BaseModel):
    component_id: str
    kind: Literal["product", "headline", "subheadline", "cta", "logo", "legal", "decoration"]
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)
    z_index: int = Field(ge=0)
    locked: bool = False

    @model_validator(mode="after")
    def inside_canvas(self) -> "LayoutComponent":
        if self.x + self.width > 1.0 or self.y + self.height > 1.0:
            raise ValueError(f"component {self.component_id} exceeds canvas")
        return self


class LayoutPlan(BaseModel):
    output_format: OutputFormat
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    safe_margin: float = Field(default=0.05, ge=0.0, le=0.25)
    components: list[LayoutComponent]
    typography_notes: list[str] = Field(default_factory=list)
    generation_prompt: str


class ReviewFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: f"finding_{uuid4().hex[:10]}")
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high", "blocking"]
    evidence: str
    violated_rule: str | None = None
    region: tuple[float, float, float, float] | None = None
    proposed_action: str
    target_agent: AgentRole


class QualityReview(BaseModel):
    asset_id: str
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    findings: list[ReviewFinding] = Field(default_factory=list)


class ComplianceReview(BaseModel):
    asset_id: str
    passed: bool
    hard_veto: bool = False
    recognized_texts: list[str] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)


class SharedEvaluation(BaseModel):
    asset_id: str
    observation: Observation


class RevisionTask(BaseModel):
    target_agent: AgentRole
    action: str
    operation: RepairOperation = RepairOperation.REGENERATE_ASSET
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_finding_ids: list[str] = Field(default_factory=list)
    preserve: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=10)


class RevisionPlan(BaseModel):
    reason: str
    tasks: list[RevisionTask]
    max_additional_generations: int = Field(default=1, ge=0, le=4)


class DirectorDecision(BaseModel):
    next_agents: list[AgentRole]
    execution_mode: Literal["serial", "parallel"] = "serial"
    reason: str
    success_criteria: list[str] = Field(default_factory=list)
    terminal: bool = False
    requested_status: CampaignStatus = CampaignStatus.RUNNING


class DirectorRecord(BaseModel):
    round: int = Field(ge=0)
    decision: DirectorDecision
    state_summary: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConflictResolution(BaseModel):
    conflict_id: str = Field(default_factory=lambda: f"conflict_{uuid4().hex[:10]}")
    asset_id: str
    parties: list[AgentRole]
    issue: str
    priority_basis: str
    resolution: str
    winning_constraint: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HumanFeedbackRoute(BaseModel):
    feedback: str
    targets: list[AgentRole]
    reason: str


class AgentRunMetric(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    agent: AgentRole
    node: str
    attempt: int = Field(default=1, ge=1)
    status: Literal["succeeded", "failed", "degraded"]
    duration_ms: float = Field(default=0.0, ge=0.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    input_units: int = Field(default=0, ge=0)
    output_units: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CampaignCheckpoint(BaseModel):
    checkpoint_id: str = Field(default_factory=lambda: f"checkpoint_{uuid4().hex[:12]}")
    node: str
    director_round: int = Field(ge=0)
    generation_count: int = Field(ge=0)
    asset_ids: list[str] = Field(default_factory=list)
    recoverable: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionBudget(BaseModel):
    max_director_rounds: int = Field(default=32, ge=2, le=80)
    max_generations: int = Field(default=8, ge=1, le=24)
    director_rounds: int = Field(default=0, ge=0)
    generations: int = Field(default=0, ge=0)

    @property
    def exhausted(self) -> bool:
        return self.director_rounds >= self.max_director_rounds or self.generations >= self.max_generations


class MultiAgentCampaign(BaseModel):
    campaign_id: str = Field(default_factory=lambda: f"campaign_{uuid4().hex[:12]}")
    request: TaskRequest
    status: CampaignStatus = CampaignStatus.PLANNING
    brief: CreativeBrief | None = None
    strategies: list[CreativeStrategy] = Field(default_factory=list)
    selected_strategy_id: str | None = None
    layouts: list[LayoutPlan] = Field(default_factory=list)
    assets: list[AssetVersion] = Field(default_factory=list)
    quality_reviews: list[QualityReview] = Field(default_factory=list)
    compliance_reviews: list[ComplianceReview] = Field(default_factory=list)
    shared_evaluations: list[SharedEvaluation] = Field(default_factory=list)
    revision_plan: RevisionPlan | None = None
    messages: list[AgentMessage] = Field(default_factory=list)
    director_records: list[DirectorRecord] = Field(default_factory=list)
    conflict_resolutions: list[ConflictResolution] = Field(default_factory=list)
    human_feedback_route: HumanFeedbackRoute | None = None
    run_metrics: list[AgentRunMetric] = Field(default_factory=list)
    checkpoints: list[CampaignCheckpoint] = Field(default_factory=list)
    budget: ExecutionBudget = Field(default_factory=ExecutionBudget)
    terminal_reason: str | None = None

    def selected_strategy(self) -> CreativeStrategy | None:
        return next((s for s in self.strategies if s.strategy_id == self.selected_strategy_id), None)

    def evaluation_for(self, asset_id: str) -> SharedEvaluation | None:
        return next((x for x in reversed(self.shared_evaluations) if x.asset_id == asset_id), None)
