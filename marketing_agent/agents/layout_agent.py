from __future__ import annotations

from typing import Any

from ..contracts import AgentRole, LayoutPlan, RepairOperation
from ..contracts.multi_agent import LayoutComponent
from ..formats import get_format_spec
from ..tools import ToolRegistry
from .base import SpecialistAgent


class LayoutAgent(SpecialistAgent):
    role = AgentRole.LAYOUT
    allowed_tools = frozenset({"safe_area_validator", "layout_validator", "typography_renderer"})

    def __init__(self, model=None, registry: ToolRegistry | None = None) -> None:
        super().__init__(model); self.registry = registry

    def act(self, campaign) -> dict[str, Any]:
        strategy = campaign.selected_strategy()
        if strategy is None or campaign.brief is None:
            raise ValueError("layout requires brief and selected strategy")
        layouts = []
        for output_format in campaign.brief.output_formats:
            spec = get_format_spec(output_format)
            components = [
                LayoutComponent(component_id="headline", kind="headline", x=.08, y=.06, width=.84, height=.12, z_index=4),
                LayoutComponent(component_id="product", kind="product", x=.22, y=.22, width=.56, height=.55, z_index=2, locked=True),
                LayoutComponent(component_id="cta", kind="cta", x=.30, y=.82, width=.40, height=.08, z_index=4),
                LayoutComponent(component_id="logo", kind="logo", x=.38, y=.90, width=.24, height=.04, z_index=5),
            ]
            components, revision_notes = self._apply_revisions(campaign, components, .05)
            prompt = "\n".join([
                strategy.visual_concept, strategy.composition, strategy.color_and_lighting,
                "Generate background and product scene only; reserve clean areas for deterministic typography.",
            ])
            layouts.append(LayoutPlan(
                output_format=output_format, width=spec.width, height=spec.height,
                components=components,
                typography_notes=[
                    "Render critical copy as a separate vector/text layer",
                    "Respect 5% safe margin",
                    *revision_notes,
                ],
                generation_prompt=prompt,
            ))
            if self.registry is not None:
                payload = [component.model_dump() for component in components]
                structure = self.registry.get("layout_validator").execute({"components": payload})
                safe_area = self.registry.get("safe_area_validator").execute({
                    "components": payload, "safe_margin": .05,
                })
                if structure.issues or safe_area.issues:
                    raise ValueError("layout validation failed: " + ", ".join(structure.issues + safe_area.issues))
        msg = self.message(campaign, AgentRole.DIRECTOR, {"layout_count": len(layouts)})
        return {"layouts": layouts, "messages": [msg]}

    @staticmethod
    def _apply_revisions(campaign, components, safe_margin):
        notes: list[str] = []
        if campaign.revision_plan is None:
            if campaign.human_feedback_route and AgentRole.LAYOUT in campaign.human_feedback_route.targets:
                notes.append("Human layout feedback: " + campaign.human_feedback_route.feedback)
            return components, notes
        updated = list(components)
        for task in campaign.revision_plan.tasks:
            if task.target_agent != AgentRole.LAYOUT:
                continue
            if task.operation == RepairOperation.MOVE_INSIDE_SAFE_AREA:
                target_id = str(task.parameters.get("component_id") or "")
                repaired = []
                for item in updated:
                    if target_id and item.component_id != target_id:
                        repaired.append(item); continue
                    x = min(max(item.x, safe_margin), 1 - safe_margin - item.width)
                    y = min(max(item.y, safe_margin), 1 - safe_margin - item.height)
                    repaired.append(item.model_copy(update={"x": x, "y": y}))
                updated = repaired
                notes.append(f"Applied safe-area repair to {target_id or 'all components'}")
            elif task.operation == RepairOperation.ADJUST_VISUAL_HIERARCHY:
                repaired = []
                for item in updated:
                    if item.kind == "product":
                        repaired.append(item.model_copy(update={"x": .20, "width": .60}))
                    elif item.kind == "headline":
                        repaired.append(item.model_copy(update={"height": .10}))
                    else:
                        repaired.append(item)
                updated = repaired
                notes.append("Adjusted product prominence and headline hierarchy")
            elif task.operation == RepairOperation.RERENDER_TYPOGRAPHY:
                notes.append("Force re-render of critical copy from the validated brief")
        if campaign.human_feedback_route and AgentRole.LAYOUT in campaign.human_feedback_route.targets:
            notes.append("Human layout feedback: " + campaign.human_feedback_route.feedback)
        return updated, notes
