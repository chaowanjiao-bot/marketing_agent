from __future__ import annotations

from typing import Any

from ..contracts import AgentRole
from ..schemas import AssetVersion
from ..tools import ToolRegistry
from .base import SpecialistAgent


class GenerationAgent(SpecialistAgent):
    role = AgentRole.GENERATION
    allowed_tools = frozenset({"generate_image", "edit_image", "typography_renderer"})

    def __init__(self, registry: ToolRegistry) -> None:
        super().__init__(None)
        self.registry = registry

    def act(self, campaign) -> dict[str, Any]:
        if not campaign.layouts:
            raise ValueError("generation requires layout")
        layout = self._target_layout(campaign)
        prompt = layout.generation_prompt
        if campaign.human_feedback_route and self.role in campaign.human_feedback_route.targets:
            prompt += "\nHuman revision feedback (must be applied):\n" + campaign.human_feedback_route.feedback
        if campaign.revision_plan:
            actions = [t.action for t in campaign.revision_plan.tasks if t.target_agent == self.role]
            if actions:
                prompt += "\nRevision requirements:\n" + "\n".join(actions)
        is_edit = bool(campaign.request.input_image)
        tool_name = "edit_image" if is_edit else "generate_image"
        seed = campaign.request.seed + campaign.budget.generations * 1009
        observation = self.registry.get(tool_name).execute({
            "prompt": prompt,
            "input_image": campaign.request.input_image,
            "target_expression": campaign.request.target_expression,
            "attempt": campaign.budget.generations + 1,
            "seed": seed,
            "width": layout.width,
            "height": layout.height,
            "output_format": layout.output_format.value,
        })
        rendered_path = str(observation.outputs["file_path"])
        layers = self._typography_layers(campaign, layout)
        if layers:
            rendered = self.registry.get("typography_renderer").execute({
                "image_path": rendered_path,
                "layers": layers,
            })
            rendered_path = str(rendered.outputs["file_path"])
        parent = campaign.assets[-1].asset_id if campaign.assets else None
        asset = AssetVersion(
            parent_id=parent, tool_name=tool_name, file_path=rendered_path,
            prompt=str(observation.outputs.get("prompt", prompt)), seed=int(observation.outputs.get("seed", seed)),
            output_format=layout.output_format, width=layout.width, height=layout.height,
        )
        msg = self.message(campaign, AgentRole.DIRECTOR, {"asset_id": asset.asset_id}, evidence=[asset.file_path])
        return {"assets": [asset], "messages": [msg], "generation_increment": 1, "revision_plan": None}

    @staticmethod
    def _target_layout(campaign):
        if campaign.revision_plan and campaign.assets:
            current_format = campaign.assets[-1].output_format
            return next(x for x in campaign.layouts if x.output_format == current_format)
        generated = {asset.output_format for asset in campaign.assets}
        return next((layout for layout in campaign.layouts if layout.output_format not in generated), campaign.layouts[0])

    @staticmethod
    def _typography_layers(campaign, layout) -> list[dict[str, Any]]:
        brief = campaign.brief
        if brief is None:
            return []
        component = {x.kind: x for x in layout.components}
        values = [
            ("headline", brief.headline),
            ("subheadline", brief.subheadline),
            ("cta", brief.call_to_action),
        ]
        layers = []
        for kind, text in values:
            box = component.get(kind)
            if text and box:
                layers.append({"text": text, "x": box.x, "y": box.y, "font_size_ratio": .05 if kind == "headline" else .035})
        represented = {str(x["text"]) for x in layers}
        for index, text in enumerate(x for x in brief.required_elements if x not in represented):
            layers.append({"text": text, "x": .08, "y": .72 + index * .045, "font_size_ratio": .028})
        return layers
