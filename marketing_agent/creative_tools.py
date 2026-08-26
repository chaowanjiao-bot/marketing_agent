from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .schemas import Observation, ObservationStatus
from .tools import AgentTool


class TypographyRenderTool(AgentTool):
    """Deterministically composites critical copy after image generation."""

    name = "typography_renderer"

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict

    def execute(self, arguments: dict[str, Any]) -> Observation:
        source = Path(str(arguments["image_path"]))
        layers = list(arguments.get("layers") or [])
        if not source.exists():
            if self.strict:
                raise FileNotFoundError(source)
            return Observation(
                tool_name=self.name, status=ObservationStatus.SUCCESS,
                outputs={"file_path": str(source), "rendered_layers": 0, "mock_passthrough": True},
            )
        from PIL import Image, ImageDraw, ImageFont

        image = Image.open(source).convert("RGBA")
        draw = ImageDraw.Draw(image)
        font_path = str(arguments.get("font_path") or os.environ.get("TYPOGRAPHY_FONT_PATH", ""))
        if not font_path:
            raise ValueError("TYPOGRAPHY_FONT_PATH or font_path is required for deterministic copy")
        for layer in layers:
            text = str(layer.get("text") or "").strip()
            if not text:
                continue
            size = max(12, int(float(layer.get("font_size_ratio", .045)) * image.height))
            font = ImageFont.truetype(font_path, size=size)
            x = int(float(layer.get("x", .08)) * image.width)
            y = int(float(layer.get("y", .08)) * image.height)
            fill = str(layer.get("color", "#111111"))
            stroke = int(layer.get("stroke_width", 0))
            draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke,
                      stroke_fill=str(layer.get("stroke_color", "#FFFFFF")))
        output = Path(str(arguments.get("output_path") or source.with_name(source.stem + "_typeset.png")))
        output.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output, format="PNG")
        return Observation(
            tool_name=self.name, status=ObservationStatus.SUCCESS,
            outputs={"file_path": str(output), "rendered_layers": len(layers)},
        )


class SafeAreaValidatorTool(AgentTool):
    name = "safe_area_validator"

    def execute(self, arguments: dict[str, Any]) -> Observation:
        margin = float(arguments.get("safe_margin", .05))
        issues = []
        for item in arguments.get("components") or []:
            if item.get("kind") == "decoration":
                continue
            x, y = float(item["x"]), float(item["y"])
            right = x + float(item["width"]); bottom = y + float(item["height"])
            if x < margin or y < margin or right > 1 - margin or bottom > 1 - margin:
                issues.append(f"{item['component_id']}:outside_safe_area")
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS if not issues else ObservationStatus.PARTIAL,
            metrics={"safe_area_compliance": 1.0 if not issues else 0.0},
            issues=issues,
            recommended_actions=["Move or resize the named component inside the configured safe margin."],
        )


class BrandRuleValidatorTool(AgentTool):
    name = "brand_rule_engine"

    def execute(self, arguments: dict[str, Any]) -> Observation:
        text = " ".join(str(x) for x in arguments.get("recognized_texts") or [])
        required = [str(x) for x in arguments.get("required_phrases") or []]
        forbidden = [str(x) for x in arguments.get("forbidden_phrases") or []]
        issues = [f"missing_required:{x}" for x in required if x and x not in text]
        issues += [f"forbidden_phrase:{x}" for x in forbidden if x and x in text]
        expected_logo_count = arguments.get("expected_logo_count")
        detected_logo_count = arguments.get("detected_logo_count")
        if expected_logo_count is not None and detected_logo_count is not None and int(expected_logo_count) != int(detected_logo_count):
            issues.append(f"logo_count:{detected_logo_count}!={expected_logo_count}")
        delta_e = arguments.get("primary_color_delta_e")
        max_delta_e = float(arguments.get("max_primary_color_delta_e", 8.0))
        if delta_e is not None and float(delta_e) > max_delta_e:
            issues.append(f"brand_color_delta_e:{float(delta_e):.2f}>{max_delta_e:.2f}")
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS if not issues else ObservationStatus.PARTIAL,
            metrics={"brand_rule_compliance": 1.0 if not issues else 0.0},
            issues=issues,
            recommended_actions=["Repair the deterministic typography layer; do not regenerate compliant imagery."],
        )


class LayoutValidatorTool(AgentTool):
    name = "layout_validator"

    def execute(self, arguments: dict[str, Any]) -> Observation:
        components = list(arguments.get("components") or [])
        ids = [str(x["component_id"]) for x in components]
        duplicates = sorted({x for x in ids if ids.count(x) > 1})
        issues = [f"duplicate_component:{x}" for x in duplicates]
        return Observation(
            tool_name=self.name,
            status=ObservationStatus.SUCCESS if not issues else ObservationStatus.PARTIAL,
            metrics={"layout_structure_valid": 1.0 if not issues else 0.0},
            issues=issues,
        )


def register_creative_tools(registry, *, strict_typography: bool) -> None:
    registry.register(TypographyRenderTool(strict=strict_typography))
    registry.register(SafeAreaValidatorTool())
    registry.register(BrandRuleValidatorTool())
    registry.register(LayoutValidatorTool())
