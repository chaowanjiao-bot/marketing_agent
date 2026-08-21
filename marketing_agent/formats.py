from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class OutputFormat(str, Enum):
    SQUARE = "1:1"
    PORTRAIT = "4:5"
    STORY = "9:16"
    LANDSCAPE = "16:9"


class FormatSpec(BaseModel):
    name: OutputFormat
    width: int
    height: int
    placement_rule: str

    def prompt_context(self) -> str:
        return (
            f"输出画幅：{self.name.value}，{self.width}x{self.height}。"
            f"版式规则：{self.placement_rule}"
        )


FORMAT_SPECS = {
    OutputFormat.SQUARE: FormatSpec(
        name=OutputFormat.SQUARE, width=1024, height=1024,
        placement_rule="主体居中，标题位于顶部安全区，四周保留8%安全边距",
    ),
    OutputFormat.PORTRAIT: FormatSpec(
        name=OutputFormat.PORTRAIT, width=1024, height=1280,
        placement_rule="主体位于中下部，顶部保留标题区，底部保留品牌和CTA安全区",
    ),
    OutputFormat.STORY: FormatSpec(
        name=OutputFormat.STORY, width=768, height=1360,
        placement_rule="核心主体位于中央60%区域，顶部和底部各保留15%界面遮挡安全区",
    ),
    OutputFormat.LANDSCAPE: FormatSpec(
        name=OutputFormat.LANDSCAPE, width=1360, height=768,
        placement_rule="主体与文案采用左右布局，核心信息不得贴近左右边缘",
    ),
}


def get_format_spec(output_format: OutputFormat) -> FormatSpec:
    return FORMAT_SPECS[output_format]
