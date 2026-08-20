from pathlib import Path

from PIL import Image

from marketing_agent.adapters.powerpaint import PowerPaintEditor


class FakeController:
    def predict(self, *args, **kwargs):
        return [Image.new("RGB", (2, 2), "red")], []


def test_powerpaint_composites_only_inside_mask(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (4, 4), "blue").save(image_path)
    mask = Image.new("L", (4, 4), 0)
    for y in range(1, 3):
        for x in range(1, 3):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)

    editor = PowerPaintEditor(tmp_path)
    editor._controller = FakeController()
    result = editor.edit(
        image_path=str(image_path),
        mask_path=str(mask_path),
        prompt="red product",
        seed=7,
    )

    output = Image.open(result["file_path"]).convert("RGB")
    assert output.size == (4, 4)
    assert output.getpixel((0, 0)) == (0, 0, 255)
    assert output.getpixel((1, 1)) == (255, 0, 0)
