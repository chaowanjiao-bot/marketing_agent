from pathlib import Path

import pytest

from marketing_agent.asset_store import AssetStore


def test_asset_store_accepts_png_signature(tmp_path: Path) -> None:
    result = AssetStore(tmp_path).save(
        content_type="image/png", data=b"\x89PNG\r\n\x1a\nmock"
    )
    assert Path(str(result["path"])).is_file()
    assert result["size"] == 12


@pytest.mark.parametrize(
    "content_type,data",
    [("text/plain", b"hello"), ("image/png", b"not-png")],
)
def test_asset_store_rejects_unsupported_or_spoofed_files(
    tmp_path: Path, content_type: str, data: bytes
) -> None:
    with pytest.raises(ValueError):
        AssetStore(tmp_path).save(content_type=content_type, data=data)


def test_asset_store_enforces_size_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="size limit"):
        AssetStore(tmp_path, max_bytes=8).save(
            content_type="image/png", data=b"\x89PNG\r\n\x1a\nextra"
        )
