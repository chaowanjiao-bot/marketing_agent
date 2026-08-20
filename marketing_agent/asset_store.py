from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4


class AssetStore:
    ALLOWED = {
        "image/png": (".png", b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": (".jpg", b"\xff\xd8\xff"),
        "image/webp": (".webp", b"RIFF"),
    }
    ASSET_ID = re.compile(r"^upload_[0-9a-f]{12}$")

    def __init__(self, root: Path, max_bytes: int = 20 * 1024 * 1024) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, asset_id: str) -> Path:
        if not self.ASSET_ID.fullmatch(asset_id):
            raise ValueError("invalid asset id")
        matches = [path for path in self.root.glob(f"{asset_id}.*") if path.is_file()]
        allowed_suffixes = {config[0] for config in self.ALLOWED.values()}
        if len(matches) != 1 or matches[0].suffix not in allowed_suffixes:
            raise KeyError(asset_id)
        return matches[0].resolve()

    def save(self, *, content_type: str, data: bytes) -> dict[str, str | int]:
        if content_type not in self.ALLOWED:
            raise ValueError("unsupported image type")
        if not data or len(data) > self.max_bytes:
            raise ValueError("image is empty or exceeds size limit")
        suffix, signature = self.ALLOWED[content_type]
        if not data.startswith(signature):
            raise ValueError("image signature does not match content type")
        if content_type == "image/webp" and data[8:12] != b"WEBP":
            raise ValueError("invalid WebP signature")
        asset_id = f"upload_{uuid4().hex[:12]}"
        path = self.root / f"{asset_id}{suffix}"
        path.write_bytes(data)
        return {
            "asset_id": asset_id,
            "path": str(path),
            "size": len(data),
            "content_type": content_type,
        }
