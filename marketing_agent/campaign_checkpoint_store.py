from __future__ import annotations

import json
import os
from pathlib import Path

from .contracts import MultiAgentCampaign


class CampaignCheckpointStore:
    """Atomic full-state snapshots used to resume a campaign after worker failure."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, campaign: MultiAgentCampaign) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{campaign.campaign_id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(campaign.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)
        return target

    def load(self, campaign_id: str) -> MultiAgentCampaign:
        if not campaign_id.startswith("campaign_") or not campaign_id.replace("_", "").isalnum():
            raise ValueError("invalid campaign_id")
        path = self.root / f"{campaign_id}.json"
        return MultiAgentCampaign.model_validate_json(path.read_text(encoding="utf-8"))

    def exists(self, campaign_id: str) -> bool:
        return (self.root / f"{campaign_id}.json").is_file()
