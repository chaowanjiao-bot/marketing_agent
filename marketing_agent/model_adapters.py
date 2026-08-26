from __future__ import annotations

import json
import os
from typing import TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleStructuredModel:
    """Schema-constrained adapter for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = httpx.Client(timeout=timeout_seconds)

    def invoke(self, *, system: str, user: str, response_model: type[T]) -> T:
        schema = response_model.model_json_schema()
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                },
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        payload = json.loads(content) if isinstance(content, str) else content
        return response_model.model_validate(payload)

    def close(self) -> None:
        self.client.close()


def structured_model_from_env() -> OpenAICompatibleStructuredModel | None:
    base_url = os.environ.get("STRUCTURED_MODEL_BASE_URL", "").strip()
    model = os.environ.get("STRUCTURED_MODEL_NAME", "").strip()
    api_key = os.environ.get("STRUCTURED_MODEL_API_KEY", "").strip()
    if not any((base_url, model, api_key)):
        return None
    if not all((base_url, model, api_key)):
        raise ValueError(
            "STRUCTURED_MODEL_BASE_URL, STRUCTURED_MODEL_NAME and "
            "STRUCTURED_MODEL_API_KEY must be configured together"
        )
    return OpenAICompatibleStructuredModel(base_url=base_url, model=model, api_key=api_key)
