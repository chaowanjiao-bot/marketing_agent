from __future__ import annotations

import sys
from types import SimpleNamespace

from marketing_agent.adapters.qwen_image import QwenImageGenerator


class _Cuda:
    def __init__(self) -> None:
        self.emptied = False

    @staticmethod
    def is_available() -> bool:
        return True

    def empty_cache(self) -> None:
        self.emptied = True


def test_unload_releases_pipeline_and_cuda_cache(monkeypatch):
    cuda = _Cuda()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    generator = QwenImageGenerator(SimpleNamespace())
    generator._pipeline = object()

    generator.unload()

    assert generator._pipeline is None
    assert cuda.emptied is True


def test_unload_after_generate_is_opt_in(monkeypatch):
    monkeypatch.delenv("QWEN_UNLOAD_AFTER_GENERATE", raising=False)
    assert QwenImageGenerator._unload_after_generate() is False
    monkeypatch.setenv("QWEN_UNLOAD_AFTER_GENERATE", "true")
    assert QwenImageGenerator._unload_after_generate() is True
