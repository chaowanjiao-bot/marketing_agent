from marketing_agent import model_worker
from marketing_agent.adapters import powerpaint


def test_worker_runtime_cache_reuses_adapter_instance(monkeypatch) -> None:
    instances = []

    class FakeEditor:
        def __init__(self, root):
            self.calls = 0
            instances.append(self)

        def edit(self, **arguments):
            self.calls += 1
            return {"value": arguments["value"], "calls": self.calls}

    monkeypatch.setattr(powerpaint, "PowerPaintEditor", FakeEditor)
    runtimes = {}
    first = model_worker.execute({
        "action": "powerpaint_edit", "arguments": {"value": 1},
    }, runtimes)
    second = model_worker.execute({
        "action": "powerpaint_edit", "arguments": {"value": 2},
    }, runtimes)
    assert len(instances) == 1
    assert first["result"]["calls"] == 1
    assert second["result"]["calls"] == 2
