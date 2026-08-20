"""Marketing creative agent MVP."""


__all__ = ["build_graph", "run_task"]

def __getattr__(name: str):
    if name in __all__:
        from .graph import build_graph, run_task

        return {"build_graph": build_graph, "run_task": run_task}[name]
    raise AttributeError(name)
