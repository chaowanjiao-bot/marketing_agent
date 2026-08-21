import time
from concurrent.futures import ThreadPoolExecutor

from marketing_agent.gpu_scheduler import GpuScheduler, ScheduledTool
from marketing_agent.schemas import Observation, ObservationStatus
from marketing_agent.tools import AgentTool


class SlowGpuTool(AgentTool):
    name = "gpu_test"

    def execute(self, arguments):
        time.sleep(0.04)
        return Observation(tool_name=self.name, status=ObservationStatus.SUCCESS)


def test_scheduler_serializes_gpu_work_and_reports_queue_time() -> None:
    scheduler = GpuScheduler(max_concurrent=1)
    tool = ScheduledTool(SlowGpuTool(), scheduler)
    with ThreadPoolExecutor(max_workers=3) as pool:
        observations = list(pool.map(lambda _: tool.execute({}), range(3)))
    snapshot = scheduler.snapshot()
    assert snapshot["peak_active"] == 1
    assert snapshot["completed"] == {"gpu_test": 3}
    assert sum(item.metrics["gpu_queue_seconds"] > 0.02 for item in observations) >= 2


def test_scheduler_rejects_invalid_capacity() -> None:
    try:
        GpuScheduler(max_concurrent=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
