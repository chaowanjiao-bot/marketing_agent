from marketing_agent.executor import classify_error


def test_classify_known_errors() -> None:
    assert classify_error(FileNotFoundError())[0] == "dependency_unavailable"
    assert classify_error(ValueError())[0] == "invalid_input"
    assert classify_error(TimeoutError()) == ("execution_timeout", True)


def test_classify_oom_without_importing_torch() -> None:
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    assert classify_error(OutOfMemoryError()) == ("gpu_out_of_memory", True)
