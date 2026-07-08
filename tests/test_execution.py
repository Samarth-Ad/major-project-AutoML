from __future__ import annotations

from src.contracts import ErrorCategory, GeneratedPipeline, TaskType
from src.execution.error_taxonomy import classify_error
from src.execution.metrics import extract_score
from src.execution.runner import execute_pipeline
from src.experiments.runner import build_error_feedback


def _pipeline(code: str) -> GeneratedPipeline:
    return GeneratedPipeline(
        code=code,
        condition="B0",
        llm_backend="stub",
        dataset_id="test",
        seed=0,
        iteration=0,
    )


def test_extract_score_valid() -> None:
    assert extract_score("Some output\nSCORE: 0.847\nDone", TaskType.BINARY_CLASSIFICATION) == 0.847


def test_extract_score_missing() -> None:
    assert extract_score("Some output\nNo score here", TaskType.BINARY_CLASSIFICATION) is None


def test_extract_score_regression() -> None:
    assert extract_score("SCORE: -2.345", TaskType.REGRESSION) == -2.345


def test_classify_syntax_error() -> None:
    stderr = 'File "test.py", line 5\n    print(\nSyntaxError: unexpected EOF'
    category, _ = classify_error(stderr, 1)
    assert category == ErrorCategory.SYNTAX_ERROR


def test_classify_import_error() -> None:
    stderr = "ModuleNotFoundError: No module named 'nonexistent'"
    category, _ = classify_error(stderr, 1)
    assert category == ErrorCategory.IMPORT_ERROR


def test_classify_shape_mismatch() -> None:
    stderr = "ValueError: Found input variables with inconsistent numbers of samples"
    category, _ = classify_error(stderr, 1)
    assert category == ErrorCategory.SHAPE_MISMATCH


def test_classify_timeout() -> None:
    category, _ = classify_error("", -9)
    assert category == ErrorCategory.TIMEOUT


def test_execute_success() -> None:
    result = execute_pipeline(
        _pipeline('print("SCORE: 0.95")'),
        TaskType.BINARY_CLASSIFICATION,
        timeout_seconds=10,
    )
    assert result.success is True
    assert result.test_score == 0.95
    assert result.error_category is None


def test_execute_syntax_error() -> None:
    result = execute_pipeline(
        _pipeline("def f(\n"),
        TaskType.BINARY_CLASSIFICATION,
        timeout_seconds=10,
    )
    assert result.success is False
    assert result.error_category == "syntax_error"


def test_execute_timeout() -> None:
    result = execute_pipeline(
        _pipeline("import time; time.sleep(30)"),
        TaskType.BINARY_CLASSIFICATION,
        timeout_seconds=2,
    )
    assert result.success is False
    assert result.error_category == "timeout"
    assert result.runtime_seconds < 5


def test_classify_missing_name() -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "test.py", line 10, in <module>\n'
        "    scaler = RobustScaler()\n"
        "NameError: name 'RobustScaler' is not defined"
    )
    category, message = classify_error(stderr, 1)
    assert category == ErrorCategory.MISSING_NAME
    assert "RobustScaler" in message


def test_execute_missing_name() -> None:
    result = execute_pipeline(
        _pipeline("x = RobustScaler()"),
        TaskType.BINARY_CLASSIFICATION,
        timeout_seconds=10,
    )
    assert result.success is False
    assert result.error_category == "missing_name"


def test_build_error_feedback_basic() -> None:
    feedback = build_error_feedback("some error happened")
    assert "some error happened" in feedback
    assert "self-contained" in feedback


def test_build_error_feedback_includes_code() -> None:
    feedback = build_error_feedback("err", prev_code="import os\nprint('hi')")
    assert "import os" in feedback
    assert "code that failed" in feedback


def test_build_error_feedback_missing_name_hint() -> None:
    feedback = build_error_feedback(
        "NameError: name 'RobustScaler' is not defined",
        error_category="missing_name",
    )
    assert "import statement" in feedback
    assert "RobustScaler" in feedback
