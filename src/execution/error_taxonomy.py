from __future__ import annotations

from src.contracts import ErrorCategory

_MAX_MSG_LEN = 500


def _last_traceback_line(stderr: str) -> str:
    lines = [line.rstrip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1][:_MAX_MSG_LEN]


def classify_error(stderr: str, returncode: int) -> tuple[ErrorCategory, str]:
    """Classify a failed subprocess run into an ErrorCategory and a cleaned message.

    Rules are checked in priority order — the first match wins.
    """
    message = _last_traceback_line(stderr)

    if returncode in (-9, -15):
        return ErrorCategory.TIMEOUT, message
    if "SyntaxError" in stderr:
        return ErrorCategory.SYNTAX_ERROR, message
    if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
        return ErrorCategory.IMPORT_ERROR, message
    if (
        "has no attribute" in stderr
        or "is not callable" in stderr
        or "unexpected keyword" in stderr
    ):
        return ErrorCategory.API_HALLUCINATION, message
    if (
        "shape" in stderr
        or "dimension" in stderr
        or "Could not convert" in stderr
        or "Found input variables with inconsistent" in stderr
    ):
        return ErrorCategory.SHAPE_MISMATCH, message
    if "TypeError" in stderr and (
        "categorical" in stderr
        or "string" in stderr
        or "could not convert string to float" in stderr
    ):
        return ErrorCategory.TYPE_ERROR, message
    if (
        "DeprecationWarning" in stderr
        or "FutureWarning" in stderr
        or "was deprecated" in stderr
    ):
        return ErrorCategory.DEPRECATED_API, message
    if "MemoryError" in stderr or "ResourceWarning" in stderr:
        return ErrorCategory.RESOURCE_LIMIT, message
    return ErrorCategory.RUNTIME_OTHER, message
