from __future__ import annotations

import math
import re
from typing import Optional

from src.contracts import TaskType

_SCORE_RE = re.compile(r"SCORE:\s*([-\d.]+(?:e[-+]?\d+)?)")


def extract_score(stdout: str, task_type: TaskType) -> Optional[float]:
    """Extract the final SCORE printed by a generated pipeline.

    Returns None if no SCORE line is found, the value is unparseable, or it
    fails a sanity check (classification: [0, 1]; regression: finite).
    """
    match = _SCORE_RE.search(stdout)
    if match is None:
        return None
    try:
        score = float(match.group(1))
    except ValueError:
        return None

    if task_type == TaskType.REGRESSION:
        return score if math.isfinite(score) else None
    if 0.0 <= score <= 1.0:
        return score
    return None
