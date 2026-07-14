"""Run an LLM-generated script and dump its fitted estimator to a .joblib path.

Usage: python _save_wrapper.py <script_path> <output_joblib_path>

Primary strategy: after runpy.run_path, look up the estimator by name in the
returned globals. If nothing matches (usually because the script fits inside
a function), fall back to a fit-hook that captured the outermost fitted
estimator with a .predict method.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils import all_estimators

# Spec defaults, then names observed in the 701-script survey (Phase-2 grounding).
SEARCH_NAMES: tuple[str, ...] = (
    "pipeline", "pipe", "model", "clf", "estimator", "final_model",
    "clf_pipeline", "model_pipeline", "full_pipeline",
    "clf_pipe", "model_pipe", "full_pipe",
    "grid_search",
)

_state: dict[str, Any] = {"depth": 0, "last_predictor": None}


def _make_traced(orig):
    def traced(self, *args, **kwargs):
        _state["depth"] += 1
        try:
            return orig(self, *args, **kwargs)
        finally:
            _state["depth"] -= 1
            if _state["depth"] == 0 and hasattr(self, "predict"):
                _state["last_predictor"] = self
    return traced


def _install_hooks() -> list[tuple[type, Any]]:
    classes: list[type] = [c for _, c in all_estimators() if hasattr(c, "fit")]
    classes.append(Pipeline)
    originals: list[tuple[type, Any]] = []
    seen: set[type] = set()
    for cls in classes:
        if cls in seen:
            continue
        seen.add(cls)
        orig = cls.fit
        originals.append((cls, orig))
        cls.fit = _make_traced(orig)  # type: ignore[method-assign]
    return originals


def _restore_hooks(originals: list[tuple[type, Any]]) -> None:
    for cls, orig in originals:
        cls.fit = orig  # type: ignore[method-assign]


def _pick_by_name(ns: dict[str, Any]) -> Any | None:
    for name in SEARCH_NAMES:
        obj = ns.get(name)
        if obj is not None and hasattr(obj, "predict"):
            return obj
    return None


def main() -> int:
    script_path, out_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    originals = _install_hooks()
    try:
        ns = runpy.run_path(str(script_path), run_name="__main__")
    finally:
        _restore_hooks(originals)

    est = _pick_by_name(ns) or _state["last_predictor"]
    if est is None:
        print("ERROR: no fitted estimator with .predict found", file=sys.stderr)
        return 1
    joblib.dump(est, out_path)
    print(f"SAVED: {type(est).__name__} -> {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
