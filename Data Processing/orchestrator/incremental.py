"""
orchestrator/incremental.py
-----------------------------
Incremental Pipeline Modification & Partial Re-Execution

Detects what changed between pipeline runs and re-executes ONLY
the affected steps, reusing cached results for unchanged steps.

Architecture
------------
    IncrementalRunner keeps a snapshot of the last successful run:
      - data fingerprint (hash of input CSV)
      - step list with their configs/thresholds
      - cached outputs per step

    When the user modifies the pipeline (change model, add SMOTE,
    change thresholds), the runner:
      1. Diffs old vs new pipeline config
      2. Identifies the FIRST changed step (invalidation point)
      3. Re-uses all cached outputs BEFORE that point
      4. Re-executes from the invalidation point onward

    This avoids redundant LLM calls for unchanged preprocessing steps.

Example
-------
    runner = IncrementalRunner(csv_path="data/train.csv")

    # First run — executes everything
    result1 = runner.run()

    # Change model — only re-executes training + evaluation
    result2 = runner.modify(changes={"models_to_try": ["xgboost", "lightgbm"]})

    # Add SMOTE — re-executes from handle_class_imbalance onward
    result3 = runner.modify(changes={"add_step": "handle_class_imbalance"})
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from agents.base_agent import DynamicAgent, _inspect_dataframe
from agents.data_understanding_agent import (
    PipelineDecision, _analyse_data, _load_thresholds,
    _rule_based_decision, _load_target_column,
)
from builder.agent_builder import create_builder
from utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Change types
# ---------------------------------------------------------------------------

@dataclass
class PipelineChange:
    """Describes a single modification to the pipeline."""
    change_type: str          # "add_step", "remove_step", "change_model",
                              # "change_threshold", "change_target", "reorder"
    description: str          # human-readable summary
    affected_step: str        # first step that's invalidated
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.change_type}] {self.description} (invalidates from: {self.affected_step})"


@dataclass
class IncrementalResult:
    """Result of an incremental pipeline run."""
    success: bool
    pipeline_id: str
    changes: List[PipelineChange]
    steps_reused: List[str]          # cached from previous run
    steps_reexecuted: List[str]      # freshly executed
    steps_skipped: List[str]         # skipped by decision
    final_data: Any
    elapsed_s: float
    decision: Optional[PipelineDecision] = None
    comparison: Optional[Dict[str, Any]] = None  # old vs new metrics

    def summary(self) -> str:
        return (
            f"IncrementalResult(success={self.success}, "
            f"reused={len(self.steps_reused)}, "
            f"reexecuted={len(self.steps_reexecuted)}, "
            f"elapsed={self.elapsed_s:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Step Cache — stores outputs per step
# ---------------------------------------------------------------------------

class _StepCache:
    """In-memory cache of step outputs keyed by (step_name, data_hash)."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._data_snapshots: Dict[str, str] = {}  # step → data hash

    def store(self, step_name: str, data_hash: str, result: Dict[str, Any]) -> None:
        self._cache[step_name] = result
        self._data_snapshots[step_name] = data_hash

    def get(self, step_name: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(step_name)

    def get_data_hash(self, step_name: str) -> str:
        return self._data_snapshots.get(step_name, "")

    def invalidate_from(self, step_name: str, all_steps: List[str]) -> List[str]:
        """Invalidate this step and all steps after it."""
        invalidated = []
        found = False
        for s in all_steps:
            if s == step_name:
                found = True
            if found:
                self._cache.pop(s, None)
                self._data_snapshots.pop(s, None)
                invalidated.append(s)
        return invalidated

    def has(self, step_name: str) -> bool:
        return step_name in self._cache

    def clear(self) -> None:
        self._cache.clear()
        self._data_snapshots.clear()

    @property
    def cached_steps(self) -> List[str]:
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Data fingerprinting
# ---------------------------------------------------------------------------

def _hash_dataframe(df: pd.DataFrame) -> str:
    """Compute a fast hash of a DataFrame for change detection."""
    try:
        # Hash shape + dtypes + first/last few rows for speed
        sig = f"{df.shape}|{list(df.dtypes)}|{df.head(3).to_json()}|{df.tail(2).to_json()}"
        return hashlib.md5(sig.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return hashlib.md5(str(df.shape).encode()).hexdigest()[:12]


def _hash_file(filepath: str) -> str:
    """Hash a file's contents for change detection."""
    try:
        with open(filepath, "rb") as f:
            # Read first and last 8KB for speed on large files
            head = f.read(8192)
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            tail = f.read(8192)
        return hashlib.md5(head + tail + str(size).encode()).hexdigest()[:12]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Change Detector
# ---------------------------------------------------------------------------

class _ChangeDetector:
    """Compares two pipeline configurations and identifies changes."""

    # Canonical step order — used to determine invalidation point
    _STEP_ORDER = [
        "load_dataset", "understand_data",
        "remove_missing_values", "handle_class_imbalance",
        "encode_categorical", "handle_skewness",
        "normalize_features", "feature_engineering",
        "dimensionality_reduction",
        "select_and_train_models", "evaluate_models",
        "select_best_model", "explain_model",
    ]

    @classmethod
    def detect_changes(
        cls,
        old_decision: Optional[PipelineDecision],
        new_decision: PipelineDecision,
        modifications: Dict[str, Any],
    ) -> List[PipelineChange]:
        """
        Compare old and new pipeline decisions + explicit modifications.

        Parameters
        ----------
        old_decision : PipelineDecision or None
            Previous run's decision (None if first run).
        new_decision : PipelineDecision
            Current run's decision.
        modifications : dict
            Explicit user modifications like:
                {"models_to_try": ["xgboost"]}
                {"add_step": "handle_class_imbalance"}
                {"remove_step": "normalize_features"}
                {"thresholds": {"imbalance_ratio_to_trigger_smote": 0.3}}

        Returns
        -------
        list[PipelineChange]
        """
        changes: List[PipelineChange] = []

        # Handle explicit modifications
        if "add_step" in modifications:
            step = modifications["add_step"]
            changes.append(PipelineChange(
                change_type="add_step",
                description=f"Added step '{step}' to pipeline",
                affected_step=step,
                details={"step": step},
            ))

        if "remove_step" in modifications:
            step = modifications["remove_step"]
            # Find next step after the removed one
            after = cls._next_step_after(step, new_decision.steps)
            changes.append(PipelineChange(
                change_type="remove_step",
                description=f"Removed step '{step}' from pipeline",
                affected_step=after or "select_and_train_models",
                details={"step": step},
            ))

        if "models_to_try" in modifications:
            old_models = old_decision.models_to_try if old_decision else []
            new_models = modifications["models_to_try"]
            if set(old_models) != set(new_models):
                changes.append(PipelineChange(
                    change_type="change_model",
                    description=(
                        f"Models changed: {old_models} → {new_models}"
                    ),
                    affected_step="select_and_train_models",
                    details={"old": old_models, "new": new_models},
                ))

        if "target_column" in modifications:
            changes.append(PipelineChange(
                change_type="change_target",
                description=f"Target column changed to '{modifications['target_column']}'",
                affected_step="remove_missing_values",  # invalidates everything
                details={"target": modifications["target_column"]},
            ))

        if "thresholds" in modifications:
            for key, val in modifications["thresholds"].items():
                # Map threshold to affected step
                step_map = {
                    "null_pct_to_trigger_imputation": "remove_missing_values",
                    "imbalance_ratio_to_trigger_smote": "handle_class_imbalance",
                    "skewness_to_trigger_correction": "handle_skewness",
                    "features_to_trigger_pca": "dimensionality_reduction",
                    "rows_to_trigger_tuning": "select_and_train_models",
                }
                affected = step_map.get(key, "select_and_train_models")
                changes.append(PipelineChange(
                    change_type="change_threshold",
                    description=f"Threshold '{key}' changed to {val}",
                    affected_step=affected,
                    details={"threshold": key, "value": val},
                ))

        # Auto-detect step list changes (even without explicit modifications)
        if old_decision and not changes:
            old_steps = set(old_decision.steps)
            new_steps = set(new_decision.steps)
            added = new_steps - old_steps
            removed = old_steps - new_steps

            for s in added:
                changes.append(PipelineChange(
                    change_type="add_step",
                    description=f"Step '{s}' added by data analysis",
                    affected_step=s,
                ))
            for s in removed:
                after = cls._next_step_after(s, new_decision.steps)
                changes.append(PipelineChange(
                    change_type="remove_step",
                    description=f"Step '{s}' removed by data analysis",
                    affected_step=after or "select_and_train_models",
                ))

        return changes

    @classmethod
    def find_invalidation_point(
        cls,
        changes: List[PipelineChange],
        all_steps: List[str],
    ) -> Optional[str]:
        """
        Find the earliest step that needs re-execution.

        Returns
        -------
        str or None
            The step name from which re-execution should start.
            None means no re-execution needed.
        """
        if not changes:
            return None

        earliest_idx = len(all_steps)
        for change in changes:
            step = change.affected_step
            if step in all_steps:
                idx = all_steps.index(step)
                earliest_idx = min(earliest_idx, idx)

        if earliest_idx >= len(all_steps):
            return None
        return all_steps[earliest_idx]

    @classmethod
    def _next_step_after(cls, step: str, steps: List[str]) -> Optional[str]:
        """Find the step that comes after `step` in the canonical order."""
        if step in cls._STEP_ORDER:
            idx = cls._STEP_ORDER.index(step)
            for s in cls._STEP_ORDER[idx + 1:]:
                if s in steps:
                    return s
        return None


# ---------------------------------------------------------------------------
# Incremental Runner
# ---------------------------------------------------------------------------

class IncrementalRunner:
    """
    Runs the adaptive pipeline with incremental modification support.

    First run executes everything. Subsequent runs detect changes
    and only re-execute affected steps.

    Parameters
    ----------
    csv_path : str
        Path to input CSV.
    target_column : str
        Target column (auto-inferred if empty).
    config_path : str
        Path to pipeline.yaml.
    api_key : str
        LLM API key.
    llm_model : str
        LLM model name.
    """

    def __init__(
        self,
        csv_path: str = "",
        target_column: str = "",
        config_path: str = "config/pipeline.yaml",
        api_key: str = "",
        llm_model: str = "",
    ) -> None:
        self.csv_path = csv_path
        self.target_column = target_column
        self.config_path = config_path
        self.api_key = api_key
        self.llm_model = llm_model

        self._logger = PipelineLogger("orchestrator.IncrementalRunner")
        self._cache = _StepCache()
        self._last_decision: Optional[PipelineDecision] = None
        self._last_data_hash: str = ""
        self._run_count: int = 0
        self._last_result: Optional[IncrementalResult] = None

    # ------------------------------------------------------------------
    # Full run (first execution)
    # ------------------------------------------------------------------

    def run(self, csv_path: str = "") -> IncrementalResult:
        """
        Execute the full adaptive pipeline.
        Caches all step outputs for future incremental runs.
        """
        start = time.perf_counter()
        self._run_count += 1
        csv_path = csv_path or self.csv_path

        self._logger.info(f"\n{'='*60}")
        self._logger.info(f"  INCREMENTAL RUNNER — Run #{self._run_count}")
        self._logger.info(f"{'='*60}")

        # Load data
        df = pd.read_csv(csv_path, encoding="utf-8")
        data_hash = _hash_dataframe(df)
        file_hash = _hash_file(csv_path)

        # Check if data changed
        data_changed = data_hash != self._last_data_hash
        if data_changed:
            self._logger.info("Data changed — full re-analysis required")
            self._cache.clear()

        # Analyse data
        thresholds = _load_thresholds(self.config_path)
        profile = _analyse_data(df, thresholds, self.target_column)
        decision_dict = _rule_based_decision(profile, self._logger)

        decision = PipelineDecision(
            problem_type=decision_dict["problem_type"],
            target_column=decision_dict.get("target_column", profile["target_column"]),
            steps=decision_dict["steps"],
            skipped=decision_dict.get("skipped", {}),
            models_to_try=decision_dict.get("models_to_try", []),
            needs_tuning=decision_dict.get("needs_tuning", False),
            n_rows=profile["n_rows"],
            n_cols=profile["n_cols"],
            has_nulls=profile["checks"]["has_nulls"],
            is_imbalanced=profile["is_imbalanced"],
            skewed_columns=profile["skewed_columns"],
            high_corr_pairs=profile["high_correlation_pairs"],
            reasoning=decision_dict.get("reasoning", {}),
        )

        adaptive_steps = [s for s in decision.steps if s != "load_dataset"]
        self._logger.info(f"Steps: {' → '.join(adaptive_steps)}")

        # Execute all steps
        steps_executed, steps_reused = [], []
        current_data = df

        for step in adaptive_steps:
            self._logger.info(f"  Executing: {step}")
            agent = DynamicAgent(
                step_name=step,
                pipeline_steps=adaptive_steps,
                api_key=self.api_key,
                llm_model=self.llm_model,
            )
            try:
                result = agent.execute(current_data)
                if result.get("status") == "success":
                    current_data = result["output_data"]
                    step_hash = _hash_dataframe(current_data) if isinstance(current_data, pd.DataFrame) else ""
                    self._cache.store(step, step_hash, result)
                    steps_executed.append(step)
                else:
                    self._logger.warning(f"  Step {step} failed: {result.get('error', '?')}")
                    steps_executed.append(f"{step} [FAILED]")
            except Exception as e:
                self._logger.error(f"  Step {step} error: {e}")
                steps_executed.append(f"{step} [ERROR]")

        # Save state
        self._last_decision = decision
        self._last_data_hash = data_hash

        elapsed = time.perf_counter() - start
        result = IncrementalResult(
            success=all("[" not in s for s in steps_executed),
            pipeline_id=f"incr-{self._run_count}-{datetime.now(timezone.utc).strftime('%H%M%S')}",
            changes=[],
            steps_reused=steps_reused,
            steps_reexecuted=steps_executed,
            steps_skipped=list(decision.skipped.keys()),
            final_data=current_data,
            elapsed_s=elapsed,
            decision=decision,
        )
        self._last_result = result

        self._logger.info(f"\n  Result: {result.summary()}")
        return result

    # ------------------------------------------------------------------
    # Incremental modification
    # ------------------------------------------------------------------

    def modify(self, changes: Dict[str, Any]) -> IncrementalResult:
        """
        Apply modifications and re-execute only affected steps.

        Parameters
        ----------
        changes : dict
            Modification specification. Supported keys:
                "models_to_try": ["xgboost", "lightgbm"]
                "add_step": "handle_class_imbalance"
                "remove_step": "normalize_features"
                "target_column": "new_target"
                "thresholds": {"imbalance_ratio_to_trigger_smote": 0.3}

        Returns
        -------
        IncrementalResult

        Example
        -------
            runner = IncrementalRunner(csv_path="data/train.csv")
            result1 = runner.run()

            # Change to XGBoost — only re-trains, skips preprocessing
            result2 = runner.modify({"models_to_try": ["xgboost", "lightgbm"]})
        """
        start = time.perf_counter()
        self._run_count += 1

        if self._last_decision is None:
            self._logger.warning("No previous run — executing full pipeline")
            return self.run()

        self._logger.info(f"\n{'='*60}")
        self._logger.info(f"  INCREMENTAL MODIFICATION — Run #{self._run_count}")
        self._logger.info(f"  Changes: {json.dumps(changes, default=str)}")
        self._logger.info(f"{'='*60}")

        # Rebuild decision with modifications
        new_decision = deepcopy(self._last_decision)

        if "models_to_try" in changes:
            new_decision.models_to_try = changes["models_to_try"]
        if "add_step" in changes:
            step = changes["add_step"]
            if step not in new_decision.steps:
                # Insert in canonical order
                new_decision.steps = self._insert_step_ordered(step, new_decision.steps)
                new_decision.skipped.pop(step, None)
        if "remove_step" in changes:
            step = changes["remove_step"]
            new_decision.steps = [s for s in new_decision.steps if s != step]
            new_decision.skipped[step] = "Removed by user modification"
        if "target_column" in changes:
            new_decision.target_column = changes["target_column"]

        # Detect changes
        detected = _ChangeDetector.detect_changes(
            self._last_decision, new_decision, changes
        )

        for c in detected:
            self._logger.info(f"  Change: {c}")

        # Find invalidation point
        adaptive_steps = [s for s in new_decision.steps if s != "load_dataset"]
        invalidation = _ChangeDetector.find_invalidation_point(detected, adaptive_steps)

        if invalidation is None:
            self._logger.info("  No changes detected — returning cached result")
            elapsed = time.perf_counter() - start
            return IncrementalResult(
                success=True,
                pipeline_id=f"incr-{self._run_count}-cached",
                changes=detected,
                steps_reused=adaptive_steps,
                steps_reexecuted=[],
                steps_skipped=list(new_decision.skipped.keys()),
                final_data=self._last_result.final_data if self._last_result else None,
                elapsed_s=elapsed,
                decision=new_decision,
            )

        self._logger.info(f"  Invalidation point: {invalidation}")

        # Invalidate cache from that point
        invalidated = self._cache.invalidate_from(invalidation, adaptive_steps)
        self._logger.info(f"  Invalidated: {invalidated}")

        # Re-execute
        csv_path = self.csv_path
        df = pd.read_csv(csv_path, encoding="utf-8")

        steps_reused = []
        steps_reexecuted = []
        current_data = df

        for step in adaptive_steps:
            if self._cache.has(step):
                # Reuse cached result
                cached = self._cache.get(step)
                current_data = cached["output_data"]
                steps_reused.append(step)
                self._logger.info(f"  ♻  Reusing cached: {step}")
            else:
                # Re-execute
                self._logger.info(f"  ▶  Re-executing: {step}")
                agent = DynamicAgent(
                    step_name=step,
                    pipeline_steps=adaptive_steps,
                    api_key=self.api_key,
                    llm_model=self.llm_model,
                )
                try:
                    result = agent.execute(current_data)
                    if result.get("status") == "success":
                        current_data = result["output_data"]
                        h = _hash_dataframe(current_data) if isinstance(current_data, pd.DataFrame) else ""
                        self._cache.store(step, h, result)
                        steps_reexecuted.append(step)
                    else:
                        steps_reexecuted.append(f"{step} [FAILED]")
                except Exception as e:
                    self._logger.error(f"  Step {step} error: {e}")
                    steps_reexecuted.append(f"{step} [ERROR]")

        # Update state
        self._last_decision = new_decision

        elapsed = time.perf_counter() - start
        result = IncrementalResult(
            success=all("[" not in s for s in steps_reexecuted),
            pipeline_id=f"incr-{self._run_count}-{datetime.now(timezone.utc).strftime('%H%M%S')}",
            changes=detected,
            steps_reused=steps_reused,
            steps_reexecuted=steps_reexecuted,
            steps_skipped=list(new_decision.skipped.keys()),
            final_data=current_data,
            elapsed_s=elapsed,
            decision=new_decision,
        )
        self._last_result = result

        self._logger.info(f"\n  Result: {result.summary()}")
        self._print_comparison(detected, steps_reused, steps_reexecuted)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _insert_step_ordered(step: str, current_steps: List[str]) -> List[str]:
        """Insert a step in its canonical position."""
        order = _ChangeDetector._STEP_ORDER
        if step not in order:
            # Unknown step — append before training
            idx = next(
                (i for i, s in enumerate(current_steps) if "train" in s or "select" in s),
                len(current_steps)
            )
            return current_steps[:idx] + [step] + current_steps[idx:]

        step_rank = order.index(step)
        for i, s in enumerate(current_steps):
            if s in order and order.index(s) > step_rank:
                return current_steps[:i] + [step] + current_steps[i:]
        return current_steps + [step]

    def _print_comparison(
        self,
        changes: List[PipelineChange],
        reused: List[str],
        reexecuted: List[str],
    ) -> None:
        """Print a human-readable comparison summary."""
        print(f"\n{'='*60}")
        print("  INCREMENTAL MODIFICATION SUMMARY")
        print(f"{'='*60}")

        if changes:
            print("  Changes detected:")
            for c in changes:
                print(f"    → {c}")

        if reused:
            print(f"\n  ♻  Reused ({len(reused)} steps):")
            for s in reused:
                print(f"      {s}")

        if reexecuted:
            print(f"\n  ▶  Re-executed ({len(reexecuted)} steps):")
            for s in reexecuted:
                print(f"      {s}")

        savings = len(reused) / max(len(reused) + len(reexecuted), 1) * 100
        print(f"\n  Efficiency: {savings:.0f}% of steps reused")
        print(f"{'='*60}\n")
