"""
builder/agent_builder.py
------------------------
AgentBuilder — Dynamic Agent Factory

Role in the System
------------------
The AgentBuilder is responsible for one thing:
    Given a step name  →  return a ready-to-execute DynamicAgent

Old approach (what we replaced)
--------------------------------
    REGISTRY = {
        "load_dataset":          DataLoaderAgent,
        "remove_missing_values": MissingValueAgent,
        ...
    }
    # If the step is not in the registry → KeyError → pipeline crashes

New approach (what this module does)
--------------------------------------
    ANY step name is valid.
    The builder creates a DynamicAgent for it.
    The LLM figures out what that step means and how to execute it.

    "load_dataset"              → DynamicAgent("load_dataset")
    "remove_missing_values"     → DynamicAgent("remove_missing_values")
    "custom_fraud_detection"    → DynamicAgent("custom_fraud_detection")
    "apply_business_rules"      → DynamicAgent("apply_business_rules")
    "xyz_completely_new_step"   → DynamicAgent("xyz_completely_new_step")

    All of the above work. The LLM adapts to each one.

Additional Responsibilities
----------------------------
1.  Validate step names (catch obvious typos / empty strings)
2.  Pre-validate that the API key is available before building
3.  Inject pipeline context into every agent (so the LLM knows
    what came before and what comes after)
4.  Maintain a build log (which agents were created, when, for
    which pipeline run)
5.  Support step aliasing (e.g. "normalize" → "normalize_features")
6.  Provide a dry-run mode that builds all agents but does not
    execute them (for validation before running expensive API calls)

Design
------
The builder is intentionally stateless between pipeline runs.
Call build_agent() for each step, or build_all() for the full
pipeline at once.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.base_agent import DynamicAgent
from utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Step name aliases
# ---------------------------------------------------------------------------
# Maps shorthand / common variations to canonical step names.
# The LLM receives the canonical name which is more descriptive.

STEP_ALIASES: Dict[str, str] = {
    # Loading variants
    "load":                  "load_dataset",
    "read_data":             "load_dataset",
    "ingest":                "load_dataset",
    "read_csv":              "load_dataset",

    # Missing value variants
    "handle_missing":        "remove_missing_values",
    "drop_nulls":            "remove_missing_values",
    "impute":                "remove_missing_values",
    "fill_nulls":            "remove_missing_values",
    "handle_nulls":          "remove_missing_values",

    # Encoding variants
    "encode":                "encode_categorical",
    "label_encode":          "encode_categorical",
    "one_hot":               "encode_categorical",

    # Scaling variants
    "scale":                 "normalize_features",
    "normalize":             "normalize_features",
    "standardize":           "normalize_features",
    "normalise":             "normalize_features",
    "standardise":           "normalize_features",

    # Feature engineering variants
    "feature_eng":           "feature_engineering",
    "engineer_features":     "feature_engineering",
    "create_features":       "feature_engineering",
    "derive_features":       "feature_engineering",

    # Modeling variants
    "train":                 "train_model",
    "fit_model":             "train_model",
    "build_model":           "train_model",
    "model":                 "train_model",

    # Outlier variants
    "remove_outliers":       "handle_outliers",
    "outliers":              "handle_outliers",

    # EDA variants
    "eda":                   "exploratory_data_analysis",
    "explore":               "exploratory_data_analysis",
    "analyse":               "exploratory_data_analysis",
    "analyze":               "exploratory_data_analysis",

    # Splitting variants
    "split":                 "train_test_split",
    "split_data":            "train_test_split",
}


# ---------------------------------------------------------------------------
# Build record (for logging and diagnostics)
# ---------------------------------------------------------------------------

class _BuildRecord:
    """Records metadata about one agent build event."""

    def __init__(
        self,
        step_name:      str,
        canonical_name: str,
        pipeline_id:    str,
        build_time_ms:  float,
    ) -> None:
        self.step_name      = step_name
        self.canonical_name = canonical_name
        self.pipeline_id    = pipeline_id
        self.build_time_ms  = build_time_ms
        self.timestamp      = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name":      self.step_name,
            "canonical_name": self.canonical_name,
            "pipeline_id":    self.pipeline_id,
            "build_time_ms":  round(self.build_time_ms, 3),
            "timestamp":      self.timestamp,
        }


# ---------------------------------------------------------------------------
# AgentBuilder
# ---------------------------------------------------------------------------

class AgentBuilder:
    """
    Dynamic factory that creates DynamicAgent instances for any step name.

    Works with ANY backend — Ollama (gpt-oss:120b-cloud, llama3.1, etc.)
    or Anthropic Claude. Backend controlled by LLM_BACKEND env var.

    Supports conditional pipelines — the step list is determined at
    runtime by DataUnderstandingAgent, not hardcoded in YAML.
    Use rebuild(new_steps) after the data profile is known.

    Parameters
    ----------
    pipeline_steps : list[str]
        Initial step list. Can be [] and updated via rebuild().
    pipeline_id : str
        Unique run identifier from MemoryManager.
    api_key : str, optional
        Only needed when LLM_BACKEND=anthropic.
    llm_model : str
        Model name. Default: gpt-oss:120b-cloud
    """

    def __init__(
        self,
        pipeline_steps: List[str],
        pipeline_id:    str,
        api_key:        str = "",
        llm_model:      str = "gpt-oss:120b-cloud",
    ) -> None:
        self.pipeline_steps  = pipeline_steps
        self.pipeline_id     = pipeline_id
        self.llm_model       = llm_model
        self._logger         = PipelineLogger("builder.AgentBuilder")
        self._build_records: List[_BuildRecord] = []

        self._backend = os.environ.get("LLM_BACKEND", "ollama").lower()

        self.api_key = ""
        if self._backend == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        self._logger.info(
            f"AgentBuilder initialised | "
            f"pipeline_id={pipeline_id} | "
            f"backend={self._backend} | "
            f"model={llm_model} | "
            f"steps={pipeline_steps}"
        )

        if self._backend == "anthropic" and not self.api_key:
            self._logger.warning(
                "LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set."
            )

    def rebuild(self, new_steps: List[str]) -> None:
        """
        Update the pipeline step list after DataUnderstandingAgent
        has determined which steps are actually needed.

        Parameters
        ----------
        new_steps : list[str]
            The adaptive step list computed from data profile.
        """
        self._logger.info(
            f"Rebuilding agent list: {self.pipeline_steps} -> {new_steps}"
        )
        self.pipeline_steps = new_steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_agent(self, step_name: str) -> DynamicAgent:
        """
        Build a single DynamicAgent for the given step name.

        Parameters
        ----------
        step_name : str
            Any pipeline step name.  Will be resolved through aliases
            and validated before the agent is created.

        Returns
        -------
        DynamicAgent
            Ready-to-execute agent instance.

        Raises
        ------
        ValueError
            If the step name is empty or clearly invalid.
        """
        start = time.perf_counter()

        # ── Validate ──────────────────────────────────────────────────
        self._validate_step_name(step_name)

        # ── Resolve alias ─────────────────────────────────────────────
        canonical = self._resolve_alias(step_name)
        if canonical != step_name:
            self._logger.info(
                f"Step alias resolved: '{step_name}' → '{canonical}'"
            )

        # ── Build canonical pipeline context ──────────────────────────
        # Replace any aliases in the full pipeline list too
        resolved_pipeline = [
            self._resolve_alias(s) for s in self.pipeline_steps
        ]

        # ── Create DynamicAgent ───────────────────────────────────────
        agent = DynamicAgent(
            step_name      = canonical,
            pipeline_steps = resolved_pipeline,
            api_key        = self.api_key,
            llm_model      = self.llm_model,
        )

        # ── Record the build ──────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - start) * 1000
        record = _BuildRecord(
            step_name      = step_name,
            canonical_name = canonical,
            pipeline_id    = self.pipeline_id,
            build_time_ms  = elapsed_ms,
        )
        self._build_records.append(record)

        self._logger.agent_event(
            "AgentBuilder",
            f"Built DynamicAgent for '{canonical}' in {elapsed_ms:.2f}ms",
        )

        return agent

    def build_all(self) -> List[DynamicAgent]:
        """
        Build DynamicAgent instances for every step in the pipeline.

        Returns
        -------
        list[DynamicAgent]
            Agents in pipeline order, ready to be executed sequentially.
        """
        self._logger.info(
            f"Building {len(self.pipeline_steps)} agent(s) for pipeline "
            f"'{self.pipeline_id}' ..."
        )

        agents: List[DynamicAgent] = []
        for step in self.pipeline_steps:
            agent = self.build_agent(step)
            agents.append(agent)

        self._logger.info(
            f"All {len(agents)} agent(s) built successfully."
        )
        return agents

    def validate_pipeline(self, pipeline_steps: List[str]) -> Dict[str, Any]:
        """
        Validate an entire pipeline definition WITHOUT building agents.

        Checks
        ------
        - No empty or whitespace-only step names
        - No duplicate step names
        - No obviously malformed names (special characters etc.)
        - API key present

        Parameters
        ----------
        pipeline_steps : list[str]
            The pipeline to validate.

        Returns
        -------
        dict with keys:
            valid    : bool
            errors   : list of error messages
            warnings : list of warning messages
            resolved : list of canonical step names
        """
        errors:   List[str] = []
        warnings: List[str] = []
        resolved: List[str] = []

        seen: set = set()

        for i, step in enumerate(pipeline_steps):
            position = f"Step {i+1} ('{step}')"

            # Empty check
            if not step or not step.strip():
                errors.append(f"{position}: step name is empty")
                resolved.append("INVALID")
                continue

            # Malformed check
            if not self._is_valid_step_name(step):
                errors.append(
                    f"{position}: contains invalid characters. "
                    "Use lowercase letters, numbers, and underscores only."
                )

            # Duplicate check
            canonical = self._resolve_alias(step)
            if canonical in seen:
                warnings.append(
                    f"{position}: duplicate step '{canonical}' — "
                    "running the same step twice is allowed but unusual."
                )
            seen.add(canonical)
            resolved.append(canonical)

            # Alias notification
            if canonical != step:
                warnings.append(
                    f"{position}: alias '{step}' will be resolved to '{canonical}'"
                )

        # Backend-aware key check — only error when Anthropic is selected
        # and its key is missing. Ollama needs no key at all.
        if self._backend == "anthropic" and not self.api_key:
            errors.append(
                "LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set. "
                "Either set the key or switch to: LLM_BACKEND=ollama"
            )
        elif self._backend == "ollama":
            # Confirm Ollama model is set
            ollama_model = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
            if not ollama_model:
                errors.append(
                    "OLLAMA_MODEL is not set. "
                    "Set it with: set OLLAMA_MODEL=gpt-oss:120b-cloud"
                )

        valid = len(errors) == 0

        self._logger.info(
            f"Pipeline validation: {'PASSED' if valid else 'FAILED'} | "
            f"{len(errors)} error(s) | {len(warnings)} warning(s)"
        )

        return {
            "valid":    valid,
            "errors":   errors,
            "warnings": warnings,
            "resolved": resolved,
        }

    def get_build_log(self) -> List[Dict[str, Any]]:
        """Return the full build log as a list of dicts."""
        return [r.to_dict() for r in self._build_records]

    def print_build_log(self) -> None:
        """Print a formatted build log to stdout."""
        print("\n" + "=" * 60)
        print(f"  AGENT BUILD LOG  (pipeline: {self.pipeline_id})")
        print("=" * 60)
        if not self._build_records:
            print("  (no agents built yet)")
        for rec in self._build_records:
            alias_note = (
                f"  [alias: {rec.step_name} → {rec.canonical_name}]"
                if rec.step_name != rec.canonical_name
                else ""
            )
            print(
                f"  ✔  {rec.canonical_name:<40} "
                f"{rec.build_time_ms:.2f}ms{alias_note}"
            )
        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_alias(step_name: str) -> str:
        """
        Resolve a step name through the alias table.
        Returns the canonical name if an alias exists, else the original.
        """
        return STEP_ALIASES.get(step_name.strip().lower(), step_name.strip())

    @staticmethod
    def _is_valid_step_name(name: str) -> bool:
        """
        Return True if the name contains only safe characters.
        Allows: lowercase/uppercase letters, digits, underscores, hyphens.
        """
        import re
        return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_\-]*$", name.strip()))

    def _validate_step_name(self, step_name: str) -> None:
        """
        Raise ValueError if the step name is fundamentally invalid.
        Warnings are logged but do not raise.
        """
        if not step_name or not step_name.strip():
            raise ValueError(
                "Step name cannot be empty. "
                "Provide a descriptive name like 'normalize_features'."
            )

        if len(step_name) > 100:
            raise ValueError(
                f"Step name too long ({len(step_name)} chars). "
                "Keep it under 100 characters."
            )

        if not self._is_valid_step_name(step_name):
            # Log warning but still allow it — the LLM can handle
            # unconventional names
            self._logger.warning(
                f"Step name '{step_name}' contains unusual characters. "
                "Proceeding, but consider using snake_case names."
            )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def create_builder(
    pipeline_steps: List[str],
    pipeline_id:    str,
    api_key:        str = "",
    llm_model:      str = "gpt-oss:120b-cloud",
) -> AgentBuilder:
    """
    Convenience factory for creating an AgentBuilder.

    Uses gpt-oss:120b-cloud via Ollama by default.
    Switch backend via LLM_BACKEND environment variable.

    Parameters
    ----------
    pipeline_steps : list[str]
        All steps in the pipeline in execution order.
    pipeline_id : str
        Unique identifier for this run (from MemoryManager).
    api_key : str, optional
        Only needed when LLM_BACKEND=anthropic.
    llm_model : str
        Model name. Default: gpt-oss:120b-cloud
    """
    return AgentBuilder(
        pipeline_steps = pipeline_steps,
        pipeline_id    = pipeline_id,
        api_key        = api_key,
        llm_model      = llm_model,
    )