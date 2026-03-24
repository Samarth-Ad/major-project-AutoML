"""
orchestrator/master_agent.py
-----------------------------
MasterAgent — Adaptive Pipeline Conductor

Two-phase execution
-------------------
Phase 1 — Bootstrap (always 2 steps, fixed):
    1a. load_dataset        → loads CSV into DataFrame
    1b. understand_data     → DataUnderstandingAgent analyses data,
                              returns PipelineDecision

Phase 2 — Adaptive (steps decided by PipelineDecision):
    The decision object's .steps list contains ONLY
    the steps this specific dataset needs. MasterAgent builds agents
    for exactly those steps and runs them.

Nothing in Phase 2 is hardcoded. The step list comes entirely
from the LLM + rule-based analysis of the actual data.

Config
------
config/pipeline.yaml contains:
  - LLM settings (backend, model, retries)
  - Data settings (filepath, target_column)
  - Thresholds (when each step fires)

It does NOT contain a fixed step list.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from agents.data_understanding_agent import DataUnderstandingAgent
from agents.pipeline_decision import PipelineDecision
from builder.agent_builder import AgentBuilder, create_builder
from execution.scheduler import ExecutionReport, Scheduler
from memory.memory_manager import MemoryManager
from observer.code_writer_agent import CodeWriterAgent
from utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full output of a MasterAgent.run() call."""
    success:         bool
    pipeline_id:     str
    final_data:      Any
    report:          ExecutionReport
    script_path:     str
    total_elapsed_s: float
    step_count:      int
    decision:        Optional[PipelineDecision] = None

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"PipelineResult("
            f"id={self.pipeline_id}, status={status}, "
            f"steps={self.step_count}, time={self.total_elapsed_s:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Config / YAML loader
# ---------------------------------------------------------------------------

def _load_yaml_config(config_path: Union[str, Path]) -> dict:
    """Load the full pipeline.yaml as a dict. Returns {} on failure."""
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _extract_initial_steps_from_config(cfg: dict) -> List[str]:
    """
    Old YAML format had a steps: list. New format does not.
    If someone passes a steps list (e.g. from CLI --steps), we use it.
    Otherwise return [] to signal adaptive mode.
    """
    return cfg.get("steps", [])


# ---------------------------------------------------------------------------
# MasterAgent
# ---------------------------------------------------------------------------

class MasterAgent:
    """
    Top-level conductor for the adaptive ML pipeline.

    Two modes
    ---------
    Adaptive (default):
        Pass a YAML config path or nothing.
        DataUnderstandingAgent decides the steps.

    Manual override:
        Pass a list of step names directly.
        Skips DataUnderstandingAgent.
        Used for testing or when you know exactly what you need.

    Parameters
    ----------
    api_key : str
        Anthropic API key (ignored for Ollama backend).
    llm_model : str
        Model name. Default: gpt-oss:120b-cloud.
    max_retries : int
        Retries per failed step.
    abort_on_failure : bool
        Stop on first failed step.
    backoff_base : float
        Exponential backoff base (seconds).
    """

    def __init__(
        self,
        api_key:          str   = "",
        llm_model:        str   = "gpt-oss:120b-cloud",
        max_retries:      int   = 3,
        abort_on_failure: bool  = True,
        backoff_base:     float = 2.0,
        config_path:      str   = "config/pipeline.yaml",
    ) -> None:
        self.llm_model        = llm_model
        self.max_retries      = max_retries
        self.abort_on_failure = abort_on_failure
        self.backoff_base     = backoff_base
        self.config_path      = config_path
        self._logger          = PipelineLogger("orchestrator.MasterAgent")

        self._backend = os.environ.get("LLM_BACKEND", "ollama").lower()
        self.api_key  = ""
        if self._backend == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        self._logger.info(
            f"MasterAgent initialised | backend={self._backend} | "
            f"model={llm_model} | retries={max_retries} | "
            f"abort_on_failure={abort_on_failure}"
        )

        if self._backend == "anthropic" and not self.api_key:
            self._logger.warning(
                "LLM_BACKEND=anthropic but ANTHROPIC_API_KEY not set. "
                "Use: set LLM_BACKEND=ollama"
            )

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        pipeline_config: Union[str, Path, List[str], Dict] = "config/pipeline.yaml",
        initial_data:    Any = None,
        target_column:   str = "",
    ) -> PipelineResult:
        """
        Execute the full adaptive pipeline.

        Parameters
        ----------
        pipeline_config : str | Path | list | dict
            - str/Path  → path to pipeline.yaml  (adaptive mode)
            - list[str] → explicit step list       (manual mode)
            - dict      → config dict
        initial_data : str | pd.DataFrame | None
            Starting data. Pass a CSV filepath or None.
        target_column : str, optional
            Override the target column. Auto-inferred if empty.
        """
        master_start = time.perf_counter()

        self._logger.info("")
        self._logger.info("=" * 60)
        self._logger.info("  ADAPTIVE PIPELINE — EXECUTION STARTING")
        self._logger.info("=" * 60)

        # ── Load YAML config ──────────────────────────────────────────
        yaml_cfg     = {}
        manual_steps = []

        if isinstance(pipeline_config, list):
            manual_steps = [str(s).strip() for s in pipeline_config if str(s).strip()]
            self._logger.info(f"Manual mode: {len(manual_steps)} steps provided")

        elif isinstance(pipeline_config, dict):
            yaml_cfg     = pipeline_config
            manual_steps = yaml_cfg.get("steps", [])

        else:
            config_path = Path(pipeline_config)
            if config_path.exists():
                yaml_cfg = _load_yaml_config(config_path)
                # Check if old-style fixed steps exist in YAML
                manual_steps = yaml_cfg.get("steps", [])
            else:
                self._logger.warning(
                    f"Config file not found: {config_path}. "
                    "Running in adaptive mode with default settings."
                )

        # ── Target column: CLI arg overrides YAML ────────────────────
        target_col = (
            target_column
            or yaml_cfg.get("data", {}).get("target_column", "")
            or os.environ.get("TARGET_COLUMN", "")
        )

        # ── Initialise memory ─────────────────────────────────────────
        self._logger.info("[1] Initialising MemoryManager ...")
        memory = MemoryManager()

        # ── PHASE 1 — Bootstrap ───────────────────────────────────────
        # Always run load_dataset first, then DataUnderstandingAgent.
        # These two steps are fixed — all others are adaptive.

        bootstrap_steps = ["load_dataset"]

        if manual_steps:
            # Manual mode — user provided explicit step list
            # Skip DataUnderstandingAgent, trust the user
            self._logger.info(
                f"[2] Manual mode — using provided steps: {manual_steps}"
            )
            all_steps = manual_steps
            decision  = None
        else:
            # Adaptive mode
            bootstrap_steps = ["load_dataset", "understand_data"]
            self._logger.info("[2] Adaptive mode — DataUnderstandingAgent will decide steps")
            decision = None   # filled in after Phase 1 runs
            all_steps = bootstrap_steps  # temporary — will be extended

        # ── Initialise pipeline in memory ─────────────────────────────
        pipeline_id = memory.init_pipeline(all_steps)
        self._logger.info(f"     Pipeline ID: {pipeline_id}")

        # ── Build initial AgentBuilder ────────────────────────────────
        self._logger.info("[3] Building AgentFactory ...")
        builder = create_builder(
            pipeline_steps = all_steps,
            pipeline_id    = pipeline_id,
            api_key        = self.api_key,
            llm_model      = self.llm_model,
        )

        # ── PHASE 1 EXECUTION — load + understand ─────────────────────
        if not manual_steps:
            self._logger.info("[4] Phase 1 — Loading and understanding data ...")

            # Run load_dataset agent
            load_agent = builder.build_agent("load_dataset")
            load_result = self._run_single_agent(load_agent, initial_data, memory)

            if load_result["status"] != "success":
                raise RuntimeError(
                    f"load_dataset failed: {load_result.get('error', 'unknown')}"
                )

            loaded_data = load_result["output_data"]
            memory.store_and_log_result("load_dataset", load_result)
            memory.set_step_status("load_dataset", "success")
            memory.set_current_data(loaded_data)

            # Run DataUnderstandingAgent
            self._logger.info("[4] Phase 1 — Running DataUnderstandingAgent ...")
            dua = DataUnderstandingAgent(
                llm_model   = self.llm_model,
                api_key     = self.api_key,
                config_path = str(pipeline_config)
                              if isinstance(pipeline_config, (str, Path))
                              else "config/pipeline.yaml",
            )
            # execute() returns (PipelineDecision, result_dict) tuple
            decision, understand_result = dua.execute(
                input_data    = loaded_data,
                target_column = target_col,
            )

            if understand_result["status"] != "success":
                self._logger.warning(
                    "DataUnderstandingAgent failed — using rule-based fallback steps"
                )
                from agents.data_understanding_agent import (
                    _rule_based_decision, _analyse_data, _load_thresholds
                )
                thresholds = _load_thresholds(
                    str(pipeline_config)
                    if isinstance(pipeline_config, (str, Path))
                    else "config/pipeline.yaml"
                )
                profile      = _analyse_data(loaded_data, thresholds, target_col)
                fallback_dict = _rule_based_decision(profile, self._logger)
                # Build a minimal PipelineDecision from fallback
                from agents.data_understanding_agent import PipelineDecision as _PD
                decision = _PD(
                    problem_type    = fallback_dict["problem_type"],
                    target_column   = fallback_dict["target_column"],
                    steps           = fallback_dict["steps"],
                    skipped         = fallback_dict["skipped"],
                    models_to_try   = fallback_dict["models_to_try"],
                    needs_tuning    = fallback_dict["needs_tuning"],
                    n_rows          = int(loaded_data.shape[0]) if hasattr(loaded_data, "shape") else 0,
                    n_cols          = int(loaded_data.shape[1]) if hasattr(loaded_data, "shape") else 0,
                    has_nulls       = profile["checks"]["has_nulls"],
                    is_imbalanced   = profile["is_imbalanced"],
                    skewed_columns  = profile["skewed_columns"],
                    high_corr_pairs = profile["high_correlation_pairs"],
                    reasoning       = fallback_dict["reasoning"],
                )

            # Store decision in memory for downstream steps
            memory.set_data_profile(decision.to_dict())
            memory.store_and_log_result("understand_data", understand_result)
            memory.set_step_status("understand_data", "success")

            # ── PHASE 2 — Build adaptive step list ────────────────────
            # decision.steps already contains the ordered step list
            adaptive_steps = [s for s in decision.steps if s != "load_dataset"]

            self._logger.info(
                f"[5] Phase 2 — Adaptive steps decided:\n"
                f"      {' -> '.join(adaptive_steps)}\n"
                f"      Problem type  : {decision.problem_type}\n"
                f"      Target column : {decision.target_column}\n"
                f"      Models        : {decision.models_to_try}"
            )

            # Update builder with final step list
            builder.rebuild(adaptive_steps)

            # Re-init memory with full step list for accurate tracking
            all_steps = ["load_dataset", "understand_data"] + adaptive_steps
            memory.init_pipeline(all_steps)   # reset with full list
            pipeline_id = memory.get_pipeline_id()

            # Mark bootstrap steps as already done
            memory.set_step_status("load_dataset",   "success")
            memory.set_step_status("understand_data", "success")

            # Set initial_data to the loaded DataFrame for Phase 2
            initial_data = loaded_data

        else:
            # Manual mode — validate then proceed
            validation = builder.validate_pipeline(all_steps)
            if not validation["valid"]:
                raise ValueError(
                    "Pipeline validation failed:\n" +
                    "\n".join(validation["errors"])
                )

        # ── PHASE 2 EXECUTION ─────────────────────────────────────────
        self._logger.info("[6] Phase 2 — Building and running pipeline agents ...")

        # Determine steps for Phase 2
        if manual_steps:
            phase2_steps = all_steps
        else:
            phase2_steps = adaptive_steps

        # Build Phase 2 agents
        p2_builder = create_builder(
            pipeline_steps = phase2_steps,
            pipeline_id    = pipeline_id,
            api_key        = self.api_key,
            llm_model      = self.llm_model,
        )
        agents = p2_builder.build_all()
        p2_builder.print_build_log()
        self._logger.info(f"     {len(agents)} agent(s) ready.")

        # ── CodeWriterAgent ───────────────────────────────────────────
        code_writer = CodeWriterAgent(
            pipeline_id    = pipeline_id,
            pipeline_steps = phase2_steps,
        )
        code_writer.init_script()
        self._logger.info(f"     Script: {code_writer.get_script_path()}")

        # Write bootstrap steps to the script if adaptive mode
        if not manual_steps and decision:
            _write_decision_to_script(code_writer, decision)

        # ── Scheduler ────────────────────────────────────────────────
        scheduler = Scheduler(
            memory           = memory,
            code_writer      = code_writer,
            max_retries      = self.max_retries,
            backoff_base     = self.backoff_base,
            abort_on_failure = self.abort_on_failure,
        )

        report = scheduler.run(
            agents       = agents,
            initial_data = initial_data,
        )

        # ── Collect final state ───────────────────────────────────────
        total_elapsed = time.perf_counter() - master_start
        script_path   = str(code_writer.get_script_path().resolve())

        memory.print_summary()

        result = PipelineResult(
            success         = report.success,
            pipeline_id     = pipeline_id,
            final_data      = report.final_data,
            report          = report,
            script_path     = script_path,
            total_elapsed_s = total_elapsed,
            step_count      = len(phase2_steps),
            decision        = decision,
        )

        self._print_final_banner(result)
        return result

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    def dry_run(
        self,
        pipeline_config: Union[str, Path, List[str], Dict] = "config/pipeline.yaml",
    ) -> Dict[str, Any]:
        """
        Validate config without any LLM calls or data processing.
        Shows thresholds and confirms settings are valid.
        """
        self._logger.info("DRY RUN — no execution")

        yaml_cfg = {}
        if isinstance(pipeline_config, list):
            # Manual mode — validate the explicit step list
            steps = pipeline_config
        else:
            yaml_cfg = (
                _load_yaml_config(pipeline_config)
                if isinstance(pipeline_config, (str, Path))
                else {}
            )
            steps = yaml_cfg.get("steps", [])

        thresholds = yaml_cfg.get("thresholds", {})

        if not steps:
            # Adaptive mode — show default fallback steps as preview
            from agents.data_understanding_agent import _rule_based_decision
            preview_profile = {
                "problem_type": "binary_classification",
                "target_column": yaml_cfg.get("data", {}).get("target_column", "target"),
                "skewed_columns": [],
                "high_correlation_pairs": [],
                "is_imbalanced": False,
                "est_cols_after_encoding": 15,
                "n_rows": 1000, "n_cols": 12, "n_numeric": 9,
                "checks": {
                    "has_nulls": True, "needs_smote": False,
                    "needs_skewness": False, "needs_pca": False,
                    "needs_tuning": True, "has_categoricals": True,
                    "needs_scaling": True, "needs_feat_eng": True,
                },
                "thresholds": {
                    "null_pct_to_trigger_imputation":   0.0,
                    "imbalance_ratio_to_trigger_smote": 0.25,
                    "skewness_to_trigger_correction":   1.0,
                    "features_to_trigger_pca":          50,
                    "rows_to_trigger_tuning":           500,
                    **thresholds,
                },
            }
            from utils.logger import PipelineLogger as _PL
            fallback = _rule_based_decision(preview_profile, _PL("dry_run"))
            steps    = fallback["steps"]
            self._logger.info(
                "Adaptive mode — showing default step preview "
                "(actual steps decided at runtime from your data)"
            )

        memory  = MemoryManager()
        pid     = memory.init_pipeline(steps)
        builder = create_builder(steps, pid, self.api_key, self.llm_model)
        result  = builder.validate_pipeline(steps)

        print("\n" + "=" * 60)
        print("  DRY RUN RESULT")
        print("=" * 60)
        print(f"  Mode       : {'Manual' if isinstance(pipeline_config, list) else 'Adaptive'}")
        print(f"  Valid      : {result['valid']}")
        print(f"  Steps ({len(steps)}): {' -> '.join(steps)}")
        if thresholds:
            print("\n  Thresholds:")
            for k, v in thresholds.items():
                print(f"    {k:<45}: {v}")
        if result["errors"]:
            print("\n  ERRORS:")
            for e in result["errors"]:
                print(f"    [X] {e}")
        if result["warnings"]:
            print("\n  WARNINGS:")
            for w in result["warnings"]:
                print(f"    [!] {w}")
        print("=" * 60 + "\n")

        return {
            "valid":    result["valid"],
            "steps":    steps,
            "resolved": result["resolved"],
            "errors":   result["errors"],
            "warnings": result["warnings"],
        }

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def run_from_yaml(self, yaml_path: Union[str, Path], initial_data: Any = None) -> PipelineResult:
        return self.run(yaml_path, initial_data=initial_data)

    def run_from_list(self, steps: List[str], initial_data: Any = None) -> PipelineResult:
        return self.run(steps, initial_data=initial_data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_single_agent(
        self,
        agent:      Any,
        input_data: Any,
        memory:     MemoryManager,
    ) -> Dict[str, Any]:
        """Run one agent with retry logic (used for bootstrap steps)."""
        memory.set_step_running(agent.step_name)
        last_result: Dict[str, Any] = {}

        for attempt in range(1, self.max_retries + 1):
            try:
                result = agent.execute(input_data)
            except Exception as exc:
                self._logger.error(
                    f"Unhandled exception in {agent.agent_name}: {exc}"
                )
                result = {
                    "status":         "failed",
                    "error":          str(exc),
                    "output_data":    input_data,
                    "code_equivalent":"",
                    "reasoning":      "",
                    "step_name":      agent.step_name,
                    "agent_name":     agent.agent_name,
                    "elapsed_ms":     0,
                    "task_id":        "bootstrap",
                    "input_summary":  "",
                    "output_summary": "",
                    "timestamp":      "",
                }

            last_result = result
            status      = result.get("status", "failed")

            if status == "success":
                return result
            if status == "retry" and attempt <= self.max_retries:
                wait = self.backoff_base ** attempt
                self._logger.warning(f"Retry {attempt}/{self.max_retries} in {wait}s ...")
                time.sleep(wait)
                memory.increment_retry(agent.step_name)
                continue
            break

        return last_result

    def _print_final_banner(self, result: PipelineResult) -> None:
        status     = "SUCCESS" if result.success else "FAILED"
        successful = result.report.successful_steps
        total      = result.step_count

        self._logger.info("")
        self._logger.info("=" * 60)
        self._logger.info(f"  PIPELINE {status}")
        self._logger.info("=" * 60)
        self._logger.info(f"  Pipeline ID  : {result.pipeline_id}")
        self._logger.info(f"  Total time   : {result.total_elapsed_s:.3f}s")
        self._logger.info(f"  Steps        : {successful}/{total} succeeded")
        self._logger.info(f"  Script saved : {result.script_path}")
        if result.decision:
            self._logger.info(f"  Problem type : {result.decision.problem_type}")
            self._logger.info(f"  Models used  : {result.decision.models_to_try}")
        self._logger.info("=" * 60)
        self._logger.info("")


# ---------------------------------------------------------------------------
# Helper: write decision summary into the generated script
# ---------------------------------------------------------------------------

def _write_decision_to_script(
    code_writer: CodeWriterAgent,
    decision:    PipelineDecision,
) -> None:
    """
    Append a data understanding summary block to pipeline_script.py
    so the generated notebook documents the decision reasoning.
    """
    try:
        from pathlib import Path
        script_path = code_writer.get_script_path()
        if not script_path.exists():
            return

        block = (
            "\n\n"
            "# ════════════════════════════════════════════════════════════\n"
            "# DATA UNDERSTANDING — PIPELINE DECISION\n"
            "# ════════════════════════════════════════════════════════════\n"
            f"# Problem type   : {decision.problem_type}\n"
            f"# Target column  : {decision.target_column}\n"
            f"# Models selected: {decision.models_to_try}\n"
            f"# Steps computed : {decision.steps}\n"
            "#\n"
            "# Steps skipped and why:\n"
        )
        for step, reason in (decision.skipped or {}).items():
            block += f"#   SKIP {step}: {reason}\n"
        block += "#\n# LLM Reasoning:\n"
        for step, reason in (decision.reasoning or {}).items():
            block += f"#   {step}: {reason}\n"
        block += "# ════════════════════════════════════════════════════════════\n"

        with script_path.open("a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        pass   # non-critical — don't crash the pipeline for a script annotation