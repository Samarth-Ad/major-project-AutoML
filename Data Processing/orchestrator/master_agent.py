"""
orchestrator/master_agent.py
-----------------------------
MasterAgent — The Top-Level Conductor

Role in the System
------------------
The MasterAgent is the single entry point for the entire pipeline.
It owns nothing except coordination. It wires every module together
and drives execution from start to finish.

Responsibilities
----------------
1.  Read pipeline configuration  (from YAML / JSON / Python list)
2.  Initialise MemoryManager     (assigns pipeline_id, registers steps)
3.  Validate the pipeline        (via AgentBuilder.validate_pipeline)
4.  Initialise CodeWriterAgent   (creates pipeline_script.py header)
5.  Build all agents             (via AgentBuilder.build_all)
6.  Create and run the Scheduler (hands agents + memory + writer to it)
7.  Collect the ExecutionReport
8.  Print final summary
9.  Return PipelineResult

What MasterAgent does NOT do
-----------------------------
- Does not process any data itself
- Does not write any code
- Does not call the LLM directly
- Does not know what any pipeline step does

It only orchestrates the modules that do those things.

Wiring Diagram
--------------
    MasterAgent.run(pipeline_config)
            |
            +-- MemoryManager.init_pipeline(steps)
            |       -> assigns pipeline_id
            |
            +-- AgentBuilder(steps, pipeline_id)
            |       -> validates pipeline
            |       -> builds DynamicAgent per step
            |
            +-- CodeWriterAgent.init_script()
            |       -> writes pipeline_script.py header
            |
            +-- Scheduler.run(agents, initial_data)
                    |
                    +-- per step:
                            agent.execute(data)
                            memory.store_and_log_result()
                            code_writer.observe()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

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
    """
    The final output of a complete pipeline run returned by MasterAgent.

    Attributes
    ----------
    success : bool
        True if every step completed without error.
    pipeline_id : str
        Unique identifier for this run.
    final_data : Any
        The processed dataset or model result from the last step.
    report : ExecutionReport
        Full per-step execution report from the Scheduler.
    script_path : str
        Absolute path to the auto-generated pipeline_script.py.
    total_elapsed_s : float
        Wall-clock seconds for the entire pipeline.
    step_count : int
        Number of steps that were attempted.
    """
    success:         bool
    pipeline_id:     str
    final_data:      Any
    report:          ExecutionReport
    script_path:     str
    total_elapsed_s: float
    step_count:      int

    def summary(self) -> str:
        """One-line human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"PipelineResult("
            f"id={self.pipeline_id}, "
            f"status={status}, "
            f"steps={self.step_count}, "
            f"time={self.total_elapsed_s:.2f}s, "
            f"script={self.script_path})"
        )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_pipeline_config(
    config: Union[str, Path, List[str], Dict],
) -> List[str]:
    """
    Parse a pipeline configuration into an ordered list of step names.

    Accepts
    -------
    - list[str]  : already a list of step names
    - dict       : dict with key "steps" or "pipeline"
    - str / Path : path to a .yaml, .yml, or .json file

    Returns
    -------
    list[str]  ordered pipeline step names.
    """
    # Already a list
    if isinstance(config, list):
        return [str(s).strip() for s in config if str(s).strip()]

    # Dict with steps / pipeline key
    if isinstance(config, dict):
        steps = config.get("steps") or config.get("pipeline") or []
        if not steps:
            raise ValueError(
                "Config dict must have a 'steps' or 'pipeline' key "
                "containing the list of step names."
            )
        return [str(s).strip() for s in steps if str(s).strip()]

    # File path
    path = Path(config)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config file not found: {path}")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return _load_pipeline_config(data)

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _load_pipeline_config(data)

    raise ValueError(
        f"Unsupported config format: '{suffix}'. Use .yaml, .yml, or .json"
    )


# ---------------------------------------------------------------------------
# MasterAgent
# ---------------------------------------------------------------------------

class MasterAgent:
    """
    Top-level orchestrator that wires all system components together
    and drives a complete pipeline execution from config to result.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    llm_model : str
        Claude model used for all agent code generation.
    max_retries : int
        Max retry attempts per step (passed to Scheduler).
    abort_on_failure : bool
        Stop pipeline on first failed step (passed to Scheduler).
    backoff_base : float
        Exponential backoff base seconds between retries.

    Example
    -------
    .. code-block:: python

        from orchestrator.master_agent import MasterAgent

        agent = MasterAgent()

        # Run from a Python list
        result = agent.run([
            "load_dataset",
            "remove_missing_values",
            "encode_categorical",
            "normalize_features",
            "feature_engineering",
            "train_model",
        ])
        print(result.summary())

        # Run from YAML file
        result = agent.run("config/pipeline.yaml")

        # Dry-run (validate only, no LLM calls)
        info = agent.dry_run(["load_dataset", "train_model"])
    """

    def __init__(
        self,
        api_key:          str   = "",
        llm_model:        str   = "gpt-oss:120b-cloud",
        max_retries:      int   = 3,
        abort_on_failure: bool  = True,
        backoff_base:     float = 2.0,
    ) -> None:
        self.llm_model        = llm_model
        self.max_retries      = max_retries
        self.abort_on_failure = abort_on_failure
        self.backoff_base     = backoff_base
        self._logger          = PipelineLogger("orchestrator.MasterAgent")

        # Determine active backend
        self._backend = os.environ.get("LLM_BACKEND", "ollama").lower()

        # api_key is only relevant for Anthropic backend
        self.api_key = ""
        if self._backend == "anthropic":
            self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        self._logger.info(
            f"MasterAgent initialised | "
            f"backend={self._backend} | "
            f"model={llm_model} | "
            f"max_retries={max_retries} | "
            f"abort_on_failure={abort_on_failure}"
        )

        # Only warn about missing key when Anthropic is the active backend
        if self._backend == "anthropic" and not self.api_key:
            self._logger.warning(
                "LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set. "
                "Agents will fail when executed. "
                "Switch to Ollama with: set LLM_BACKEND=ollama"
            )

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run(
        self,
        pipeline_config: Union[str, Path, List[str], Dict],
        initial_data:    Any = None,
    ) -> PipelineResult:
        """
        Execute a full pipeline end-to-end.

        Parameters
        ----------
        pipeline_config : str | Path | list | dict
            Pipeline definition — step list, YAML path, or config dict.
        initial_data : Any, optional
            Pre-loaded data to begin the pipeline with.
            Pass None when the first step is 'load_dataset'.

        Returns
        -------
        PipelineResult
        """
        master_start = time.perf_counter()

        # ── Banner ────────────────────────────────────────────────────
        self._logger.info("")
        self._logger.info("=" * 60)
        self._logger.info("  AGENTIC PIPELINE BUILDER — EXECUTION STARTING")
        self._logger.info("=" * 60)

        # ── 1. Parse config ───────────────────────────────────────────
        self._logger.info("[1/6] Parsing pipeline configuration ...")
        try:
            steps = _load_pipeline_config(pipeline_config)
        except (FileNotFoundError, ValueError) as exc:
            self._logger.error(f"Config parse failed: {exc}")
            raise

        self._logger.info(
            f"      {len(steps)} step(s) found: {' -> '.join(steps)}"
        )

        # ── 2. Initialise MemoryManager ───────────────────────────────
        self._logger.info("[2/6] Initialising MemoryManager ...")
        memory      = MemoryManager()
        pipeline_id = memory.init_pipeline(steps)
        self._logger.info(f"      Pipeline ID: {pipeline_id}")

        # ── 3. Build AgentBuilder + validate ──────────────────────────
        self._logger.info("[3/6] Building AgentFactory and validating pipeline ...")
        builder = create_builder(
            pipeline_steps = steps,
            pipeline_id    = pipeline_id,
            api_key        = self.api_key,
            llm_model      = self.llm_model,
        )

        validation = builder.validate_pipeline(steps)

        for w in validation.get("warnings", []):
            self._logger.warning(f"      WARNING: {w}")

        if not validation["valid"]:
            for e in validation["errors"]:
                self._logger.error(f"      ERROR: {e}")
            raise ValueError(
                f"Pipeline validation failed:\n"
                + "\n".join(validation["errors"])
            )

        self._logger.info("      Validation passed.")

        # ── 4. Build all agents ───────────────────────────────────────
        self._logger.info("[4/6] Building DynamicAgents ...")
        agents = builder.build_all()
        builder.print_build_log()
        self._logger.info(f"      {len(agents)} agent(s) ready.")

        # ── 5. Initialise CodeWriterAgent ─────────────────────────────
        self._logger.info("[5/6] Initialising CodeWriterAgent ...")
        code_writer = CodeWriterAgent(
            pipeline_id    = pipeline_id,
            pipeline_steps = steps,
        )
        code_writer.init_script()
        self._logger.info(
            f"      Script path: {code_writer.get_script_path()}"
        )

        # ── 6. Run Scheduler ──────────────────────────────────────────
        self._logger.info("[6/6] Launching Scheduler ...")
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

        # ── Print memory snapshot ─────────────────────────────────────
        memory.print_summary()

        # ── Build PipelineResult ──────────────────────────────────────
        result = PipelineResult(
            success         = report.success,
            pipeline_id     = pipeline_id,
            final_data      = report.final_data,
            report          = report,
            script_path     = script_path,
            total_elapsed_s = total_elapsed,
            step_count      = len(steps),
        )

        self._print_final_banner(result)
        return result

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def run_from_yaml(
        self,
        yaml_path:    Union[str, Path],
        initial_data: Any = None,
    ) -> PipelineResult:
        """Run a pipeline defined in a YAML file."""
        return self.run(yaml_path, initial_data=initial_data)

    def run_from_list(
        self,
        steps:        List[str],
        initial_data: Any = None,
    ) -> PipelineResult:
        """Run a pipeline defined as a Python list."""
        return self.run(steps, initial_data=initial_data)

    def dry_run(
        self,
        pipeline_config: Union[str, Path, List[str], Dict],
    ) -> Dict[str, Any]:
        """
        Validate a pipeline WITHOUT executing any agents or LLM calls.

        Returns
        -------
        dict  with keys: valid, steps, resolved, errors, warnings
        """
        self._logger.info("DRY RUN — validating pipeline (no execution)")

        steps   = _load_pipeline_config(pipeline_config)
        memory  = MemoryManager()
        pid     = memory.init_pipeline(steps)
        builder = create_builder(steps, pid, self.api_key, self.llm_model)
        result  = builder.validate_pipeline(steps)

        print("\n" + "=" * 56)
        print("  DRY RUN RESULT")
        print("=" * 56)
        print(f"  Valid      : {result['valid']}")
        print(f"  Steps ({len(steps)}): {' -> '.join(steps)}")
        if result["errors"]:
            print("\n  ERRORS:")
            for e in result["errors"]:
                print(f"    [X] {e}")
        if result["warnings"]:
            print("\n  WARNINGS:")
            for w in result["warnings"]:
                print(f"    [!] {w}")
        print("=" * 56 + "\n")

        return {
            "valid":    result["valid"],
            "steps":    steps,
            "resolved": result["resolved"],
            "errors":   result["errors"],
            "warnings": result["warnings"],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_final_banner(self, result: PipelineResult) -> None:
        """Print the completion banner."""
        status = "COMPLETED SUCCESSFULLY" if result.success else "COMPLETED WITH ERRORS"
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
        self._logger.info("=" * 60)
        self._logger.info("")