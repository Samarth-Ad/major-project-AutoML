"""
execution/scheduler.py
----------------------
Scheduler — The Execution Engine

Role in the System
------------------
The Scheduler sits between the MasterAgent and the individual
DynamicAgents. It is responsible for:

    1. Running agents in correct sequential order
    2. Passing output of each agent as input to the next
    3. Retrying failed steps with exponential backoff
    4. Notifying CodeWriterAgent after every step
    5. Notifying MemoryManager after every step
    6. Tracking per-step and total execution timing
    7. Deciding when to abort vs when to continue on failure
    8. Producing a structured ExecutionReport at the end

Execution Flow
--------------
    Scheduler.run(agents, initial_data)
         |
         +-- for each agent in order:
         |       1. memory.set_step_running(step)
         |       2. result = agent.execute(current_data)
         |       3. if status == "retry": wait -> retry (max_retries)
         |       4. if status == "failed": abort / skip
         |       5. memory.store_and_log_result(result)
         |       6. code_writer.observe(result)
         |       7. current_data = result["output_data"]
         |
         +-- return ExecutionReport

Retry Policy
------------
    max_retries     : 3  (configurable)
    backoff_base    : 2  (seconds)
    backoff formula : wait = backoff_base ^ attempt
    attempt 1 -> wait 2s
    attempt 2 -> wait 4s
    attempt 3 -> wait 8s

Abort Policy
------------
    abort_on_failure=True  (default)
        Pipeline stops at first failed step.
        Remaining steps are marked "skipped".

    abort_on_failure=False
        Pipeline continues even if a step fails.
        Failed step's input data is passed through unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.base_agent import DynamicAgent
from memory.memory_manager import MemoryManager
from observer.code_writer_agent import CodeWriterAgent
from utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# StepOutcome
# ---------------------------------------------------------------------------

@dataclass
class StepOutcome:
    """Holds the outcome of one scheduler-managed step execution."""
    step_index:     int
    step_name:      str
    status:         str           # success | failed | skipped | retry_exhausted
    agent_result:   Dict[str, Any]
    attempts:       int
    elapsed_s:      float
    skipped_reason: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    @property
    def output_data(self) -> Any:
        return self.agent_result.get("output_data")

    @property
    def code(self) -> str:
        return self.agent_result.get("code_equivalent", "")

    @property
    def reasoning(self) -> str:
        return self.agent_result.get("reasoning", "")


# ---------------------------------------------------------------------------
# ExecutionReport
# ---------------------------------------------------------------------------

@dataclass
class ExecutionReport:
    """Full summary of the pipeline execution returned by Scheduler.run()."""
    pipeline_id:     str
    pipeline_steps:  List[str]
    outcomes:        List[StepOutcome] = field(default_factory=list)
    final_data:      Any               = None
    started_at:      str               = ""
    finished_at:     str               = ""
    total_elapsed_s: float             = 0.0
    success:         bool              = False

    @property
    def total_steps(self) -> int:
        return len(self.pipeline_steps)

    @property
    def successful_steps(self) -> int:
        return sum(1 for o in self.outcomes if o.succeeded)

    @property
    def failed_steps(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")

    @property
    def skipped_steps(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "skipped")

    def print_report(self) -> None:
        divider = "=" * 64
        print(f"\n{divider}")
        print(f"  PIPELINE EXECUTION REPORT")
        print(f"  Pipeline ID  : {self.pipeline_id}")
        print(f"  Status       : {'SUCCESS' if self.success else 'FAILED'}")
        print(f"  Total time   : {self.total_elapsed_s:.3f}s")
        print(f"  Started      : {self.started_at}")
        print(f"  Finished     : {self.finished_at}")
        print(divider)
        print(f"  Steps total  : {self.total_steps}")
        print(f"  Successful   : {self.successful_steps}")
        print(f"  Failed       : {self.failed_steps}")
        print(f"  Skipped      : {self.skipped_steps}")
        print(divider)
        for outcome in self.outcomes:
            icon = {
                "success":         "[OK]",
                "failed":          "[FAIL]",
                "skipped":         "[SKIP]",
                "retry_exhausted": "[RETRY]",
            }.get(outcome.status, "[?]")
            attempts_str = (
                f"  ({outcome.attempts} attempts)"
                if outcome.attempts > 1 else ""
            )
            print(
                f"  {icon:<8} {outcome.step_index}. "
                f"{outcome.step_name:<38} "
                f"{outcome.elapsed_s:.3f}s{attempts_str}"
            )
            if outcome.skipped_reason:
                print(f"           Reason: {outcome.skipped_reason}")
        print(f"{divider}\n")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id":      self.pipeline_id,
            "success":          self.success,
            "total_elapsed_s":  round(self.total_elapsed_s, 3),
            "total_steps":      self.total_steps,
            "successful_steps": self.successful_steps,
            "failed_steps":     self.failed_steps,
            "skipped_steps":    self.skipped_steps,
            "started_at":       self.started_at,
            "finished_at":      self.finished_at,
            "step_outcomes": [
                {
                    "index":     o.step_index,
                    "step":      o.step_name,
                    "status":    o.status,
                    "attempts":  o.attempts,
                    "elapsed_s": round(o.elapsed_s, 3),
                }
                for o in self.outcomes
            ],
        }


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """
    Sequential execution engine for the agentic pipeline.

    Parameters
    ----------
    memory : MemoryManager
        Shared memory store for state and logging.
    code_writer : CodeWriterAgent
        Observer that writes the pipeline script incrementally.
    max_retries : int
        Maximum retry attempts per step on recoverable failure.
    backoff_base : float
        Base seconds for exponential backoff between retries.
    abort_on_failure : bool
        If True, stop pipeline when any step fails.
        If False, pass step input through unchanged and continue.
    """

    def __init__(
        self,
        memory:           MemoryManager,
        code_writer:      CodeWriterAgent,
        max_retries:      int   = 3,
        backoff_base:     float = 2.0,
        abort_on_failure: bool  = True,
    ) -> None:
        self.memory           = memory
        self.code_writer      = code_writer
        self.max_retries      = max_retries
        self.backoff_base     = backoff_base
        self.abort_on_failure = abort_on_failure
        self._logger          = PipelineLogger("execution.Scheduler")

        self._logger.info(
            f"Scheduler ready | max_retries={max_retries} | "
            f"backoff={backoff_base}s | abort_on_failure={abort_on_failure}"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        agents:       List[DynamicAgent],
        initial_data: Any = None,
    ) -> ExecutionReport:
        """
        Execute all agents sequentially and return the execution report.

        Parameters
        ----------
        agents : list[DynamicAgent]
            Agents in pipeline order from AgentBuilder.build_all().
        initial_data : Any, optional
            Starting data for first agent. None for load_dataset first step.

        Returns
        -------
        ExecutionReport
        """
        pipeline_id    = self.memory.get_pipeline_id()
        pipeline_steps = [a.step_name for a in agents]
        started_at     = datetime.now(timezone.utc).isoformat()
        pipeline_start = time.perf_counter()
        total          = len(agents)

        self._logger.pipeline_start(pipeline_steps)

        report = ExecutionReport(
            pipeline_id    = pipeline_id,
            pipeline_steps = pipeline_steps,
            started_at     = started_at,
        )

        current_data = initial_data
        aborted      = False

        # ── Execute each agent in order ───────────────────────────────
        for idx, agent in enumerate(agents, start=1):
            step_name = agent.step_name

            # ── Skip if aborted ───────────────────────────────────────
            if aborted:
                outcome = self._make_skipped_outcome(
                    idx, step_name,
                    reason="Pipeline aborted due to earlier failure"
                )
                report.outcomes.append(outcome)
                self.memory.set_step_status(step_name, "skipped")
                self._logger.info(
                    f"  [{idx}/{total}] Skipping '{step_name}' (aborted)"
                )
                continue

            # ── Progress indicator ────────────────────────────────────
            self._logger.info(
                f"\n{'─' * 56}\n"
                f"  STEP {idx}/{total}: {step_name.upper()}\n"
                f"  {self._progress_bar(idx - 1, total)}\n"
                f"{'─' * 56}"
            )

            # ── Execute with retries ──────────────────────────────────
            outcome, current_data = self._run_step_with_retries(
                agent       = agent,
                step_index  = idx,
                total_steps = total,
                input_data  = current_data,
            )
            report.outcomes.append(outcome)

            # ── Notify memory ─────────────────────────────────────────
            self.memory.store_and_log_result(step_name, outcome.agent_result)
            self.memory.set_step_status(step_name, outcome.status)
            self.memory.set_current_data(current_data)
            self.memory.advance_step()

            # ── Notify code writer ────────────────────────────────────
            self.code_writer.observe(outcome.agent_result)

            # ── Abort check ───────────────────────────────────────────
            if not outcome.succeeded and self.abort_on_failure:
                self._logger.error(
                    f"Step '{step_name}' {outcome.status}. "
                    "Aborting pipeline."
                )
                aborted = True

            # ── Progress after step ───────────────────────────────────
            self._logger.info(
                f"  {self._progress_bar(idx, total)}"
            )

        # ── Finalise report ───────────────────────────────────────────
        total_elapsed   = time.perf_counter() - pipeline_start
        finished_at     = datetime.now(timezone.utc).isoformat()
        overall_success = (
            not aborted
            and all(o.succeeded for o in report.outcomes)
        )

        report.final_data       = current_data
        report.finished_at      = finished_at
        report.total_elapsed_s  = total_elapsed
        report.success          = overall_success

        # ── Log to memory and close script ────────────────────────────
        self.memory.log_pipeline_completion(total_elapsed, overall_success)
        self.code_writer.finalise(total_elapsed, overall_success)

        self._logger.pipeline_end(overall_success, total_elapsed)
        report.print_report()

        return report

    # ------------------------------------------------------------------
    # Step execution with retry loop
    # ------------------------------------------------------------------

    def _run_step_with_retries(
        self,
        agent:       DynamicAgent,
        step_index:  int,
        total_steps: int,
        input_data:  Any,
    ) -> tuple:
        """
        Execute one step with retry logic.

        Returns
        -------
        tuple[StepOutcome, Any]
            (outcome, data_for_next_step)
        """
        step_name  = agent.step_name
        step_start = time.perf_counter()
        attempt    = 0
        last_result: Dict[str, Any] = {}

        self._logger.step_start(
            step_name, f"{step_index}/{total_steps}"
        )
        self.memory.set_step_running(step_name)

        while attempt <= self.max_retries:
            attempt += 1

            # ── Backoff wait (skip on first attempt) ──────────────────
            if attempt > 1:
                wait = self.backoff_base ** (attempt - 1)
                self._logger.retry(step_name, attempt - 1, self.max_retries)
                self._logger.info(
                    f"  Waiting {wait:.1f}s before retry..."
                )
                time.sleep(wait)
                self.memory.increment_retry(step_name)

            # ── Call the agent ────────────────────────────────────────
            self._logger.agent_event(
                agent.agent_name,
                f"Attempt {attempt}/{self.max_retries + 1}"
            )

            try:
                result = agent.execute(input_data)
            except Exception as exc:
                # Catch any completely unhandled exception from the agent
                # (e.g. from _inspect_dataframe, _sanitise_code, or exec())
                # that escaped DynamicAgent's own try/except.
                # Convert to a failed result dict so the scheduler can
                # handle it cleanly instead of crashing the whole pipeline.
                elapsed = time.perf_counter() - step_start
                self._logger.error(
                    f"Unhandled exception in agent '{agent.agent_name}': {exc}"
                )
                result = {
                    "task_id":         "unhandled",
                    "agent_name":      agent.agent_name,
                    "step_name":       agent.step_name,
                    "input_summary":   str(type(input_data).__name__),
                    "output_data":     input_data,
                    "output_summary":  "[unhandled exception]",
                    "code_equivalent": "",
                    "reasoning":       f"Unhandled exception: {exc}",
                    "status":          "failed",
                    "error":           f"{type(exc).__name__}: {exc}",
                    "elapsed_ms":      elapsed * 1000,
                    "timestamp":       "",
                }

            last_result = result
            status  = result.get("status", "failed")
            elapsed = time.perf_counter() - step_start

            self._logger.step_end(
                step_name,
                f"{step_index}/{total_steps}",
                status,
                elapsed * 1000,
            )

            # ── Success ───────────────────────────────────────────────
            if status == "success":
                outcome = StepOutcome(
                    step_index   = step_index,
                    step_name    = step_name,
                    status       = "success",
                    agent_result = result,
                    attempts     = attempt,
                    elapsed_s    = elapsed,
                )
                return outcome, result["output_data"]

            # ── Retry requested ───────────────────────────────────────
            if status == "retry":
                if attempt <= self.max_retries:
                    continue
                break  # retries exhausted

            # ── Hard failure ──────────────────────────────────────────
            if status == "failed":
                self._logger.error(
                    f"Step '{step_name}' failed: "
                    f"{result.get('error', 'unknown error')}"
                )
                break

        # ── Exhausted / hard failed ───────────────────────────────────
        elapsed     = time.perf_counter() - step_start
        final_status = (
            "retry_exhausted"
            if last_result.get("status") == "retry"
            else "failed"
        )

        self._logger.error(
            f"Step '{step_name}' {final_status} "
            f"after {attempt} attempt(s) | {elapsed:.3f}s"
        )

        # Pass input_data through so pipeline can continue if configured
        fallback_result = {
            **last_result,
            "output_data": input_data,
            "status":      final_status,
        }

        outcome = StepOutcome(
            step_index   = step_index,
            step_name    = step_name,
            status       = final_status,
            agent_result = fallback_result,
            attempts     = attempt,
            elapsed_s    = elapsed,
        )
        return outcome, input_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_skipped_outcome(
        self,
        step_index: int,
        step_name:  str,
        reason:     str,
    ) -> StepOutcome:
        return StepOutcome(
            step_index   = step_index,
            step_name    = step_name,
            status       = "skipped",
            agent_result = {
                "step_name":       step_name,
                "status":          "skipped",
                "code_equivalent": "",
                "reasoning":       reason,
                "output_data":     None,
                "elapsed_ms":      0.0,
            },
            attempts       = 0,
            elapsed_s      = 0.0,
            skipped_reason = reason,
        )

    @staticmethod
    def _progress_bar(
        current: int,
        total:   int,
        width:   int = 28,
    ) -> str:
        filled = int(width * current / max(total, 1))
        bar    = "#" * filled + "-" * (width - filled)
        pct    = int(100 * current / max(total, 1))
        return f"[{bar}] {pct:3d}%  ({current}/{total} steps)"