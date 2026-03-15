"""
memory/memory_manager.py
------------------------
Two-layer memory system for the Agentic Pipeline Builder.

LAYER 1 — ShortTermMemory
    Stores the *live* pipeline execution state: which step is running,
    what data is currently flowing between agents, retry counters, etc.
    Lives entirely in RAM. Wiped when the process ends.

LAYER 2 — LongTermMemory
    Persists logs and intermediate outputs to disk as JSON Lines files
    inside the ``memory_store/`` directory.  Acts as the audit trail and
    allows post-run inspection of every agent's input/output.

COMBINED — MemoryManager
    Facade that exposes a single, unified API used by the orchestrator,
    scheduler, and agents.  Neither layer is exposed directly.

Design notes
------------
* No external dependencies (Redis, ChromaDB) are required.  The prototype
  uses plain Python dicts + JSON files so the system runs out-of-the-box.
* The interface is intentionally generic so you can swap the backend
  (e.g. replace _LongTermMemory with a Redis/ChromaDB adapter) without
  changing any caller code.
* Thread-safety: a ``threading.Lock`` guards every write so the memory
  store is safe if parallel step execution is added later.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import PipelineLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STORE_DIR = Path("memory_store")
_AGENT_LOG_FILE   = _STORE_DIR / "agent_outputs.jsonl"
_PIPELINE_LOG_FILE = _STORE_DIR / "pipeline_runs.jsonl"

# ---------------------------------------------------------------------------
# Short-Term Memory
# ---------------------------------------------------------------------------

class ShortTermMemory:
    """
    In-RAM store for the currently executing pipeline.

    Stores
    ------
    - pipeline_id          : unique run identifier
    - steps                : ordered list of step names
    - current_step_index   : which step is active
    - step_statuses        : per-step status dict  {"step": "pending|running|success|failed"}
    - retry_counts         : per-step retry counter
    - current_data         : the data object flowing between agents (any type)
    - step_results         : full result dicts keyed by step name
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}
        self._logger = PipelineLogger(__name__ + ".ShortTermMemory")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_pipeline(self, steps: List[str]) -> str:
        """
        Initialise state for a new pipeline run.

        Parameters
        ----------
        steps:
            Ordered list of pipeline step names.

        Returns
        -------
        str
            A unique pipeline_id for this run.
        """
        pipeline_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._state = {
                "pipeline_id": pipeline_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "steps": steps,
                "current_step_index": 0,
                "step_statuses": {s: "pending" for s in steps},
                "retry_counts": {s: 0 for s in steps},
                "current_data": None,
                "step_results": {},
            }
        self._logger.debug(f"Short-term memory initialised — pipeline_id={pipeline_id}")
        return pipeline_id

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def set_step_running(self, step_name: str) -> None:
        """Mark a step as currently running."""
        with self._lock:
            self._state["step_statuses"][step_name] = "running"
        self._logger.debug(f"Step '{step_name}' → RUNNING")

    def set_step_status(self, step_name: str, status: str) -> None:
        """
        Update the status of a step.

        Parameters
        ----------
        step_name:
            The pipeline step name.
        status:
            One of ``"pending"``, ``"running"``, ``"success"``, ``"failed"``.
        """
        with self._lock:
            self._state["step_statuses"][step_name] = status

    def increment_retry(self, step_name: str) -> int:
        """
        Increment the retry counter for a step.

        Returns
        -------
        int
            Updated retry count.
        """
        with self._lock:
            self._state["retry_counts"][step_name] += 1
            count = self._state["retry_counts"][step_name]
        self._logger.debug(f"Retry count for '{step_name}' → {count}")
        return count

    def advance_step(self) -> None:
        """Move the current step index forward by one."""
        with self._lock:
            self._state["current_step_index"] += 1

    # ------------------------------------------------------------------
    # Data flow
    # ------------------------------------------------------------------

    def set_current_data(self, data: Any) -> None:
        """Store the latest data object (output of the last agent)."""
        with self._lock:
            self._state["current_data"] = data

    def get_current_data(self) -> Any:
        """Retrieve the latest data object."""
        with self._lock:
            return self._state.get("current_data")

    def store_step_result(self, step_name: str, result: Dict[str, Any]) -> None:
        """
        Persist the full result dict from an agent for later retrieval.

        Parameters
        ----------
        step_name:
            The pipeline step name.
        result:
            The dict returned by the agent's ``execute()`` method.
        """
        with self._lock:
            self._state["step_results"][step_name] = deepcopy(result)

    def get_step_result(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Return the stored result for a step, or None if not yet run."""
        with self._lock:
            return self._state["step_results"].get(step_name)

    # ------------------------------------------------------------------
    # Snapshot / read
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a deep-copy snapshot of the entire state dict.
        Safe to inspect without holding the lock.
        """
        with self._lock:
            snap = deepcopy(self._state)
        # Remove the live data object from the snapshot to keep it serialisable
        snap.pop("current_data", None)
        return snap

    def get_pipeline_id(self) -> str:
        """Return the current pipeline run ID."""
        return self._state.get("pipeline_id", "unknown")

    def get_steps(self) -> List[str]:
        """Return the ordered list of steps."""
        return self._state.get("steps", [])

    def get_current_step_index(self) -> int:
        """Return the index of the currently active step."""
        return self._state.get("current_step_index", 0)

    def get_retry_count(self, step_name: str) -> int:
        """Return the retry count for a step."""
        return self._state.get("retry_counts", {}).get(step_name, 0)

    def all_step_statuses(self) -> Dict[str, str]:
        """Return a copy of all step statuses."""
        with self._lock:
            return dict(self._state.get("step_statuses", {}))


# ---------------------------------------------------------------------------
# Long-Term Memory
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Disk-backed store for audit logs and intermediate agent outputs.

    Uses JSON Lines (one JSON object per line) so that:
    - appends are O(1) — no need to rewrite the whole file
    - every line is independently parseable
    - files can be inspected with ``cat`` / ``jq`` without special tools

    Files
    -----
    memory_store/agent_outputs.jsonl
        One record per agent execution, containing the full result dict.

    memory_store/pipeline_runs.jsonl
        One record per pipeline run (written at the end of the run).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._logger = PipelineLogger(__name__ + ".LongTermMemory")
        _STORE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        """Append a single JSON record to a .jsonl file."""
        record["_written_at"] = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def log_agent_output(
        self,
        pipeline_id: str,
        step_name: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Persist the result dict from one agent execution.

        Parameters
        ----------
        pipeline_id:
            The run ID from ShortTermMemory.
        step_name:
            The pipeline step name.
        result:
            The agent's ``execute()`` return value.
        """
        record = {
            "pipeline_id": pipeline_id,
            "step_name": step_name,
            "result": result,
        }
        self._append_jsonl(_AGENT_LOG_FILE, record)
        self._logger.debug(f"Agent output logged for step '{step_name}'")

    def log_pipeline_run(
        self,
        pipeline_id: str,
        steps: List[str],
        step_statuses: Dict[str, str],
        elapsed_s: float,
        success: bool,
    ) -> None:
        """
        Write a summary record for the entire pipeline run.

        Parameters
        ----------
        pipeline_id:
            The run ID.
        steps:
            Ordered list of step names.
        step_statuses:
            Final status of each step.
        elapsed_s:
            Wall-clock seconds the run took.
        success:
            Whether the pipeline completed without error.
        """
        record = {
            "pipeline_id": pipeline_id,
            "steps": steps,
            "step_statuses": step_statuses,
            "elapsed_seconds": round(elapsed_s, 3),
            "success": success,
        }
        self._append_jsonl(_PIPELINE_LOG_FILE, record)
        self._logger.debug(f"Pipeline run summary logged — id={pipeline_id}")

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def _read_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Read all records from a .jsonl file."""
        if not path.exists():
            return []
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        self._logger.warning(f"Skipping malformed record in {path}")
        return records

    def get_agent_outputs(
        self,
        pipeline_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored agent outputs, optionally filtered by pipeline_id.
        """
        records = self._read_jsonl(_AGENT_LOG_FILE)
        if pipeline_id:
            records = [r for r in records if r.get("pipeline_id") == pipeline_id]
        return records

    def get_pipeline_runs(self) -> List[Dict[str, Any]]:
        """Retrieve all pipeline run summaries."""
        return self._read_jsonl(_PIPELINE_LOG_FILE)

    def get_last_pipeline_run(self) -> Optional[Dict[str, Any]]:
        """Return the most recent pipeline run summary, or None."""
        runs = self.get_pipeline_runs()
        return runs[-1] if runs else None


# ---------------------------------------------------------------------------
# MemoryManager — unified facade
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Single entry point for all memory operations.

    Combines :class:`ShortTermMemory` (RAM) and :class:`LongTermMemory`
    (disk) behind one API.  All orchestrator / agent / scheduler code
    should import and use this class only.

    Parameters
    ----------
    None — instantiate once, pass the instance around.

    Example
    -------
    .. code-block:: python

        mm = MemoryManager()
        pid = mm.init_pipeline(["load_dataset", "normalize_features"])
        mm.set_step_running("load_dataset")
        mm.set_current_data(df)
        mm.store_and_log_result("load_dataset", agent_result)
        mm.set_step_status("load_dataset", "success")
        mm.advance_step()
    """

    def __init__(self) -> None:
        self._st  = ShortTermMemory()
        self._lt  = LongTermMemory()
        self._logger = PipelineLogger(__name__ + ".MemoryManager")
        self._logger.info("MemoryManager initialised (short-term=RAM, long-term=disk/jsonl)")

    # ------------------------------------------------------------------
    # Delegated short-term operations
    # ------------------------------------------------------------------

    def init_pipeline(self, steps: List[str]) -> str:
        """Initialise both memory layers for a new pipeline run."""
        pid = self._st.init_pipeline(steps)
        return pid

    def set_step_running(self, step_name: str) -> None:
        self._st.set_step_running(step_name)

    def set_step_status(self, step_name: str, status: str) -> None:
        self._st.set_step_status(step_name, status)

    def increment_retry(self, step_name: str) -> int:
        return self._st.increment_retry(step_name)

    def advance_step(self) -> None:
        self._st.advance_step()

    def set_current_data(self, data: Any) -> None:
        self._st.set_current_data(data)

    def get_current_data(self) -> Any:
        return self._st.get_current_data()

    def get_step_result(self, step_name: str) -> Optional[Dict[str, Any]]:
        return self._st.get_step_result(step_name)

    def snapshot(self) -> Dict[str, Any]:
        return self._st.snapshot()

    def get_pipeline_id(self) -> str:
        return self._st.get_pipeline_id()

    def get_steps(self) -> List[str]:
        return self._st.get_steps()

    def get_current_step_index(self) -> int:
        return self._st.get_current_step_index()

    def get_retry_count(self, step_name: str) -> int:
        return self._st.get_retry_count(step_name)

    def all_step_statuses(self) -> Dict[str, str]:
        return self._st.all_step_statuses()

    # ------------------------------------------------------------------
    # Combined store + log (most-used method by agents)
    # ------------------------------------------------------------------

    def store_and_log_result(
        self,
        step_name: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Store a step result in short-term memory **and** persist it to
        the long-term log in one call.

        Parameters
        ----------
        step_name:
            The pipeline step name.
        result:
            The agent's ``execute()`` return value.
        """
        self._st.store_step_result(step_name, result)
        self._lt.log_agent_output(
            pipeline_id=self._st.get_pipeline_id(),
            step_name=step_name,
            result=result,
        )

    # ------------------------------------------------------------------
    # Pipeline-level long-term logging
    # ------------------------------------------------------------------

    def log_pipeline_completion(self, elapsed_s: float, success: bool) -> None:
        """
        Write the pipeline run summary to the long-term store.

        Parameters
        ----------
        elapsed_s:
            Total wall-clock time for the run.
        success:
            Overall success flag.
        """
        self._lt.log_pipeline_run(
            pipeline_id=self._st.get_pipeline_id(),
            steps=self._st.get_steps(),
            step_statuses=self._st.all_step_statuses(),
            elapsed_s=elapsed_s,
            success=success,
        )

    # ------------------------------------------------------------------
    # Retrieval helpers (for reporting / testing)
    # ------------------------------------------------------------------

    def get_all_agent_outputs(
        self,
        pipeline_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return logged agent outputs, optionally filtered by run ID."""
        pid = pipeline_id or self._st.get_pipeline_id()
        return self._lt.get_agent_outputs(pid)

    def get_last_pipeline_run(self) -> Optional[Dict[str, Any]]:
        """Return the most recent pipeline run summary."""
        return self._lt.get_last_pipeline_run()

    def print_summary(self) -> None:
        """Print a human-readable summary of the current pipeline state."""
        snap = self.snapshot()
        print("\n" + "=" * 60)
        print(f"  PIPELINE MEMORY SNAPSHOT  (id={snap.get('pipeline_id')})")
        print("=" * 60)
        print(f"  Started      : {snap.get('started_at')}")
        print(f"  Current step : {snap.get('current_step_index')} / {len(snap.get('steps', []))}")
        print()
        statuses = snap.get("step_statuses", {})
        for step in snap.get("steps", []):
            icon = {"pending": "○", "running": "◉", "success": "✔", "failed": "✘"}.get(
                statuses.get(step, "pending"), "?"
            )
            retries = snap.get("retry_counts", {}).get(step, 0)
            retry_str = f"  (retried {retries}x)" if retries > 0 else ""
            print(f"    {icon}  {step:<35} {statuses.get(step, 'pending').upper()}{retry_str}")
        print("=" * 60 + "\n")