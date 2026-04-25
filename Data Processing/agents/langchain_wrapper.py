"""
agents/langchain_wrapper.py
-----------------------------
LangChain-Compatible Wrapper for the Agentic Pipeline

Exposes each pipeline step as a LangChain Tool and the full
adaptive pipeline as a RunnableSequence / Chain.

Design
------
- Works WITH or WITHOUT langchain installed.
- If langchain is available: creates real BaseTool subclasses.
- If not: provides a compatible shim so the rest of the system
  works identically (duck-typed interface).

Usage
-----
    # As standalone tools
    from agents.langchain_wrapper import build_tools, run_tool_chain
    tools = build_tools(["remove_missing_values", "encode_categorical"])
    result = tools[0].invoke({"df": my_dataframe})

    # As a full pipeline chain
    chain = PipelineChain(csv_path="data/train.csv")
    output = chain.invoke({"objective": "classify survival"})
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd

from agents.base_agent import DynamicAgent, _inspect_dataframe, _execute_code
from agents.data_understanding_agent import (
    DataUnderstandingAgent, PipelineDecision,
    _analyse_data, _load_thresholds, _rule_based_decision,
)
from utils.logger import PipelineLogger

_logger = PipelineLogger("agents.langchain_wrapper")

# ---------------------------------------------------------------------------
# Try importing langchain — graceful fallback if not installed
# ---------------------------------------------------------------------------

_HAS_LANGCHAIN = False
try:
    from langchain_core.tools import BaseTool as _LCBaseTool
    from pydantic import BaseModel as _PydanticBase, Field as _Field
    _HAS_LANGCHAIN = True
    _logger.info("LangChain detected — using native BaseTool")
except ImportError:
    _logger.info("LangChain not installed — using compatible shim")
    _LCBaseTool = None
    _PydanticBase = None


# ---------------------------------------------------------------------------
# Tool Shim (when langchain is not installed)
# ---------------------------------------------------------------------------

class _ToolShim:
    """
    Minimal duck-typed replacement for langchain BaseTool.
    Provides .invoke(), .name, .description so callers don't
    need to know whether langchain is installed.
    """

    def __init__(self, name: str, description: str, func: Callable) -> None:
        self.name = name
        self.description = description
        self._func = func

    def invoke(self, input_data: Any) -> Any:
        return self._func(input_data)

    def run(self, input_data: Any) -> Any:
        return self.invoke(input_data)

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


# ---------------------------------------------------------------------------
# Pipeline Step Tool — wraps DynamicAgent as a LangChain Tool
# ---------------------------------------------------------------------------

def _make_step_tool(
    step_name: str,
    pipeline_steps: List[str],
    api_key: str = "",
    llm_model: str = "",
) -> Any:
    """
    Create a LangChain-compatible tool for a single pipeline step.

    Parameters
    ----------
    step_name : str
        The pipeline step (e.g. "remove_missing_values").
    pipeline_steps : list[str]
        Full pipeline context for the LLM.
    api_key : str
        Anthropic key (if using Anthropic backend).
    llm_model : str
        Model name override.

    Returns
    -------
    BaseTool (if langchain installed) or _ToolShim
    """
    agent = DynamicAgent(
        step_name=step_name,
        pipeline_steps=pipeline_steps,
        api_key=api_key,
        llm_model=llm_model,
    )

    def _execute(input_data: Any) -> Dict[str, Any]:
        """Execute the pipeline step on input data."""
        if isinstance(input_data, dict):
            df = input_data.get("df", input_data)
        else:
            df = input_data
        return agent.execute(df)

    if _HAS_LANGCHAIN and _LCBaseTool is not None:
        # Build a real LangChain tool via the decorator pattern
        from langchain_core.tools import tool as _tool_decorator

        @_tool_decorator
        def pipeline_step(input_json: str) -> str:
            """Execute a pipeline step. Input: JSON with 'csv_path' or 'data'."""
            try:
                params = json.loads(input_json)
                if "csv_path" in params:
                    df = pd.read_csv(params["csv_path"], encoding="utf-8")
                else:
                    df = pd.DataFrame(params.get("data", []))
                result = agent.execute(df)
                return json.dumps({
                    "status": result.get("status"),
                    "reasoning": result.get("reasoning", ""),
                    "summary": result.get("output_summary", ""),
                }, default=str)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)})

        pipeline_step.name = f"pipeline_{step_name}"
        pipeline_step.description = (
            f"Execute the '{step_name}' step of the ML pipeline. "
            f"Accepts a DataFrame and returns the processed result."
        )
        # Attach the raw executor for programmatic use
        pipeline_step._raw_execute = _execute
        return pipeline_step
    else:
        return _ToolShim(
            name=f"pipeline_{step_name}",
            description=f"Execute '{step_name}' pipeline step on a DataFrame.",
            func=_execute,
        )


# ---------------------------------------------------------------------------
# Public API: build_tools
# ---------------------------------------------------------------------------

def build_tools(
    pipeline_steps: List[str],
    api_key: str = "",
    llm_model: str = "",
) -> List[Any]:
    """
    Build LangChain-compatible tools for each pipeline step.

    Parameters
    ----------
    pipeline_steps : list[str]
        Step names to create tools for.
    api_key : str
        API key (for Anthropic backend).
    llm_model : str
        LLM model name.

    Returns
    -------
    list[BaseTool | _ToolShim]
        One tool per step, in order.

    Example
    -------
        tools = build_tools(["remove_missing_values", "encode_categorical"])
        for tool in tools:
            print(tool.name, tool.description)
    """
    tools = []
    for step in pipeline_steps:
        tool = _make_step_tool(step, pipeline_steps, api_key, llm_model)
        tools.append(tool)
        _logger.info(f"Built tool: {tool.name}")
    return tools


# ---------------------------------------------------------------------------
# Pipeline Chain — full adaptive pipeline as a chain
# ---------------------------------------------------------------------------

@dataclass
class PipelineChainResult:
    """Result from a PipelineChain execution."""
    success: bool
    pipeline_id: str
    steps_executed: List[str]
    steps_skipped: List[str]
    final_data: Any
    metrics: Dict[str, Any]
    elapsed_s: float
    decision: Optional[PipelineDecision] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "pipeline_id": self.pipeline_id,
            "steps_executed": self.steps_executed,
            "steps_skipped": self.steps_skipped,
            "elapsed_s": round(self.elapsed_s, 3),
            "metrics": self.metrics,
        }


class PipelineChain:
    """
    LangChain-compatible chain that runs the full adaptive pipeline.

    Each step is a separate tool, chained sequentially.
    The chain auto-detects which steps are needed based on data.

    Works as a LangChain Runnable (has .invoke()) or standalone.

    Parameters
    ----------
    csv_path : str
        Path to the input CSV file.
    target_column : str
        Target column name (auto-inferred if empty).
    api_key : str
        LLM API key.
    llm_model : str
        LLM model name.
    config_path : str
        Path to pipeline.yaml.
    """

    def __init__(
        self,
        csv_path: str = "",
        target_column: str = "",
        api_key: str = "",
        llm_model: str = "",
        config_path: str = "config/pipeline.yaml",
    ) -> None:
        self.csv_path = csv_path
        self.target_column = target_column
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.llm_model = llm_model or os.environ.get("OLLAMA_MODEL", "")
        self.config_path = config_path
        self._logger = PipelineLogger("PipelineChain")
        self._last_result: Optional[PipelineChainResult] = None
        self._step_cache: Dict[str, Dict[str, Any]] = {}

    def invoke(self, input_data: Optional[Dict[str, Any]] = None) -> PipelineChainResult:
        """
        Execute the full adaptive pipeline.

        Parameters
        ----------
        input_data : dict, optional
            Override inputs: {"csv_path": "...", "target_column": "...",
                              "objective": "classify ..."}

        Returns
        -------
        PipelineChainResult
        """
        start = time.perf_counter()
        input_data = input_data or {}

        csv_path = input_data.get("csv_path", self.csv_path)
        target_col = input_data.get("target_column", self.target_column)

        if not csv_path:
            raise ValueError("No csv_path provided. Set it in constructor or input.")

        # Phase 1: Load data
        self._logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path, encoding="utf-8")
        self._logger.info(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

        # Phase 1: Understand data
        thresholds = _load_thresholds(self.config_path)
        profile = _analyse_data(df, thresholds, target_col)
        target_col = profile["target_column"]

        # Decide steps
        decision_dict = _rule_based_decision(profile, self._logger)
        decision = PipelineDecision(
            problem_type=decision_dict["problem_type"],
            target_column=decision_dict.get("target_column", target_col),
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

        # Phase 2: Build tools and execute
        tools = build_tools(adaptive_steps, self.api_key, self.llm_model)

        steps_executed = []
        steps_skipped = list(decision.skipped.keys())
        current_data = df

        for tool in tools:
            step_name = tool.name.replace("pipeline_", "")
            self._logger.info(f"Executing: {step_name}")
            try:
                if hasattr(tool, "_raw_execute"):
                    result = tool._raw_execute(current_data)
                else:
                    result = tool.invoke(current_data)

                if isinstance(result, dict) and result.get("status") == "success":
                    current_data = result["output_data"]
                    steps_executed.append(step_name)
                    self._step_cache[step_name] = result
                else:
                    self._logger.warning(f"Step {step_name} failed")
                    steps_executed.append(f"{step_name} [FAILED]")
            except Exception as e:
                self._logger.error(f"Step {step_name} error: {e}")
                steps_executed.append(f"{step_name} [ERROR]")

        elapsed = time.perf_counter() - start
        pipeline_id = f"chain-{datetime.now(timezone.utc).strftime('%H%M%S')}"

        result = PipelineChainResult(
            success=all("[" not in s for s in steps_executed),
            pipeline_id=pipeline_id,
            steps_executed=steps_executed,
            steps_skipped=steps_skipped,
            final_data=current_data,
            metrics={},
            elapsed_s=elapsed,
            decision=decision,
        )
        self._last_result = result
        return result

    def get_tools(self) -> List[Any]:
        """Return the tools without executing (for LangChain AgentExecutor)."""
        if not self.csv_path:
            return build_tools(
                ["remove_missing_values", "encode_categorical",
                 "normalize_features", "select_and_train_models"],
                self.api_key, self.llm_model,
            )
        df = pd.read_csv(self.csv_path, encoding="utf-8")
        thresholds = _load_thresholds(self.config_path)
        profile = _analyse_data(df, thresholds, self.target_column)
        decision_dict = _rule_based_decision(profile, self._logger)
        steps = [s for s in decision_dict["steps"] if s != "load_dataset"]
        return build_tools(steps, self.api_key, self.llm_model)
