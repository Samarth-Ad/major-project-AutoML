"""
agents/base_agent.py
--------------------
TRUE Agentic Base — LLM-powered, zero hardcoded logic.
Supports BOTH Anthropic Claude AND local Ollama models.

LLM Backend selection
---------------------
Set the backend via environment variable or constructor param:

    # Use Ollama (local, no API key needed)
    export LLM_BACKEND=ollama
    export OLLAMA_MODEL=llama3.1:70b          # or any model you pulled
    export OLLAMA_BASE_URL=http://localhost:11434

    # Use Anthropic Claude (cloud, needs API key)
    export LLM_BACKEND=anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

How it works
------------
1.  DynamicAgent.execute(df) is called by the Scheduler
2.  _inspect_dataframe(df) builds a full schema dict
3.  _call_llm() routes to:
        _call_ollama()     if backend == "ollama"
        _call_anthropic()  if backend == "anthropic"
4.  LLM returns JSON: {reasoning, code, validation, summary}
5.  _execute_code() runs the code via exec() on the real df
6.  Result dict is returned to Scheduler
"""

from __future__ import annotations

import json
import os
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Non-recoverable agent failure."""

class AgentRetryError(AgentError):
    """Recoverable failure — scheduler should retry."""

class AgentValidationError(AgentError):
    """Input data failed precondition check."""

class LLMCodeError(AgentError):
    """LLM returned code that failed to execute."""


# ---------------------------------------------------------------------------
# Schema inspector
# ---------------------------------------------------------------------------

def _inspect_dataframe(df: Any) -> dict:
    """
    Build a rich schema summary of a DataFrame for the LLM prompt.
    Includes dtypes, null counts, value distributions, and sample rows.

    Handles all column types safely including:
    - bool / uint8  (from one-hot encoding — describe() returns different keys)
    - float64 / int64 (normal numeric — full stats)
    - object / category (categorical — top value counts)
    """
    if df is None:
        return {
            "type": "None",
            "description": "No data yet — this is the first step in the pipeline."
        }

    if not isinstance(df, pd.DataFrame):
        # Handle _ExecutionResult wrapper (df + trained model)
        if hasattr(df, 'df') and isinstance(df.df, pd.DataFrame):
            df = df.df
        else:
            return {
                "type": type(df).__name__,
                "description": str(df)[:500]
            }

    schema = {
        "type":        "pandas.DataFrame",
        "shape":       list(df.shape),
        "columns":     [],
        "total_nulls": int(df.isnull().sum().sum()),
        "sample_rows": json.loads(
            df.head(3).to_json(orient="records", default_handler=str)
        ),
    }

    for col in df.columns:
        col_info: dict = {
            "name":     col,
            "dtype":    str(df[col].dtype),
            "nulls":    int(df[col].isnull().sum()),
            "null_pct": round(float(df[col].isnull().mean()) * 100, 2),
            "n_unique": int(df[col].nunique()),
        }

        # ── Determine how to summarise this column ────────────────────
        is_bool   = df[col].dtype == bool or str(df[col].dtype) == "bool"
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) and not is_bool

        if is_bool:
            # bool columns: show value counts (True/False frequencies)
            col_info["top_values"] = {
                str(k): int(v)
                for k, v in df[col].value_counts().head(5).items()
            }

        elif is_numeric:
            # Safe describe() — use .get() so missing keys never crash
            # Some dtypes (uint8 from one-hot) may not have all stats
            desc = df[col].describe()
            col_info["stats"] = {
                "min":  round(float(desc.get("min",  desc.get("0%",  0))), 4),
                "max":  round(float(desc.get("max",  desc.get("100%", 0))), 4),
                "mean": round(float(desc.get("mean", 0)), 4),
                "std":  round(float(desc.get("std",  0)), 4),
            }

        else:
            # Categorical / object columns: top value frequencies
            col_info["top_values"] = {
                str(k): int(v)
                for k, v in df[col].value_counts().head(5).items()
            }

        schema["columns"].append(col_info)

    return schema


# ---------------------------------------------------------------------------
# Shared prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    step_name:        str,
    df_schema:        dict,
    pipeline_context: list,
) -> tuple[str, str]:
    """
    Build system + user prompt strings used by both backends.

    Returns
    -------
    tuple[str, str]   (system_prompt, user_prompt)
    """
    pipeline_str = " -> ".join(pipeline_context)
    schema_str   = json.dumps(df_schema, indent=2)

    system_prompt = (
        "You are an expert data scientist embedded inside an agentic ML pipeline.\n"
        "Your job: inspect real DataFrame schemas and write the BEST Python code "
        "to execute a specific pipeline step on that exact data.\n\n"
        "STRICT RULES:\n"
        "1. Inspect the ACTUAL column names, dtypes, nulls, and value distributions.\n"
        "2. Choose strategies based on the real data — not generic defaults.\n"
        "3. The DataFrame variable is always named `df`.\n"
        "4. All imports must be inside your code block.\n"
        "5. Code must be complete and executable via Python exec().\n"
        "6. ALWAYS keep `df` as a pandas DataFrame at the end.\n"
        "7. For train_model step: store the fitted model in a variable called"
        " `trained_model` (e.g. trained_model = model). This is required.\n"
        "8. Return ONLY valid JSON — no markdown fences, no explanation outside JSON.\n\n"
        "PANDAS RULES (these will cause crashes if violated):\n"
        "8. For pd.read_csv: NEVER combine engine='python' with low_memory=False.\n"
        "   Use EITHER:  pd.read_csv(path, low_memory=False)          [c engine]\n"
        "   OR:          pd.read_csv(path, engine='python')           [python engine]\n"
        "   NEVER both together. Default (no engine arg) is safest.\n"
        "9. ALWAYS open files with encoding='utf-8' on Windows to avoid codec errors.\n"
        "   Correct:  pd.read_csv(path, encoding='utf-8')\n"
        "10. Use forward slashes or raw strings for file paths:\n"
        "    Correct:  pd.read_csv(r'data\\train.csv')  or  pd.read_csv('data/train.csv')\n"
        "11. After pd.read_csv, ALWAYS verify: assert isinstance(df, pd.DataFrame)\n\n"
        "RETURN FORMAT (strict JSON only, no extra text):\n"
        "{\n"
        '  "reasoning": "Why this approach for THIS specific data",\n'
        '  "code": "complete Python as single string with \\\\n for newlines",\n'
        '  "validation": "what check confirms this step worked",\n'
        '  "summary": "one-line description of what the code does"\n'
        "}"
    )

    user_prompt = (
        f"Full pipeline: {pipeline_str}\n"
        f"Current step: \"{step_name}\"\n\n"
        f"DataFrame schema:\n{schema_str}\n\n"
        f"Write the BEST Python code to execute step \"{step_name}\" "
        f"on this exact DataFrame.\n"
        "Base EVERY decision on the actual schema — column names, "
        "dtypes, null counts, value distributions.\n"
        "Return ONLY the JSON object, nothing else."
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# JSON response parser (shared)
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_text: str, logger: PipelineLogger) -> dict:
    """
    Parse and validate the LLM JSON response.
    Strips markdown fences if present.
    Raises ValueError if required keys are missing.
    """
    text = raw_text.strip()

    # Strip markdown fences  ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",           "", text)
    text = text.strip()

    # Sometimes models wrap with extra prose — try to extract JSON block
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)

    parsed = json.loads(text)

    required = {"reasoning", "code", "validation", "summary"}
    missing  = required - set(parsed.keys())
    if missing:
        raise ValueError(f"LLM response missing required keys: {missing}")

    logger.info(f"LLM reasoning: {parsed['reasoning'][:300]}")
    return parsed


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _call_ollama(
    step_name:        str,
    df_schema:        dict,
    pipeline_context: list,
    logger:           PipelineLogger,
    model:            str   = "gpt-oss:120b-cloud",
    base_url:         str   = "http://localhost:11434",
    max_retries:      int   = 3,
) -> dict:
    """
    Call an Ollama model via /api/chat — works for both local and
    cloud-routed models like gpt-oss:120b-cloud.

    Uses the same pattern as strategic_lm.py:
        POST /api/chat
        body: { model, stream, options, messages }
        response: { message: { role, content } }

    Parameters
    ----------
    model    : pulled Ollama model, e.g. "gpt-oss:120b-cloud",
               "llama3.1:70b", "llama3.1:8b", "qwen2.5:32b"
    base_url : Ollama server URL (default: http://localhost:11434)
    """
    import urllib.request
    import urllib.error

    system_prompt, user_prompt = _build_prompt(
        step_name, df_schema, pipeline_context
    )

    endpoint = f"{base_url.rstrip('/')}/api/chat"

    # ── Payload follows strategic_lm.py pattern exactly ──────────────
    # temperature inside options (not top-level)
    # no num_predict / top_p for cloud models — they are silently ignored
    # stream=False to get full response at once
    payload = json.dumps({
        "model":  model,
        "stream": False,
        "options": {
            "temperature": 0.2,    # matches strategic_lm.py value
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(
                f"Ollama /api/chat attempt {attempt}/{max_retries} "
                f"| model={model} | endpoint={endpoint}"
            )

            req = urllib.request.Request(
                endpoint,
                data    = payload,
                headers = headers,
                method  = "POST",
            )

            # 180s timeout — cloud-routed models (gpt-oss:120b-cloud)
            # need more time than local models
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            # /api/chat response:
            # { "message": { "role": "assistant", "content": "..." } }
            content = raw["message"]["content"]
            return _parse_llm_response(content, logger)

        except urllib.error.URLError as exc:
            last_error = exc
            err_str = str(exc)
            # Give specific, actionable error messages
            if "Connection refused" in err_str:
                raise AgentError(
                    f"Cannot connect to Ollama at {base_url}.\n"
                    "  Start Ollama with:   ollama serve\n"
                    f"  Pull the model with: ollama pull {model}"
                ) from exc
            if "timed out" in err_str or "TimeoutError" in err_str:
                logger.warning(
                    f"Ollama request timed out on attempt {attempt}. "
                    "Cloud models may be slow — will retry."
                )
            else:
                logger.warning(f"Ollama URLError attempt {attempt}: {exc}")

        except KeyError as exc:
            # Response shape was unexpected — log the raw response
            last_error = exc
            logger.warning(
                f"Unexpected response shape on attempt {attempt}: "
                f"missing key {exc}. "
                f"Raw keys: {list(raw.keys()) if 'raw' in dir() else 'unknown'}"
            )

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                f"JSON parse error on attempt {attempt}: {exc}"
            )

        except Exception as exc:
            last_error = exc
            logger.warning(f"Unexpected error on attempt {attempt}: {exc}")

        if attempt < max_retries:
            wait = 2 ** attempt
            logger.info(f"Retrying in {wait}s ...")
            time.sleep(wait)

    raise AgentError(
        f"Ollama call failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Anthropic backend (kept as fallback)
# ---------------------------------------------------------------------------

def _call_anthropic(
    step_name:        str,
    df_schema:        dict,
    pipeline_context: list,
    logger:           PipelineLogger,
    api_key:          str = "",
    model:            str = "claude-sonnet-4-20250514",
    max_retries:      int = 3,
) -> dict:
    """
    Call the Anthropic Claude API.
    Used when LLM_BACKEND=anthropic (the cloud fallback).
    """
    import urllib.request
    import urllib.error

    system_prompt, user_prompt = _build_prompt(
        step_name, df_schema, pipeline_context
    )

    payload = json.dumps({
        "model":      model,
        "max_tokens": 2000,
        "system":     system_prompt,
        "messages":   [{"role": "user", "content": user_prompt}],
    }).encode("utf-8")

    headers = {
        "Content-Type":      "application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data    = payload,
                headers = headers,
                method  = "POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            content = raw["content"][0]["text"]
            return _parse_llm_response(content, logger)

        except Exception as exc:
            last_error = exc
            logger.warning(f"Anthropic attempt {attempt} failed: {exc}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    raise AgentError(
        f"Anthropic call failed after {max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# Unified LLM router
# ---------------------------------------------------------------------------

def _call_llm(
    step_name:        str,
    df_schema:        dict,
    pipeline_context: list,
    logger:           PipelineLogger,
    api_key:          str = "",
    model:            str = "",
    max_retries:      int = 3,
) -> dict:
    """
    Route the LLM call to the correct backend based on LLM_BACKEND env var.

    Environment variables
    ---------------------
    LLM_BACKEND        : "ollama" (default) or "anthropic"
    OLLAMA_MODEL       : model name  (default: "llama3.1:70b")
    OLLAMA_BASE_URL    : server URL  (default: "http://localhost:11434")
    ANTHROPIC_API_KEY  : required only when LLM_BACKEND=anthropic
    """
    backend = os.environ.get("LLM_BACKEND", "ollama").lower().strip()

    if backend == "ollama":
        ollama_model = (
            model
            or os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
        )
        ollama_url = os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        logger.info(
            f"[LLM Router] Backend=OLLAMA | "
            f"model={ollama_model} | url={ollama_url}"
        )
        return _call_ollama(
            step_name        = step_name,
            df_schema        = df_schema,
            pipeline_context = pipeline_context,
            logger           = logger,
            model            = ollama_model,
            base_url         = ollama_url,
            max_retries      = max_retries,
        )

    elif backend == "anthropic":
        anthropic_model = (
            model
            or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        logger.info(
            f"[LLM Router] Backend=ANTHROPIC | model={anthropic_model}"
        )
        return _call_anthropic(
            step_name        = step_name,
            df_schema        = df_schema,
            pipeline_context = pipeline_context,
            logger           = logger,
            api_key          = key,
            model            = anthropic_model,
            max_retries      = max_retries,
        )

    else:
        raise AgentError(
            f"Unknown LLM_BACKEND='{backend}'. "
            "Valid options: 'ollama' or 'anthropic'.\n"
            "Set with:  export LLM_BACKEND=ollama"
        )


def _sanitise_code(code: str, logger: PipelineLogger) -> str:
    """
    Auto-fix known bad patterns that LLMs occasionally generate.
    Acts as a safety net even when the system prompt is followed correctly.

    Fixes applied
    -------------
    1. engine='python' + low_memory=False  → remove engine='python'
       (these two args are mutually exclusive in pandas)
    2. read_csv without encoding            → add encoding='utf-8'
       (prevents Windows cp1252 UnicodeDecodeError)
    3. Backslash file paths without raw string → convert to forward slashes
    """
    original = code

    # ── Fix 1: remove engine='python' when low_memory is also present ──
    if "low_memory" in code and "engine='python'" in code:
        code = re.sub(r",?\s*engine=['\"]python['\"]", "", code)
        logger.debug("Sanitiser: removed engine='python' (conflicts with low_memory)")

    if "low_memory" in code and 'engine="python"' in code:
        code = re.sub(r',?\s*engine="python"', "", code)
        logger.debug("Sanitiser: removed engine=\"python\" (conflicts with low_memory)")

    # ── Fix 2: add encoding='utf-8' to read_csv if missing ─────────────
    # Matches pd.read_csv(...) calls that don't already have encoding=
    if "read_csv" in code and "encoding=" not in code:
        # Insert encoding='utf-8' as the last argument before closing paren
        # Handles single-line and multi-line read_csv calls
        code = re.sub(
            r"(pd\.read_csv\s*\([^)]*?)(\s*\))",
            lambda m: m.group(1) + ",\n    encoding='utf-8'" + m.group(2),
            code,
        )
        logger.debug("Sanitiser: added encoding='utf-8' to read_csv")

    if code != original:
        logger.info("Sanitiser applied fixes to LLM-generated code before execution")

    return code


# ---------------------------------------------------------------------------
# Safe code executor
# ---------------------------------------------------------------------------

def _execute_code(
    code:   str,
    df:     Any,
    logger: PipelineLogger,
) -> Any:
    """
    Execute LLM-generated Python code in a sandboxed namespace.
    Returns a dict with:
        df            - the processed DataFrame (always present)
        trained_model - the trained model object (present after train_model step)
    Falls back to returning just the DataFrame for backward compatibility.
    """
    code = _sanitise_code(code, logger)

    namespace: dict = {
        "df":            df.copy() if isinstance(df, pd.DataFrame) else df,
        "pd":            pd,
        "np":            np,
        "print":         print,
        "__builtins__":  __builtins__,
    }

    for lib_name in ("sklearn", "scipy", "joblib"):
        try:
            import importlib
            namespace[lib_name] = importlib.import_module(lib_name)
        except ImportError:
            pass

    logger.debug(f"Executing LLM code ({len(code)} chars) ...")

    try:
        exec(compile(code, "<llm_generated>", "exec"), namespace)
    except Exception as exc:
        raise LLMCodeError(
            f"Execution error: {exc}\n"
            f"--- Generated Code ---\n{code}\n"
            f"--- Traceback ---\n{traceback.format_exc()}"
        ) from exc

    result_df = namespace.get("df")
    if result_df is None:
        raise LLMCodeError(
            "LLM code did not produce a `df` variable. "
            "The code must assign the final result to `df`."
        )

    if not isinstance(result_df, pd.DataFrame):
        logger.warning(
            f"After execution `df` is {type(result_df).__name__}, not DataFrame. "
            "Acceptable for terminal steps like model training."
        )

    # ── Extract trained model if present ─────────────────────────────
    # LLM is instructed to store the model in `trained_model`.
    # We package it alongside df so downstream code can save it.
    trained_model = namespace.get("trained_model", None)
    if trained_model is not None:
        logger.info(
            f"Trained model extracted from namespace: "
            f"{type(trained_model).__name__}"
        )
        # Return a package so both df and model are preserved
        return _ExecutionResult(df=result_df, model=trained_model)

    return result_df


class _ExecutionResult:
    """
    Thin wrapper returned when a step produces both a DataFrame and a model.
    Behaves like a DataFrame for isinstance() checks from legacy code,
    but also carries the model object.
    """
    def __init__(self, df: "pd.DataFrame", model: Any) -> None:
        self.df    = df
        self.model = model

    # Make it transparently usable as a DataFrame in the pipeline
    def __getattr__(self, item: str) -> Any:
        return getattr(self.df, item)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"ExecutionResult(df={self.df.shape}, model={type(self.model).__name__})"


# ---------------------------------------------------------------------------
# Result builder
# ---------------------------------------------------------------------------

def _build_result(
    task_id:         str,
    agent_name:      str,
    step_name:       str,
    input_summary:   str,
    output_data:     Any,
    output_summary:  str,
    code_equivalent: str,
    reasoning:       str,
    status:          str,
    error:           str,
    elapsed_ms:      float,
) -> Dict[str, Any]:
    return {
        "task_id":         task_id,
        "agent_name":      agent_name,
        "step_name":       step_name,
        "input_summary":   input_summary,
        "output_data":     output_data,
        "output_summary":  output_summary,
        "code_equivalent": code_equivalent,
        "reasoning":       reasoning,
        "status":          status,
        "error":           error,
        "elapsed_ms":      round(elapsed_ms, 3),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# DynamicAgent
# ---------------------------------------------------------------------------

class DynamicAgent:
    """
    A truly intelligent pipeline agent powered by a local or cloud LLM.

    Does NOT contain any data-processing logic.
    All logic is generated at runtime by the LLM after inspecting
    the actual DataFrame schema.

    Backend is controlled by the LLM_BACKEND environment variable:
        export LLM_BACKEND=ollama       # use local Ollama (default)
        export LLM_BACKEND=anthropic    # use Anthropic Claude

    Parameters
    ----------
    step_name : str
        Pipeline step label, e.g. "normalize_features".
    pipeline_steps : list, optional
        Full ordered pipeline for LLM context.
    api_key : str, optional
        Anthropic API key (only needed when LLM_BACKEND=anthropic).
    llm_model : str, optional
        Override the model name.
        For Ollama: "gpt-oss:120b-cloud" (default), "llama3.1:70b", "llama3.1:8b".
        For Anthropic: "claude-sonnet-4-20250514".
    """

    def __init__(
        self,
        step_name:      str,
        pipeline_steps: list | None = None,
        api_key:        str         = "",
        llm_model:      str         = "",
    ) -> None:
        self.step_name      = step_name
        self.agent_name     = f"DynamicAgent[{step_name}]"
        self.pipeline_steps = pipeline_steps or [step_name]
        self.api_key        = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.llm_model      = llm_model
        self._logger        = PipelineLogger(f"agents.dynamic.{step_name}")

        backend = os.environ.get("LLM_BACKEND", "ollama")
        self._logger.agent_event(
            self.agent_name,
            f"Initialised | backend={backend} | "
            f"model={llm_model or '(from env)'}"
        )

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute this step:
        1. Inspect the input DataFrame schema
        2. Ask the LLM for the best code
        3. Run the LLM-generated code
        4. Return the standard result dict
        """
        task_id    = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        input_summary = self._summarise(input_data)
        self._logger.agent_event(
            self.agent_name,
            f"execute() task_id={task_id} | input={input_summary}"
        )

        # ── Inspect schema ────────────────────────────────────────────
        df_schema = _inspect_dataframe(input_data)

        # ── LLM call ──────────────────────────────────────────────────
        self._logger.info(
            f"Querying LLM for step: '{self.step_name}' ..."
        )
        try:
            llm_resp = _call_llm(
                step_name        = self.step_name,
                df_schema        = df_schema,
                pipeline_context = self.pipeline_steps,
                logger           = self._logger,
                api_key          = self.api_key,
                model            = self.llm_model,
            )
        except AgentError as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return _build_result(
                task_id=task_id, agent_name=self.agent_name,
                step_name=self.step_name, input_summary=input_summary,
                output_data=input_data, output_summary="[LLM call failed]",
                code_equivalent="", reasoning="LLM unavailable",
                status="failed", error=str(exc), elapsed_ms=elapsed_ms,
            )

        code      = llm_resp["code"]
        reasoning = llm_resp["reasoning"]

        # ── Execute generated code ────────────────────────────────────
        try:
            output_data = _execute_code(code, input_data, self._logger)
        except LLMCodeError as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._logger.error(f"Code exec failed: {exc}")
            return _build_result(
                task_id=task_id, agent_name=self.agent_name,
                step_name=self.step_name, input_summary=input_summary,
                output_data=input_data, output_summary="[Execution failed]",
                code_equivalent=code, reasoning=reasoning,
                status="failed", error=str(exc), elapsed_ms=elapsed_ms,
            )

        elapsed_ms     = (time.perf_counter() - start_time) * 1000
        output_summary = self._summarise(output_data)

        self._logger.agent_event(
            self.agent_name,
            f"DONE in {elapsed_ms:.1f}ms | {output_summary}"
        )

        return _build_result(
            task_id=task_id, agent_name=self.agent_name,
            step_name=self.step_name, input_summary=input_summary,
            output_data=output_data, output_summary=output_summary,
            code_equivalent=code, reasoning=reasoning,
            status="success", error="", elapsed_ms=elapsed_ms,
        )

    def _summarise(self, data: Any) -> str:
        if data is None:
            return "None"
        # Handle ExecutionResult wrapper
        if hasattr(data, 'df') and hasattr(data, 'model'):
            return (
                f"ExecutionResult("
                f"df={data.df.shape[0]}x{data.df.shape[1]}, "
                f"model={type(data.model).__name__})"
            )
        if isinstance(data, pd.DataFrame):
            return (
                f"DataFrame({data.shape[0]}x{data.shape[1]}) "
                f"nulls={data.isnull().sum().sum()}"
            )
        if isinstance(data, np.ndarray):
            return f"ndarray{data.shape}"
        if isinstance(data, dict):
            return f"dict({list(data.keys())[:5]})"
        return type(data).__name__

    def __repr__(self) -> str:
        return f"DynamicAgent(step='{self.step_name}')"


# Alias
BaseAgent = DynamicAgent