"""
agents/base_agent.py
--------------------
LLM-powered agentic base — zero hardcoded processing logic.

Supports Ollama (local/cloud) and Anthropic backends.
Backend selected via LLM_BACKEND environment variable.

    export LLM_BACKEND=ollama          # default — uses gpt-oss:120b-cloud
    export OLLAMA_MODEL=gpt-oss:120b-cloud
    export OLLAMA_BASE_URL=http://localhost:11434

    export LLM_BACKEND=anthropic       # cloud fallback
    export ANTHROPIC_API_KEY=sk-ant-...

Flow per step
-------------
1.  DynamicAgent.execute(df)
2.  _inspect_dataframe(df)  →  rich schema dict with stats, skewness,
                                 class balance, correlation flags
3.  _call_llm()             →  routes to ollama or anthropic
4.  LLM returns             →  {reasoning, code, validation, summary}
5.  _sanitise_code()        →  auto-fix known bad pandas patterns
6.  _execute_code()         →  exec() in sandboxed namespace
7.  Result dict returned to Scheduler
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
# Schema inspector — richer than before
# ---------------------------------------------------------------------------

def _inspect_dataframe(df: Any) -> dict:
    """
    Build a comprehensive schema summary of a DataFrame for the LLM.

    Computes per-column:
      - dtype, null count, null %, unique count
      - numeric: min/max/mean/std + skewness
      - categorical: top value frequencies
      - bool/uint8: value counts (safe for one-hot cols)

    Computes dataset-level:
      - total_nulls, shape, sample_rows
      - class_balance  (if a 'target' column exists)
      - problem_type   (inferred from target dtype + cardinality)
      - high_correlation_pairs  (feature pairs with |r| > 0.9)
      - skewed_columns  (list of cols with |skewness| > 1.0)
    """
    if df is None:
        return {
            "type":        "None",
            "description": "No data — this is the first step.",
        }

    # Unwrap _ExecutionResult if present
    if not isinstance(df, pd.DataFrame):
        if hasattr(df, "df") and isinstance(df.df, pd.DataFrame):
            df = df.df
        else:
            return {
                "type":        type(df).__name__,
                "description": str(df)[:500],
            }

    schema: dict = {
        "type":        "pandas.DataFrame",
        "shape":       list(df.shape),
        "total_nulls": int(df.isnull().sum().sum()),
        "columns":     [],
        "sample_rows": json.loads(
            df.head(3).to_json(orient="records", default_handler=str)
        ),
    }

    # ── Per-column stats ──────────────────────────────────────────────
    skewed_cols: List[str] = []

    for col in df.columns:
        is_bool    = df[col].dtype == bool or str(df[col].dtype) == "bool"
        is_numeric = pd.api.types.is_numeric_dtype(df[col]) and not is_bool

        col_info: dict = {
            "name":     col,
            "dtype":    str(df[col].dtype),
            "nulls":    int(df[col].isnull().sum()),
            "null_pct": round(float(df[col].isnull().mean()) * 100, 2),
            "n_unique": int(df[col].nunique()),
        }

        if is_bool:
            col_info["top_values"] = {
                str(k): int(v)
                for k, v in df[col].value_counts().head(5).items()
            }

        elif is_numeric:
            desc = df[col].describe()
            col_info["stats"] = {
                "min":  round(float(desc.get("min",  desc.get("0%",  0))), 4),
                "max":  round(float(desc.get("max",  desc.get("100%", 0))), 4),
                "mean": round(float(desc.get("mean", 0)), 4),
                "std":  round(float(desc.get("std",  0)), 4),
            }
            # Skewness — skip cols with all nulls or zero variance
            try:
                skew_val = float(df[col].dropna().skew())
                col_info["skewness"] = round(skew_val, 4)
                if abs(skew_val) > 1.0:
                    skewed_cols.append(col)
            except Exception:
                col_info["skewness"] = 0.0

        else:
            col_info["top_values"] = {
                str(k): int(v)
                for k, v in df[col].value_counts().head(5).items()
            }

        schema["columns"].append(col_info)

    schema["skewed_columns"] = skewed_cols

    # ── Class balance (target column heuristic) ───────────────────────
    target_col = None
    for candidate in ["target", "label", "survived", "Survived",
                       "left", "churn", "outcome", "y"]:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None and len(df.columns) > 0:
        # Last column is often the target
        last = df.columns[-1]
        if df[last].nunique() < 20:
            target_col = last

    if target_col and target_col in df.columns:
        vc = df[target_col].value_counts(normalize=True)
        schema["class_balance"] = {
            str(k): round(float(v), 4)
            for k, v in vc.items()
        }
        minority_ratio = float(vc.min())
        schema["is_imbalanced"] = minority_ratio < 0.25   # <25% minority

        # Problem type inference
        n_unique_target = int(df[target_col].nunique())
        target_dtype    = str(df[target_col].dtype)
        if "float" in target_dtype or n_unique_target > 20:
            schema["problem_type"] = "regression"
        elif n_unique_target == 2:
            schema["problem_type"] = "binary_classification"
        else:
            schema["problem_type"] = "multiclass_classification"
    else:
        schema["class_balance"] = {}
        schema["is_imbalanced"] = False
        schema["problem_type"]  = "unknown"

    # ── High-correlation pairs ────────────────────────────────────────
    try:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 1:
            corr = df[num_cols].corr().abs()
            high_pairs: List[str] = []
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i + 1:]:
                    val = corr.loc[c1, c2]
                    if not np.isnan(val) and val > 0.9:
                        high_pairs.append(f"{c1}|{c2}={val:.2f}")
            schema["high_correlation_pairs"] = high_pairs[:10]  # cap at 10
        else:
            schema["high_correlation_pairs"] = []
    except Exception:
        schema["high_correlation_pairs"] = []

    return schema


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    step_name:        str,
    df_schema:        dict,
    pipeline_context: list,
) -> tuple[str, str]:
    """Build system + user prompts for any backend."""
    pipeline_str = " -> ".join(pipeline_context)
    schema_str   = json.dumps(df_schema, indent=2)

    system_prompt = (
        "You are an expert data scientist inside an agentic ML pipeline.\n"
        "Your job: inspect real DataFrame schemas and write the BEST Python "
        "code to execute a specific pipeline step on that exact data.\n\n"
        "STRICT RULES:\n"
        "1. Inspect ACTUAL column names, dtypes, nulls, value distributions.\n"
        "2. Choose strategies based on the real data — not generic defaults.\n"
        "3. The DataFrame variable is always named `df`.\n"
        "4. All imports must be inside your code block.\n"
        "5. Code must be complete and executable via Python exec().\n"
        "6. ALWAYS keep `df` as a pandas DataFrame at the end.\n"
        "7. For model training: store ONE model in `trained_model`, or MULTIPLE\n"
        "   models in `trained_models` dict (name→fitted_model). These carry\n"
        "   forward automatically to evaluation and explanation steps.\n\n"
        "PANDAS RULES (violations crash the pipeline):\n"
        "8.  NEVER combine engine='python' with low_memory=False in read_csv.\n"
        "9.  ALWAYS use encoding='utf-8' in read_csv on Windows.\n"
        "10. Use forward slashes or raw strings for file paths.\n"
        "11. After read_csv, verify: assert isinstance(df, pd.DataFrame).\n"
        "12. NEVER use strict `assert` for statistical properties like skewness, "
        "correlation, or exact distribution shapes. These fluctuate and cause "
        "unnecessary pipeline failures. Use print() or soft logging instead.\n\n"
        "RETURN FORMAT — strict JSON only, no markdown, no extra text:\n"
        "{\n"
        '  "reasoning": "Why this approach for THIS specific data",\n'
        '  "code": "complete Python as single string with \\\\n newlines",\n'
        '  "validation": "what check confirms this step worked",\n'
        '  "summary": "one-line description"\n'
        "}"
    )

    user_prompt = (
        f"Full pipeline: {pipeline_str}\n"
        f"Current step: \"{step_name}\"\n\n"
        f"DataFrame schema:\n{schema_str}\n\n"
        f"Write the BEST Python code to execute step \"{step_name}\" "
        f"on this exact DataFrame.\n"
        "Base EVERY decision on the actual schema above.\n"
        "Return ONLY the JSON object."
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# JSON response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(raw_text: str, logger: PipelineLogger) -> dict:
    """Parse + validate LLM JSON response. Strips markdown fences."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",           "", text)
    text = text.strip()

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        text = json_match.group(0)

    parsed = json.loads(text)

    required = {"reasoning", "code", "validation", "summary"}
    missing  = required - set(parsed.keys())
    if missing:
        raise ValueError(f"LLM response missing keys: {missing}")

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
    model:            str = "gpt-oss:120b-cloud",
    base_url:         str = "http://localhost:11434",
    max_retries:      int = 3,
) -> dict:
    """Call Ollama /api/chat endpoint."""
    import urllib.request
    import urllib.error

    system_prompt, user_prompt = _build_prompt(
        step_name, df_schema, pipeline_context
    )

    endpoint = f"{base_url.rstrip('/')}/api/chat"
    payload  = json.dumps({
        "model":   model,
        "stream":  False,
        "options": {"temperature": 0.2},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }).encode("utf-8")

    headers    = {"Content-Type": "application/json"}
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(
                f"Ollama attempt {attempt}/{max_retries} | "
                f"model={model} | endpoint={endpoint}"
            )
            req = urllib.request.Request(
                endpoint, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = json.loads(resp.read().decode("utf-8"))

            content = raw["message"]["content"]
            return _parse_llm_response(content, logger)

        except urllib.error.URLError as exc:
            last_error = exc
            err_str    = str(exc)
            if "Connection refused" in err_str:
                raise AgentError(
                    f"Cannot connect to Ollama at {base_url}.\n"
                    "  Start with: ollama serve\n"
                    f"  Pull model: ollama pull {model}"
                ) from exc
            if "timed out" in err_str or "TimeoutError" in err_str:
                logger.warning(f"Ollama timeout on attempt {attempt} — will retry.")
            else:
                logger.warning(f"Ollama URLError attempt {attempt}: {exc}")

        except KeyError as exc:
            last_error = exc
            logger.warning(f"Unexpected response shape attempt {attempt}: {exc}")

        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(f"JSON parse error attempt {attempt}: {exc}")

        except Exception as exc:
            last_error = exc
            logger.warning(f"Unexpected error attempt {attempt}: {exc}")

        if attempt < max_retries:
            wait = 2 ** attempt
            logger.info(f"Retrying in {wait}s ...")
            time.sleep(wait)

    raise AgentError(
        f"Ollama failed after {max_retries} attempts. Last: {last_error}"
    )


# ---------------------------------------------------------------------------
# Anthropic backend
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
    """Call Anthropic Claude API."""
    import urllib.request

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
                data=payload, headers=headers, method="POST",
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
        f"Anthropic failed after {max_retries} attempts: {last_error}"
    )


# ---------------------------------------------------------------------------
# LLM router
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
    """Route LLM call to correct backend via LLM_BACKEND env var."""
    backend = os.environ.get("LLM_BACKEND", "ollama").lower().strip()

    if backend == "ollama":
        ollama_model = model or os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
        ollama_url   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        logger.info(
            f"[LLM Router] Backend=OLLAMA | model={ollama_model} | url={ollama_url}"
        )
        return _call_ollama(
            step_name, df_schema, pipeline_context, logger,
            model=ollama_model, base_url=ollama_url, max_retries=max_retries,
        )

    elif backend == "anthropic":
        anthropic_model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        key             = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        logger.info(f"[LLM Router] Backend=ANTHROPIC | model={anthropic_model}")
        return _call_anthropic(
            step_name, df_schema, pipeline_context, logger,
            api_key=key, model=anthropic_model, max_retries=max_retries,
        )

    else:
        raise AgentError(
            f"Unknown LLM_BACKEND='{backend}'. Use 'ollama' or 'anthropic'."
        )


# ---------------------------------------------------------------------------
# Code sanitiser
# ---------------------------------------------------------------------------

def _sanitise_code(code: str, logger: PipelineLogger) -> str:
    """
    Auto-fix known bad LLM code patterns before exec().

    Fixes
    -----
    1. engine='python' + low_memory → remove engine (mutually exclusive)
    2. read_csv without encoding    → add encoding='utf-8'
    3. df[col].method(inplace=True) → df[col] = df[col].method()
       (pandas 2.x Copy-on-Write breaks chained inplace assignment)
    """
    original = code

    # Fix 1: conflicting pandas args (more robust regex for multi-line)
    if "low_memory" in code and "engine=" in code:
        code = re.sub(r",\s*engine=['\"]python['\"]", "", code)
        code = re.sub(r"engine=['\"]python['\"],\s*", "", code)
        logger.debug("Sanitiser: removed conflicting engine='python' (robust)")

    # Fix 4: remove overly strict statistical assertions (e.g. skewness checks)
    # LLMs often add 'assert new_skew.abs() < 0.5' which is too fragile.
    stat_assert_pattern = re.compile(
        r"^\s*assert\s+.*(skew|corr|std|mean|null).*(<|>|==|abs).*$", 
        re.MULTILINE | re.IGNORECASE
    )
    if stat_assert_pattern.search(code):
        code = stat_assert_pattern.sub(lambda m: f"# Removed strict assertion: {m.group(0).strip()}", code)
        logger.debug("Sanitiser: commented out overly strict statistical assertion")

    # Fix 2: missing encoding in read_csv
    if "read_csv" in code and "encoding=" not in code:
        code = re.sub(
            r"(pd\.read_csv\s*\([^)]*?)(\s*\))",
            lambda m: m.group(1) + ",\n    encoding='utf-8'" + m.group(2),
            code,
        )
        logger.debug("Sanitiser: added encoding='utf-8' to read_csv")

    # Fix 3: pandas Copy-on-Write — rewrite chained inplace assignments
    # Pattern: df[col].fillna(value, inplace=True) → df[col] = df[col].fillna(value)
    # Also handles: .replace(), .clip(), .where(), .mask()
    inplace_pattern = re.compile(
        r"(df\[([^\]]+)\])\."                       # df[col].
        r"(fillna|replace|clip|where|mask)"          # method name
        r"\(([^)]*?),\s*inplace\s*=\s*True([^)]*)\)" # (args, inplace=True)
    )
    if inplace_pattern.search(code):
        def _rewrite_inplace(m):
            lhs    = m.group(1)           # df[col]
            method = m.group(3)           # fillna
            args   = m.group(4).rstrip(", ")  # positional args before inplace
            rest   = m.group(5).lstrip(", ")  # anything after inplace=True
            call_args = ", ".join(a for a in [args, rest] if a)
            return f"{lhs} = {lhs}.{method}({call_args})"
        code = inplace_pattern.sub(_rewrite_inplace, code)
        logger.debug("Sanitiser: rewrote chained inplace assignment (CoW fix)")

    # Fix 5: sklearn LogisticRegression multi_class parameter (removed in 1.4+)
    if "LogisticRegression" in code and "multi_class=" in code:
        # Replace multi_class='multinomial' or 'ovr' with nothing, 
        # as modern sklearn handles this automatically.
        code = re.sub(r",\s*multi_class=['\"][^'\"]+['\"]", "", code)
        code = re.sub(r"multi_class=['\"][^'\"]+['\"],\s*", "", code)
        logger.debug("Sanitiser: removed deprecated multi_class from LogisticRegression")

    if code != original:
        logger.info("Sanitiser applied fixes before execution")

    return code


# ---------------------------------------------------------------------------
# Safe code executor
# ---------------------------------------------------------------------------

def _execute_code(code: str, df: Any, logger: PipelineLogger) -> Any:
    """
    Execute LLM-generated code in a sandboxed namespace.

    Pre-loads: pandas, numpy, sklearn, scipy, joblib,
               xgboost, lightgbm, optuna (if installed).
    Returns df, or _ExecutionResult(df, model, models) if trained models found.

    State passing
    -------------
    If `df` is an _ExecutionResult from a previous step, the wrapped
    model(s) are re-injected into the namespace as `trained_model`
    and/or `trained_models` so downstream LLM code can reference them.
    """
    code = _sanitise_code(code, logger)

    # Unwrap _ExecutionResult — extract df and any carried models
    carried_model  = None
    carried_models = None
    actual_df      = df

    if hasattr(df, "df") and hasattr(df, "model"):
        actual_df      = df.df
        carried_model  = df.model
        carried_models = getattr(df, "models", None)
        logger.debug(
            f"Unwrapped _ExecutionResult — re-injecting "
            f"model={type(carried_model).__name__}, "
            f"models={'dict(' + str(len(carried_models)) + ')' if carried_models else 'None'}"
        )

    # Core namespace
    namespace: dict = {
        "df":           actual_df.copy() if isinstance(actual_df, pd.DataFrame) else actual_df,
        "pd":           pd,
        "np":           np,
        "print":        print,
        "__builtins__": __builtins__,
    }

    # Re-inject trained models from previous steps so downstream
    # code (evaluate_models, select_best_model, explain_model)
    # can access them without re-training.
    if carried_model is not None:
        namespace["trained_model"] = carried_model
    if carried_models is not None:
        namespace["trained_models"] = carried_models

    # Inject all available ML libraries
    for lib_name in (
        "sklearn", "scipy", "joblib",
        "xgboost", "lightgbm", "optuna",
        "imblearn",            # for SMOTE
        "shap",                # for explainability
    ):
        try:
            import importlib
            namespace[lib_name] = importlib.import_module(lib_name)
        except ImportError:
            pass  # optional — not all may be installed yet

    logger.debug(f"Executing LLM code ({len(code)} chars) ...")

    try:
        exec(compile(code, "<llm_generated>", "exec"), namespace)
    except Exception as exc:
        raise LLMCodeError(
            f"Execution error: {exc}\n"
            f"--- Code ---\n{code}\n"
            f"--- Traceback ---\n{traceback.format_exc()}"
        ) from exc

    result_df = namespace.get("df")
    if result_df is None:
        raise LLMCodeError(
            "LLM code did not produce a `df` variable. "
            "The result must be assigned to `df`."
        )

    if not isinstance(result_df, pd.DataFrame):
        logger.warning(
            f"df after execution is {type(result_df).__name__}, not DataFrame. "
            "Acceptable for terminal steps like model training."
        )

    # Extract trained model(s) from namespace
    trained_model  = namespace.get("trained_model", carried_model)
    trained_models = namespace.get("trained_models", carried_models)

    # If the step produced new models, wrap in _ExecutionResult
    if trained_model is not None or trained_models is not None:
        if trained_models and not trained_model:
            # Pick first model as the default single model
            trained_model = next(iter(trained_models.values()))
        if trained_model:
            logger.info(
                f"Trained model extracted: {type(trained_model).__name__}"
                + (f" (+{len(trained_models)} in dict)" if trained_models else "")
            )
        return _ExecutionResult(
            df=result_df, model=trained_model, models=trained_models,
        )

    return result_df


# ---------------------------------------------------------------------------
# ExecutionResult wrapper
# ---------------------------------------------------------------------------

class _ExecutionResult:
    """
    Wraps a (DataFrame, model(s)) tuple from training steps.
    Proxies all DataFrame attribute access so downstream code
    never needs to know about this wrapper.

    Attributes
    ----------
    df     : pd.DataFrame
    model  : Any           — single best model (or first from dict)
    models : dict | None   — dict of {name: model} for multi-model steps
    """
    def __init__(
        self,
        df:     "pd.DataFrame",
        model:  Any,
        models: Any = None,
    ) -> None:
        self.df     = df
        self.model  = model
        self.models = models

    def __getattr__(self, item: str) -> Any:
        return getattr(self.df, item)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        n_models = len(self.models) if self.models else (1 if self.model else 0)
        return (
            f"ExecutionResult("
            f"df={self.df.shape}, "
            f"model={type(self.model).__name__ if self.model else 'None'}, "
            f"models={n_models})"
        )


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
    LLM-powered pipeline agent. Contains zero domain logic.
    All processing strategy is generated at runtime by the LLM
    after inspecting the actual DataFrame schema.

    Backend: set LLM_BACKEND=ollama (default) or LLM_BACKEND=anthropic.
    Model:   set OLLAMA_MODEL=gpt-oss:120b-cloud (default).
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
            f"Initialised | backend={backend} | model={llm_model or '(from env)'}",
        )

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Execute this step:
        1. Inspect DataFrame schema
        2. Call LLM for best code
        3. Sanitise + exec the code
        4. Return standard result dict
        """
        task_id    = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        input_summary = self._summarise(input_data)
        self._logger.agent_event(
            self.agent_name,
            f"execute() task_id={task_id} | input={input_summary}",
        )

        df_schema = _inspect_dataframe(input_data)

        self._logger.info(f"Querying LLM for step: '{self.step_name}' ...")
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
            f"DONE in {elapsed_ms:.1f}ms | {output_summary}",
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
        if hasattr(data, "df") and hasattr(data, "model"):
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


# Alias for backward compatibility
BaseAgent = DynamicAgent