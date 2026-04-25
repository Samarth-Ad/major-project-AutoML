"""
agents/data_understanding_agent.py
------------------------------------
DataUnderstandingAgent — First agent in the adaptive pipeline.

Runs immediately after load_dataset. Does two passes:

Pass 1 — Pure Python (no LLM, instant):
  Computes skewness, class balance, null %, correlation, cardinality.
  Applies threshold checks from config/pipeline.yaml.

Pass 2 — LLM interpretation (one call):
  Sends computed profile to the LLM.
  LLM returns a PipelineDecision JSON: which steps are needed/skipped,
  which models to try, whether tuning is worthwhile, and why.

Fallback: If LLM fails, rule-based decision from thresholds alone.

Output — PipelineDecision dataclass consumed by MasterAgent.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from agents.base_agent import _inspect_dataframe, _build_result
from utils.logger import PipelineLogger


# ---------------------------------------------------------------------------
# PipelineDecision
# ---------------------------------------------------------------------------

@dataclass
class PipelineDecision:
    """Adaptive pipeline plan produced by DataUnderstandingAgent."""

    problem_type:    str
    target_column:   str
    steps:           List[str]
    skipped:         Dict[str, str]
    models_to_try:   List[str]
    needs_tuning:    bool
    n_rows:          int
    n_cols:          int
    has_nulls:       bool
    is_imbalanced:   bool
    skewed_columns:  List[str]
    high_corr_pairs: List[str]
    reasoning:       Dict[str, str]
    pipeline_id:     str = ""
    timestamp:       str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_type":    self.problem_type,
            "target_column":   self.target_column,
            "steps":           self.steps,
            "skipped":         self.skipped,
            "models_to_try":   self.models_to_try,
            "needs_tuning":    self.needs_tuning,
            "n_rows":          self.n_rows,
            "n_cols":          self.n_cols,
            "has_nulls":       self.has_nulls,
            "is_imbalanced":   self.is_imbalanced,
            "skewed_columns":  self.skewed_columns,
            "high_corr_pairs": self.high_corr_pairs,
            "reasoning":       self.reasoning,
            "pipeline_id":     self.pipeline_id,
            "timestamp":       self.timestamp,
        }

    def summary(self) -> str:
        return (
            f"PipelineDecision("
            f"type={self.problem_type}, "
            f"steps={len(self.steps)}, "
            f"skipped={len(self.skipped)}, "
            f"models={self.models_to_try})"
        )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_thresholds(config_path: str = "config/pipeline.yaml") -> Dict[str, Any]:
    """Read thresholds from YAML. Falls back to sensible defaults."""
    defaults: Dict[str, Any] = {
        "null_pct_to_trigger_imputation":   0.0,
        "imbalance_ratio_to_trigger_smote": 0.25,
        "skewness_to_trigger_correction":   1.0,
        "features_to_trigger_pca":          50,
        "rows_to_trigger_tuning":           500,
    }
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return {**defaults, **cfg.get("thresholds", {})}
    except Exception:
        return defaults


def _load_target_column(config_path: str = "config/pipeline.yaml") -> str:
    """Read target_column from YAML data section."""
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("data", {}).get("target_column", "") or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Pure-Python analysis
# ---------------------------------------------------------------------------

def _analyse_data(
    df:            pd.DataFrame,
    thresholds:    Dict[str, Any],
    target_column: str,
) -> Dict[str, Any]:
    """Compute a comprehensive data profile — no LLM, pure Python."""
    schema = _inspect_dataframe(df)

    n_rows, n_cols = df.shape
    total_nulls    = int(df.isnull().sum().sum())

    # Resolve target column
    resolved_target = target_column
    if not resolved_target:
        for candidate in ["target", "label", "survived", "Survived",
                          "left", "churn", "outcome", "y"]:
            if candidate in df.columns:
                resolved_target = candidate
                break
    if not resolved_target and n_cols > 0:
        last = df.columns[-1]
        if df[last].nunique() <= 20:
            resolved_target = last

    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    # Also catch 'str' dtype which appears in some pandas versions
    cat_cols = list(dict.fromkeys(
        cat_cols + [
            c for c in df.columns
            if c not in cat_cols
            and str(df[c].dtype).lower() in ("str", "string", "object", "category")
        ]
    ))
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # High-cardinality cats (likely IDs — skip OHE)
    high_card_cats = [
        c for c in cat_cols
        if df[c].nunique() > min(50, n_rows * 0.5)
    ]
    encodable_cats = [c for c in cat_cols if c not in high_card_cats]

    # Estimated cols after OHE (cap per-col expansion at 20)
    est_cols_after_encoding = (
        (n_cols - len(cat_cols))
        + sum(min(df[c].nunique(), 20) for c in encodable_cats)
    )

    skewed_cols     = schema.get("skewed_columns", [])
    is_imbalanced   = schema.get("is_imbalanced", False)
    class_balance   = schema.get("class_balance", {})
    problem_type    = schema.get("problem_type", "unknown")
    high_corr_pairs = schema.get("high_correlation_pairs", [])

    # ── Determine if scaling is needed ───────────────────────────
    # Scaling is needed when numeric features have very different
    # magnitude ranges (max range / min range > 10), or when
    # scale-sensitive models (logistic regression, SVM) are likely.
    needs_scaling = False
    if len(num_cols) >= 2:
        ranges = []
        for c in num_cols:
            col_range = float(df[c].max() - df[c].min())
            if col_range > 0:
                ranges.append(col_range)
        if len(ranges) >= 2:
            ratio = max(ranges) / max(min(ranges), 1e-9)
            needs_scaling = ratio > 10

    # ── Determine if feature engineering is useful ────────────────
    # Need at least 2 numeric columns for interaction features.
    needs_feat_eng = len(num_cols) >= 2

    checks: Dict[str, bool] = {
        "has_nulls":        total_nulls > 0,
        "needs_smote":      is_imbalanced,
        "needs_skewness":   len(skewed_cols) > 0,
        "needs_pca":        est_cols_after_encoding > thresholds["features_to_trigger_pca"],
        "needs_tuning":     n_rows > thresholds["rows_to_trigger_tuning"],
        "has_categoricals": len(encodable_cats) > 0,
        "needs_scaling":    needs_scaling,
        "needs_feat_eng":   needs_feat_eng,
    }

    return {
        "n_rows":                  n_rows,
        "n_cols":                  n_cols,
        "n_numeric":               len(num_cols),
        "n_categorical":           len(cat_cols),
        "n_high_cardinality_cats": len(high_card_cats),
        "high_cardinality_cols":   high_card_cats,
        "total_nulls":             total_nulls,
        "skewed_columns":          skewed_cols,
        "is_imbalanced":           is_imbalanced,
        "class_balance":           class_balance,
        "problem_type":            problem_type,
        "target_column":           resolved_target,
        "high_correlation_pairs":  high_corr_pairs,
        "est_cols_after_encoding": est_cols_after_encoding,
        "thresholds":              thresholds,
        "checks":                  checks,
        "sample_rows":             schema.get("sample_rows", []),
        "columns_summary": [
            {k: v for k, v in col.items() if k != "top_values"}
            for col in schema.get("columns", [])
        ],
    }


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

def _build_decision_prompt(profile: Dict[str, Any], user_prompt: str = "") -> tuple[str, str]:
    system_prompt = (
        "You are an autonomous ML architect. Given a dataset profile and optional user intent, decide:\n"
        "1. Which preprocessing steps are needed (and why)\n"
        "2. Which to skip (and why)\n"
        "3. Which 2-3 models to try (be selective to save time)\n"
        "4. Whether hyperparameter tuning is worthwhile\n\n"
        "DECISION RULES:\n"
        "- remove_missing_values  : only if checks.has_nulls = true\n"
        "- handle_class_imbalance : only if checks.needs_smote = true AND classification\n"
        "- encode_categorical      : only if checks.has_categoricals = true\n"
        "- handle_skewness         : only if checks.needs_skewness = true\n"
        "- normalize_features      : include if logistic_regression or svm in models\n"
        "- feature_engineering     : include if meaningful numeric interactions possible\n"
        "- dimensionality_reduction: only if checks.needs_pca = true\n"
        "- hyperparameter_tuning   : only if checks.needs_tuning = true\n\n"
        "MODEL SELECTION RULES (SELECT ONLY 2-3):\n"
        "- For Regression: xgboost, random_forest, or ridge_classifier (choose the best 2 based on data size/complexity)\n"
        "- For Classification: xgboost, random_forest, or logistic_regression (choose the best 2 based on data size/complexity)\n"
        "- ONLY include more than 2 models if specifically requested by user intent.\n\n"
        "USER INTENT OVERRIDE:\n"
        "- If the user provides a prompt, prioritize their instructions (e.g., 'Use XGBoost', 'Skip scaling').\n"
        "- However, do not include steps that are technically impossible (e.g., SMOTE on regression).\n\n"
        "VALID STEPS (use exactly these names):\n"
        "load_dataset, remove_missing_values, handle_class_imbalance,\n"
        "encode_categorical, handle_skewness, normalize_features,\n"
        "feature_engineering, dimensionality_reduction,\n"
        "select_and_train_models, evaluate_models,\n"
        "select_best_model, explain_model\n\n"
        "VALID MODEL NAMES:\n"
        "random_forest, xgboost, lightgbm, logistic_regression, svm,\n"
        "gradient_boosting, extra_trees, ridge_classifier, decision_tree\n\n"
        "RETURN strict JSON only, no markdown:\n"
        "{\n"
        '  "problem_type": "binary_classification|multiclass_classification|regression",\n'
        '  "target_column": "column_name",\n'
        '  "steps": ["step1", "step2", ...],\n'
        '  "skipped": {"step_name": "reason", ...},\n'
        '  "models_to_try": ["model1", "model2", "model3"],\n'
        '  "needs_tuning": true,\n'
        '  "reasoning": {"step_name": "one-line reason", ...}\n'
        "}"
    )
    prompt_content = f"Dataset profile:\n{json.dumps(profile, indent=2, default=str)}"
    if user_prompt:
        prompt_content += f"\n\nUser intent/instructions:\n{user_prompt}"
    
    user_prompt_str = (
        "Analyze this dataset profile and user intent to decide the optimal pipeline:\n\n"
        + prompt_content
        + "\n\nReturn ONLY the JSON decision."
    )
    return system_prompt, user_prompt_str


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

_VALID_STEPS = {
    "load_dataset", "remove_missing_values", "handle_class_imbalance",
    "encode_categorical", "handle_skewness", "normalize_features",
    "feature_engineering", "dimensionality_reduction",
    "select_and_train_models", "evaluate_models",
    "select_best_model", "explain_model",
}


def _parse_decision_response(
    raw_text: str,
    profile:  Dict[str, Any],
    logger:   PipelineLogger,
) -> Dict[str, Any]:
    """Parse LLM JSON decision. Falls back to rule-based on failure."""
    text = raw_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)

    try:
        d = json.loads(text)
        required = {"problem_type", "target_column", "steps",
                    "skipped", "models_to_try", "needs_tuning"}
        missing = required - set(d.keys())
        if missing:
            raise ValueError(f"Missing keys: {missing}")
        d["steps"] = [s for s in d["steps"] if s in _VALID_STEPS]

        # ── HARD VALIDATION: enforce data profile checks ──────────
        # Even if the LLM says "include step X", we remove it if the
        # actual data profile says it's not needed. This prevents
        # unnecessary agents from being created.
        checks = profile.get("checks", {})
        pt = d.get("problem_type", "")

        # Map: step_name → (check_key, extra_condition)
        _STEP_GUARDS = {
            "remove_missing_values":  ("has_nulls",        True),
            "handle_class_imbalance": ("needs_smote",       "classification" in pt),
            "encode_categorical":     ("has_categoricals",  True),
            "handle_skewness":        ("needs_skewness",    True),
            "normalize_features":     ("needs_scaling",     True),
            "feature_engineering":    ("needs_feat_eng",    True),
            "dimensionality_reduction": ("needs_pca",       True),
        }

        validated_steps = []
        if not d.get("skipped"):
            d["skipped"] = {}
        if not d.get("reasoning"):
            d["reasoning"] = {}

        for step in d["steps"]:
            if step in _STEP_GUARDS:
                check_key, extra_cond = _STEP_GUARDS[step]
                data_says_needed = checks.get(check_key, False)
                if not data_says_needed or not extra_cond:
                    reason = f"Data profile check '{check_key}' is False"
                    d["skipped"][step] = reason
                    logger.info(f"  OVERRIDE '{step}': {reason} — removed from pipeline")
                    continue
            validated_steps.append(step)

        d["steps"] = validated_steps
        # ── END VALIDATION ────────────────────────────────────────

        logger.info(
            f"LLM decision: type={d['problem_type']} | "
            f"steps={d['steps']} | models={d['models_to_try']}"
        )
        return d
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"LLM parse failed ({exc}). Using rule-based fallback.")
        return _rule_based_decision(profile, logger)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_based_decision(
    profile: Dict[str, Any],
    logger:  PipelineLogger,
) -> Dict[str, Any]:
    """Deterministic decision from threshold checks — no LLM needed."""
    checks = profile["checks"]
    pt     = profile["problem_type"]

    steps:     List[str]       = ["load_dataset"]
    skipped:   Dict[str, str]  = {}
    reasoning: Dict[str, str]  = {}

    if checks["has_nulls"]:
        steps.append("remove_missing_values")
        reasoning["remove_missing_values"] = "Dataset has null values"
    else:
        skipped["remove_missing_values"] = "No null values found"

    if checks["needs_smote"] and "classification" in pt:
        steps.append("handle_class_imbalance")
        reasoning["handle_class_imbalance"] = "Minority class < 25%"
    else:
        skipped["handle_class_imbalance"] = (
            "Data is balanced" if not checks["needs_smote"]
            else "Not a classification problem"
        )

    if checks["has_categoricals"]:
        steps.append("encode_categorical")
        reasoning["encode_categorical"] = "Categorical columns present"
    else:
        skipped["encode_categorical"] = "No encodable categorical columns"

    if checks["needs_skewness"]:
        steps.append("handle_skewness")
        reasoning["handle_skewness"] = f"Skewed cols: {profile['skewed_columns']}"
    else:
        skipped["handle_skewness"] = "No significant skewness"

    if checks["needs_scaling"]:
        steps.append("normalize_features")
        reasoning["normalize_features"] = "Numeric features have very different scales"
    else:
        skipped["normalize_features"] = "Numeric features are on comparable scales"

    if checks["needs_feat_eng"]:
        steps.append("feature_engineering")
        reasoning["feature_engineering"] = (
            f"{profile['n_numeric']} numeric columns — interaction features may help"
        )
    else:
        skipped["feature_engineering"] = (
            "Not enough numeric columns for meaningful interactions"
        )

    if checks["needs_pca"]:
        steps.append("dimensionality_reduction")
        reasoning["dimensionality_reduction"] = (
            f"Est. {profile['est_cols_after_encoding']} cols after encoding"
        )
    else:
        skipped["dimensionality_reduction"] = (
            f"Only {profile['est_cols_after_encoding']} cols after encoding"
        )

    steps += ["select_and_train_models", "evaluate_models",
              "select_best_model", "explain_model"]

    models = (
        ["random_forest", "xgboost", "logistic_regression"]
        if "classification" in pt
        else ["random_forest", "xgboost", "ridge"]
    )

    return {
        "problem_type":  pt,
        "target_column": profile["target_column"],
        "steps":         steps,
        "skipped":       skipped,
        "models_to_try": models,
        "needs_tuning":  checks["needs_tuning"],
        "reasoning":     reasoning,
    }


# ---------------------------------------------------------------------------
# DataUnderstandingAgent
# ---------------------------------------------------------------------------

class DataUnderstandingAgent:
    """
    Adaptive first agent — analyses data and decides the pipeline.

    Pass 1: Pure Python — instant, reads actual data stats.
    Pass 2: LLM call   — interprets profile, returns decision JSON.
    Fallback: rule-based if LLM fails.

    Returns (PipelineDecision, agent_result_dict).
    """

    def __init__(
        self,
        llm_model:   str = "",
        api_key:     str = "",
        config_path: str = "config/pipeline.yaml",
    ) -> None:
        self.llm_model   = llm_model
        self.api_key     = api_key
        self.config_path = config_path
        self.step_name   = "understand_data"
        self.agent_name  = "DataUnderstandingAgent"
        self._logger     = PipelineLogger("agents.DataUnderstandingAgent")
        self._logger.agent_event(self.agent_name, f"Initialised | config={config_path}")

    def execute(
        self,
        input_data:    Any,
        target_column: str = "",
        user_prompt:   str = "",
    ) -> tuple[PipelineDecision, Dict[str, Any]]:
        """
        Analyse DataFrame and return (PipelineDecision, result_dict).

        Parameters
        ----------
        input_data    : DataFrame or _ExecutionResult from load_dataset
        target_column : override from CLI/YAML; auto-inferred if empty
        user_prompt   : natural language intent from user
        """
        task_id    = str(uuid.uuid4())[:8]
        start_time = time.perf_counter()

        # Unwrap
        if hasattr(input_data, "df"):
            df = input_data.df
        elif isinstance(input_data, pd.DataFrame):
            df = input_data
        else:
            raise ValueError(
                f"Expected DataFrame, got {type(input_data).__name__}"
            )

        thresholds   = _load_thresholds(self.config_path)
        cfg_target   = _load_target_column(self.config_path)
        resolved_tgt = target_column or cfg_target

        self._logger.info(
            f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols | "
            f"target='{resolved_tgt or '(auto-infer)'}'"
        )

        # ── Pass 1 ────────────────────────────────────────────────────
        self._logger.info("Pass 1: computing data profile ...")
        profile = _analyse_data(df, thresholds, resolved_tgt)
        self._logger.info(
            f"  type={profile['problem_type']} | nulls={profile['total_nulls']} | "
            f"imbalanced={profile['is_imbalanced']} | skewed={profile['skewed_columns']}"
        )

        # ── Pass 2 ────────────────────────────────────────────────────
        self._logger.info("Pass 2: querying LLM for pipeline decision ...")
        system_prompt, user_prompt_str = _build_decision_prompt(profile, user_prompt)

        try:
            import urllib.request as _urlreq

            backend  = os.environ.get("LLM_BACKEND", "ollama").lower()
            model    = self.llm_model or os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

            if backend == "ollama":
                payload = json.dumps({
                    "model":   model,
                    "stream":  False,
                    "options": {"temperature": 0.1},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt_str},
                    ],
                }).encode("utf-8")
                req = _urlreq.Request(
                    f"{base_url.rstrip('/')}/api/chat",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with _urlreq.urlopen(req, timeout=180) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                content = raw["message"]["content"]
            else:
                api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
                payload = json.dumps({
                    "model":      os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                    "max_tokens": 2000,
                    "system":     system_prompt,
                    "messages":   [{"role": "user", "content": user_prompt_str}],
                }).encode("utf-8")
                req = _urlreq.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=payload,
                    headers={
                        "Content-Type":      "application/json",
                        "anthropic-version": "2023-06-01",
                        "x-api-key":         api_key,
                    },
                    method="POST",
                )
                with _urlreq.urlopen(req, timeout=60) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                content = raw["content"][0]["text"]

            decision_dict = _parse_decision_response(content, profile, self._logger)

        except Exception as exc:
            self._logger.warning(f"LLM failed ({exc}). Using rule-based fallback.")
            decision_dict = _rule_based_decision(profile, self._logger)

        # ── Build PipelineDecision ────────────────────────────────────
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        decision = PipelineDecision(
            problem_type    = decision_dict["problem_type"],
            target_column   = decision_dict.get("target_column") or profile["target_column"],
            steps           = decision_dict["steps"],
            skipped         = decision_dict.get("skipped", {}),
            models_to_try   = decision_dict.get("models_to_try", ["random_forest", "xgboost"]),
            needs_tuning    = bool(decision_dict.get("needs_tuning", profile["checks"]["needs_tuning"])),
            n_rows          = profile["n_rows"],
            n_cols          = profile["n_cols"],
            has_nulls       = profile["checks"]["has_nulls"],
            is_imbalanced   = profile["is_imbalanced"],
            skewed_columns  = profile["skewed_columns"],
            high_corr_pairs = profile["high_correlation_pairs"],
            reasoning       = decision_dict.get("reasoning", {}),
        )

        self._logger.info(f"Decision : {decision.summary()}")
        self._logger.info(f"Steps    : {decision.steps}")
        for step, reason in decision.skipped.items():
            self._logger.info(f"  SKIP {step}: {reason}")

        # ── Scheduler-compatible result dict ──────────────────────────
        agent_result = _build_result(
            task_id         = task_id,
            agent_name      = self.agent_name,
            step_name       = self.step_name,
            input_summary   = f"DataFrame({df.shape[0]}x{df.shape[1]})",
            output_data     = df,
            output_summary  = decision.summary(),
            code_equivalent = (
                "# DataUnderstandingAgent — adaptive pipeline decision\n"
                f"# problem_type  = '{decision.problem_type}'\n"
                f"# target_column = '{decision.target_column}'\n"
                f"# steps         = {decision.steps}\n"
                f"# models        = {decision.models_to_try}\n"
            ),
            reasoning = (
                f"Analysed {df.shape[0]}x{df.shape[1]}. "
                f"Problem: {decision.problem_type}. "
                f"{len(decision.steps)} steps, {len(decision.skipped)} skipped."
            ),
            status    = "success",
            error     = "",
            elapsed_ms = elapsed_ms,
        )

        # Attach for MasterAgent to extract
        agent_result["pipeline_decision"] = decision

        return decision, agent_result