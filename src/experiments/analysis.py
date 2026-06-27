from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import wilcoxon


def load_results(path: str = "results/experiment_results.jsonl") -> pd.DataFrame:
    """Read a JSONL of RunResult records into a DataFrame."""
    p = Path(path)
    records: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame.from_records(records)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, llm_backend): mean/median score on successes, success rate, mean runtime."""
    grouped = df.groupby(["condition", "llm_backend"], dropna=False, sort=True)
    rows = []
    for (condition, llm), group in grouped:
        succ = group[group["success"] == True]
        rows.append(
            {
                "condition": condition,
                "llm_backend": llm,
                "mean_score": succ["test_score"].mean() if len(succ) else float("nan"),
                "median_score": succ["test_score"].median() if len(succ) else float("nan"),
                "success_rate": float(group["success"].sum()) / len(group),
                "mean_runtime": group["runtime_seconds"].mean(),
                "n_runs": len(group),
            }
        )
    return pd.DataFrame(rows)


def error_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Count failures by (condition, error_category). Excludes successes."""
    failed = df[df["success"] == False]
    counts = (
        failed.groupby(["condition", "error_category"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    return counts


def wilcoxon_test(df: pd.DataFrame, condition_a: str, condition_b: str) -> dict[str, Any]:
    """Paired Wilcoxon over per-dataset mean success scores for two conditions."""
    succ = df[df["success"] == True]
    per_dataset = (
        succ.groupby(["dataset_id", "condition"])["test_score"].mean().reset_index()
    )
    pivot = per_dataset.pivot(index="dataset_id", columns="condition", values="test_score")
    if condition_a not in pivot.columns or condition_b not in pivot.columns:
        return {
            "statistic": None,
            "p_value": None,
            "n_datasets": 0,
            "condition_a_mean": None,
            "condition_b_mean": None,
        }
    paired = pivot[[condition_a, condition_b]].dropna()
    if len(paired) == 0:
        return {
            "statistic": None,
            "p_value": None,
            "n_datasets": 0,
            "condition_a_mean": None,
            "condition_b_mean": None,
        }
    a = paired[condition_a]
    b = paired[condition_b]
    stat, p = wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_datasets": int(len(paired)),
        "condition_a_mean": float(a.mean()),
        "condition_b_mean": float(b.mean()),
    }
