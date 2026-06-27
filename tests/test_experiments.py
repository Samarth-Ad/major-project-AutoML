from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.contracts import GeneratedPipeline, RunResult, TaskType
from src.execution.runner import execute_pipeline
from src.experiments.analysis import error_breakdown, load_results, summary_table
from src.experiments.datasets import build_task_description


def test_build_task_description() -> None:
    dataset_info = {
        "dataset_id": 37,
        "dataset_name": "diabetes",
        "task_type": TaskType.BINARY_CLASSIFICATION,
        "target_col": "class",
        "train_path": "/tmp/diabetes/train.csv",
        "test_path": "/tmp/diabetes/test.csv",
        "df_train": None,
        "df_test": None,
    }
    prompt = build_task_description(dataset_info)
    assert "SCORE:" in prompt
    assert "/tmp/diabetes/train.csv" in prompt
    assert "/tmp/diabetes/test.csv" in prompt
    assert "class" in prompt
    assert "balanced_accuracy" in prompt


def test_extract_score_integration() -> None:
    pipeline = GeneratedPipeline(
        code='print("SCORE: 0.85")',
        condition="B0",
        llm_backend="stub",
        dataset_id="integration",
        seed=0,
        iteration=0,
    )
    result = execute_pipeline(pipeline, TaskType.BINARY_CLASSIFICATION, timeout_seconds=10)
    assert result.success is True
    assert result.test_score == 0.85


def _mock_record(**overrides) -> dict:
    base = {
        "dataset_id": "ds1",
        "condition": "B0",
        "llm_backend": "stub",
        "seed": 0,
        "iteration": 0,
        "success": True,
        "error_category": None,
        "error_message": None,
        "test_score": 0.8,
        "runtime_seconds": 1.0,
        "generated_code_path": "x.py",
    }
    base.update(overrides)
    return base


def test_load_results(tmp_path: Path) -> None:
    jsonl = tmp_path / "results.jsonl"
    lines = [
        RunResult(**_mock_record()).model_dump_json(),
        RunResult(**_mock_record(condition="B1", test_score=0.85)).model_dump_json(),
        RunResult(
            **_mock_record(
                condition="B2",
                success=False,
                test_score=None,
                error_category="timeout",
                error_message="x",
            )
        ).model_dump_json(),
    ]
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")

    df = load_results(str(jsonl))
    assert len(df) == 3
    for col in ("dataset_id", "condition", "llm_backend", "success", "test_score", "runtime_seconds"):
        assert col in df.columns


def test_summary_table() -> None:
    df = pd.DataFrame(
        [
            _mock_record(condition="B0", success=True, test_score=0.7, runtime_seconds=1.0),
            _mock_record(condition="B0", success=False, test_score=None, runtime_seconds=2.0),
            _mock_record(condition="B1", success=True, test_score=0.8, runtime_seconds=3.0),
            _mock_record(condition="B1", success=True, test_score=0.9, runtime_seconds=4.0),
            _mock_record(condition="B2", success=False, test_score=None, runtime_seconds=5.0),
        ]
    )
    table = summary_table(df)
    for col in ("condition", "llm_backend", "mean_score", "median_score", "success_rate", "mean_runtime", "n_runs"):
        assert col in table.columns
    assert ((table["success_rate"] >= 0) & (table["success_rate"] <= 1)).all()
    b0 = table[table["condition"] == "B0"].iloc[0]
    assert b0["n_runs"] == 2
    assert b0["success_rate"] == 0.5
    assert b0["mean_score"] == 0.7


def test_error_breakdown() -> None:
    df = pd.DataFrame(
        [
            _mock_record(condition="B0", success=False, error_category="timeout", test_score=None),
            _mock_record(condition="B0", success=False, error_category="timeout", test_score=None),
            _mock_record(condition="B0", success=False, error_category="syntax_error", test_score=None),
            _mock_record(condition="B1", success=False, error_category="import_error", test_score=None),
            _mock_record(condition="B2", success=True),
        ]
    )
    counts = error_breakdown(df)
    by_pair = {(row["condition"], row["error_category"]): row["count"] for _, row in counts.iterrows()}
    assert by_pair[("B0", "timeout")] == 2
    assert by_pair[("B0", "syntax_error")] == 1
    assert by_pair[("B1", "import_error")] == 1
    assert ("B2", None) not in by_pair
