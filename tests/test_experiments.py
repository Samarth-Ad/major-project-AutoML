from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.contracts import GeneratedPipeline, RunResult, TaskType
from src.execution.runner import execute_pipeline
from src.experiments.analysis import (
    error_breakdown,
    iteration_efficiency,
    load_results,
    model_comparison,
    summary_table,
)
from src.experiments.datasets import build_task_description, load_custom_dataset


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


def test_load_custom_dataset(tmp_path: Path) -> None:
    csv = tmp_path / "my_data.csv"
    csv.write_text(
        "feat1,feat2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n"
        "9.0,10.0,0\n11.0,12.0,1\n13.0,14.0,0\n15.0,16.0,1\n"
        "17.0,18.0,0\n19.0,20.0,1\n",
        encoding="utf-8",
    )
    info = load_custom_dataset(str(csv), target_col="label")
    assert info["dataset_name"] == "my_data"
    assert info["target_col"] == "label"
    assert info["task_type"] == TaskType.BINARY_CLASSIFICATION
    assert len(info["df_train"]) + len(info["df_test"]) == 10
    assert "feat1" in info["df_train"].columns


def test_load_custom_dataset_bad_target(tmp_path: Path) -> None:
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        load_custom_dataset(str(csv), target_col="nonexistent")


def test_load_custom_dataset_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_custom_dataset("/no/such/file.csv", target_col="x")


def _mock_sweep_record(**overrides) -> dict:
    base = {
        "dataset_id": 37,
        "dataset_name": "diabetes",
        "condition": "B0",
        "llm_backend": "stub",
        "seed": 42,
        "success": True,
        "test_score": 0.8,
        "error_category": None,
        "error_message": None,
        "iterations_used": 1,
        "max_iterations": 5,
        "runtime_seconds": 10.0,
    }
    base.update(overrides)
    return base


def test_iteration_efficiency() -> None:
    df = pd.DataFrame(
        [
            _mock_sweep_record(condition="B0", iterations_used=3),
            _mock_sweep_record(condition="B0", iterations_used=1),
            _mock_sweep_record(condition="B2", iterations_used=1),
            _mock_sweep_record(condition="B2", iterations_used=2),
            _mock_sweep_record(condition="B0", success=False, test_score=None, iterations_used=5),
        ]
    )
    result = iteration_efficiency(df)
    assert not result.empty
    b0 = result[result["condition"] == "B0"].iloc[0]
    assert b0["mean_iterations"] == 2.0
    assert b0["count"] == 2


def test_model_comparison() -> None:
    df = pd.DataFrame(
        [
            _mock_sweep_record(llm_backend="model-a", test_score=0.9),
            _mock_sweep_record(llm_backend="model-a", test_score=0.8),
            _mock_sweep_record(llm_backend="model-b", test_score=0.7),
            _mock_sweep_record(
                llm_backend="model-b", success=False, test_score=None,
            ),
        ]
    )
    result = model_comparison(df)
    assert len(result) == 2
    ma = result[result["llm_backend"] == "model-a"].iloc[0]
    assert ma["success_rate"] == 1.0
    assert abs(ma["mean_score"] - 0.85) < 1e-9
    mb = result[result["llm_backend"] == "model-b"].iloc[0]
    assert mb["success_rate"] == 0.5
