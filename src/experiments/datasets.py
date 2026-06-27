from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.contracts import TaskType

SELECTED_CLASSIFICATION: dict[int, str] = {
    # Small (<1000 rows)
    37: "diabetes",
    15: "breast-w",
    29: "credit-approval",
    1510: "wdbc",
    11: "balance-scale",
    54: "vehicle",
    188: "eucalyptus",
    23381: "dresses-sales",
    # Medium (1k-10k)
    31: "credit-g",
    40975: "car",
    44: "spambase",
    1494: "qsar-biodeg",
    1489: "phoneme",
    38: "sick",
    46: "splice",
    3: "kr-vs-kp",
    182: "satimage",
    40982: "steel-plates-fault",
    307: "vowel",
    1485: "madelon",
    40966: "MiceProtein",
    1487: "ozone-level-8hr",
    # Large (>10k)
    1590: "adult",
    1461: "bank-marketing",
    151: "electricity",
    6: "letter",
    32: "pendigits",
    1468: "cnae-9",
}

_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "experiments"
_ID_NAME_RE = re.compile(r"(^|[_\W])(id|Id|ID)(\b|[_\W])")


def _detect_task_type(y: pd.Series) -> TaskType:
    n_unique = y.nunique(dropna=True)
    dtype = y.dtype
    is_classification_dtype = (
        dtype == object
        or isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_integer_dtype(dtype)
    )
    if n_unique <= 20 and is_classification_dtype:
        if n_unique == 2:
            return TaskType.BINARY_CLASSIFICATION
        return TaskType.MULTICLASS_CLASSIFICATION
    return TaskType.REGRESSION


def _is_monotonic_int_sequence(series: pd.Series) -> bool:
    if not pd.api.types.is_integer_dtype(series):
        return False
    if series.isna().any():
        return False
    diffs = series.diff().dropna().unique()
    return len(diffs) == 1 and diffs[0] == 1


def _drop_id_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return df
    drop = []
    first_col = df.columns[0]
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].nunique(dropna=True) / n <= 0.9:
            continue
        name_looks_like_id = bool(_ID_NAME_RE.search(col))
        is_first_monotonic = col == first_col and _is_monotonic_int_sequence(df[col])
        if name_looks_like_id or is_first_monotonic:
            drop.append(col)
    return df.drop(columns=drop) if drop else df


def load_dataset(dataset_id: int, seed: int = 42, test_size: float = 0.2) -> dict[str, Any]:
    """Load (or cache) an OpenML dataset and return a split + metadata dict."""
    name = SELECTED_CLASSIFICATION.get(dataset_id, f"openml_{dataset_id}")
    dataset_dir = _DATA_ROOT / name
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    meta_path = dataset_dir / "meta.json"

    if train_path.exists() and test_path.exists():
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            target_col = meta["target_col"]
            task_type = TaskType(meta["task_type"])
        else:
            target_col = _infer_target_from_cache(df_train, df_test)
            task_type = _detect_task_type(df_train[target_col])
            _write_meta(meta_path, target_col, task_type, dataset_id, seed)
    else:
        import openml  # imported lazily so tests don't require network setup

        dataset = openml.datasets.get_dataset(dataset_id)
        target_col = dataset.default_target_attribute
        df, _, _, _ = dataset.get_data()
        df = _drop_id_columns(df, target_col)

        y = df[target_col]
        task_type = _detect_task_type(y)
        stratify = y if task_type != TaskType.REGRESSION else None
        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=seed, stratify=stratify
        )

        dataset_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(train_path, index=False)
        df_test.to_csv(test_path, index=False)
        _write_meta(meta_path, target_col, task_type, dataset_id, seed)

    return {
        "dataset_id": dataset_id,
        "dataset_name": name,
        "task_type": task_type,
        "target_col": target_col,
        "train_path": str(train_path.resolve()),
        "test_path": str(test_path.resolve()),
        "df_train": df_train,
        "df_test": df_test,
    }


def _write_meta(
    meta_path: Path,
    target_col: str,
    task_type: TaskType,
    dataset_id: int,
    seed: int,
) -> None:
    meta = {
        "target_col": target_col,
        "task_type": task_type.value,
        "openml_id": dataset_id,
        "split_seed": seed,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _infer_target_from_cache(df_train: pd.DataFrame, df_test: pd.DataFrame) -> str:
    common = [c for c in df_train.columns if c in df_test.columns]
    if not common:
        raise ValueError("Cached train/test have no common columns; cache is corrupt.")
    return common[-1]


_HUMAN_TASK = {
    TaskType.BINARY_CLASSIFICATION: "binary classification",
    TaskType.MULTICLASS_CLASSIFICATION: "multiclass classification",
    TaskType.REGRESSION: "regression",
}


def build_task_description(dataset_info: dict[str, Any]) -> str:
    """Render the task description string fed to every prompt condition."""
    task_type = dataset_info["task_type"]
    human = _HUMAN_TASK[task_type]
    return (
        f"Build a machine learning pipeline for {human} on the dataset at:\n"
        f"  Training data: {dataset_info['train_path']}\n"
        f"  Test data: {dataset_info['test_path']}\n"
        f"  Target column: {dataset_info['target_col']}\n"
        "\n"
        "Requirements:\n"
        "- Load training and test CSVs using pandas.\n"
        "- Preprocess features as needed (handle missing values, encode categoricals, scale numerics).\n"
        "- Train a model on the training set.\n"
        "- Predict on the test set and compute the evaluation metric.\n"
        "- For classification, use balanced_accuracy_score from sklearn.metrics.\n"
        "- For regression, use mean_squared_error from sklearn.metrics and compute negative RMSE as: -sqrt(mse).\n"
        "- Print the result as exactly: SCORE: <number>\n"
        "- Do NOT use any data from the test set during training or preprocessing fitting."
    )
