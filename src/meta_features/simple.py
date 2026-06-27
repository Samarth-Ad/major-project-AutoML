import pandas as pd
from src.contracts import SimpleStats, TaskType


def _classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(s.dtype) and s.nunique() < 20:
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(s.dtype):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def compute_simple(df: pd.DataFrame, target_col: str, task_type: TaskType) -> SimpleStats:
    n_rows, n_cols = df.shape
    numeric_cols, categorical_cols = _classify_columns(df)

    missing_ratio_overall = float(df.isnull().sum().sum() / (n_rows * n_cols))

    feature_cols = [c for c in df.columns if c != target_col]
    missing_ratio_per_column = {col: float(df[col].isnull().mean()) for col in feature_cols}

    if task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        counts = df[target_col].value_counts()
        class_balance_ratio: float | None = float(counts.min() / counts.max())
    else:
        class_balance_ratio = None

    cat_feature_cols = [c for c in categorical_cols if c != target_col]
    categorical_cardinality = {col: int(df[col].nunique()) for col in cat_feature_cols}

    return SimpleStats(
        n_rows=n_rows,
        n_cols=n_cols,
        n_numeric=len(numeric_cols),
        n_categorical=len(categorical_cols),
        missing_ratio_overall=missing_ratio_overall,
        missing_ratio_per_column=missing_ratio_per_column,
        class_balance_ratio=class_balance_ratio,
        categorical_cardinality=categorical_cardinality,
    )
