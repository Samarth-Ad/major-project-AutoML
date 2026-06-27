import numpy as np
import pandas as pd
from scipy import stats
from src.contracts import DistributionalStats


def _numeric_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            continue
        if pd.api.types.is_integer_dtype(s.dtype) and s.nunique() < 20:
            continue
        if pd.api.types.is_numeric_dtype(s.dtype):
            cols.append(col)
    return cols


def _outlier_ratio(values: np.ndarray) -> float:
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return 0.0
    q1, q3 = np.percentile(clean, [25, 75])
    iqr = q3 - q1
    if iqr == 0.0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return float(np.mean((clean < lower) | (clean > upper)))


def compute_distributional(df: pd.DataFrame, target_col: str) -> DistributionalStats:
    numeric_cols = _numeric_feature_cols(df, target_col)

    skewness_per_numeric: dict[str, float] = {}
    kurtosis_per_numeric: dict[str, float] = {}
    outlier_ratio_per_numeric: dict[str, float] = {}

    for col in numeric_cols:
        vals = df[col].to_numpy(dtype=float)
        skewness_per_numeric[col] = float(stats.skew(vals, nan_policy="omit"))
        kurtosis_per_numeric[col] = float(stats.kurtosis(vals, nan_policy="omit", fisher=True))
        outlier_ratio_per_numeric[col] = _outlier_ratio(vals)

    return DistributionalStats(
        skewness_per_numeric=skewness_per_numeric,
        kurtosis_per_numeric=kurtosis_per_numeric,
        outlier_ratio_per_numeric=outlier_ratio_per_numeric,
    )
