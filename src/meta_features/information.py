import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder
from src.contracts import InformationStats, TaskType


def _is_categorical(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique() < 20:
        return True
    return not pd.api.types.is_numeric_dtype(s.dtype)


def _numeric_feature_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    return [
        col for col in df.columns
        if col != target_col and not _is_categorical(df[col])
    ]


def compute_information(df: pd.DataFrame, target_col: str, task_type: TaskType) -> InformationStats:
    is_classification = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

    feature_cols = [c for c in df.columns if c != target_col]
    cat_cols = [c for c in feature_cols if _is_categorical(df[c])]
    discrete_mask = [c in cat_cols for c in feature_cols]

    # Build feature matrix: ordinal-encode categoricals, drop rows with any NaN in features or target
    work = df[feature_cols + [target_col]].dropna()
    mi_scores: dict[str, float] = {}
    if len(work) >= 2 and len(feature_cols) > 0:
        X_df = work[feature_cols].copy()
        if cat_cols:
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_df[cat_cols] = enc.fit_transform(X_df[cat_cols].astype(str))
        X = X_df.to_numpy(dtype=float)
        y = work[target_col].to_numpy()
        try:
            if is_classification:
                raw = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)
            else:
                raw = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)
            mi_scores = {col: float(val) for col, val in zip(feature_cols, raw)}
        except Exception:
            mi_scores = {col: float("nan") for col in feature_cols}
    else:
        mi_scores = {col: float("nan") for col in feature_cols}

    if is_classification:
        n_classes = int(df[target_col].nunique())
        norm = np.log2(n_classes) if n_classes > 1 else 1.0
        mi_scores = {k: v / norm for k, v in mi_scores.items()}

    # Pearson correlation among numeric feature pairs
    numeric_cols = _numeric_feature_cols(df, target_col)
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True).abs()
        upper_mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        corr_vals = corr.values[upper_mask]
        finite_vals = corr_vals[np.isfinite(corr_vals)]
        mean_abs_corr = float(np.mean(finite_vals)) if len(finite_vals) > 0 else 0.0
        max_pairwise_corr = float(np.max(finite_vals)) if len(finite_vals) > 0 else 0.0
    else:
        mean_abs_corr = 0.0
        max_pairwise_corr = 0.0

    # Target entropy (classification only)
    if is_classification:
        probs = df[target_col].value_counts(normalize=True).values
        target_entropy: float | None = float(stats.entropy(probs, base=2))
    else:
        target_entropy = None

    return InformationStats(
        mutual_info_to_target=mi_scores,
        mean_abs_correlation=mean_abs_corr,
        max_pairwise_correlation=max_pairwise_corr,
        target_entropy=target_entropy,
    )
