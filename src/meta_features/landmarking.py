import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.contracts import LandmarkScores, TaskType


def _is_categorical(s: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s.dtype) and s.nunique() < 20:
        return True
    if not pd.api.types.is_numeric_dtype(s.dtype):
        return True
    return False


def compute_landmarks(df: pd.DataFrame, target_col: str, task_type: TaskType) -> LandmarkScores:
    is_classification = task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION)

    n = min(len(df), 5000)
    sample = df.sample(n=n, random_state=42).dropna()
    if len(sample) == 0:
        sample = df.sample(n=n, random_state=42).copy()
        for col in sample.columns:
            if pd.api.types.is_numeric_dtype(sample[col]):
                sample[col] = sample[col].fillna(sample[col].median())
            else:
                mode = sample[col].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else "__missing__"
                sample[col] = sample[col].fillna(fill)
        sample = sample.dropna(subset=[target_col])

    X_df = sample.drop(columns=[target_col]).copy()
    y = sample[target_col].to_numpy()

    cat_cols = [col for col in X_df.columns if _is_categorical(X_df[col])]
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_df[cat_cols] = enc.fit_transform(X_df[cat_cols].astype(str))

    X = X_df.to_numpy(dtype=float)

    def _score(model, scoring: str) -> float:
        try:
            return float(np.mean(cross_val_score(model, X, y, cv=3, scoring=scoring)))
        except Exception:
            return float("nan")

    if is_classification:
        return LandmarkScores(
            decision_stump_score=_score(DecisionTreeClassifier(max_depth=1, random_state=42), "balanced_accuracy"),
            naive_bayes_score=_score(GaussianNB(), "balanced_accuracy"),
            one_nn_score=_score(KNeighborsClassifier(n_neighbors=1), "balanced_accuracy"),
            metric_used="balanced_accuracy",
        )
    else:
        return LandmarkScores(
            decision_stump_score=_score(DecisionTreeRegressor(max_depth=1, random_state=42), "neg_root_mean_squared_error"),
            naive_bayes_score=_score(Ridge(), "neg_root_mean_squared_error"),
            one_nn_score=_score(KNeighborsRegressor(n_neighbors=1), "neg_root_mean_squared_error"),
            metric_used="neg_rmse",
        )
