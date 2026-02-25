import pandas as pd
import numpy as np


def select_target_column(df):

    best_target = None
    best_score = -np.inf

    for col in df.columns:

        series = df[col]
        unique_count = series.nunique(dropna=True)
        unique_ratio = unique_count / len(df)

        # Skip constant columns
        if unique_count <= 1:
            continue

        score = -np.inf

        # -------------------------
        # CATEGORICAL TARGET
        # -------------------------
        if series.dtype == "object":

            # Skip categorical near-identifiers
            if unique_ratio > 0.8:
                continue

            score = 2 - unique_ratio  # higher score for moderate cardinality

        # -------------------------
        # NUMERIC TARGET (REGRESSION)
        # -------------------------
        elif pd.api.types.is_numeric_dtype(series):

            # DO NOT skip high uniqueness numeric columns
            score = 1  # base score for regression candidates

        if score > best_score:
            best_score = score
            best_target = col

    return best_target

def profile_dataset(file_path):

    df = pd.read_csv(file_path)

    num_rows, num_cols = df.shape
    numerical_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    missing_ratio = df.isnull().mean().mean()

    # ----------------------------
    # SMART TARGET SELECTION
    # ----------------------------

    target_col = select_target_column(df)

    if target_col is not None:
        target = df[target_col]

        if target.dtype == "object":
            imbalance_ratio = target.value_counts(normalize=True).max()
            task_type = "classification"
        else:
            imbalance_ratio = 0
            task_type = "regression"
    else:
        target = None
        imbalance_ratio = None
        task_type = None

    # Skew detection
    skewed_columns = {}
    for col in numerical_cols:
        skew_val = float(df[col].skew())
        if abs(skew_val) > 1:
            skewed_columns[col] = round(skew_val, 3)

    # High-cardinality columns
    high_cardinality = {}
    for col in categorical_cols:
        ratio = df[col].nunique() / len(df)
        if ratio > 0.7:
            high_cardinality[col] = round(float(ratio), 3)

    profile = {
        "task_type": task_type,
        "target_column": target_col,
        "rows": int(num_rows),
        "features": int(num_cols),
        "numerical_features": len(numerical_cols),
        "categorical_features": len(categorical_cols),
        "missing_ratio": round(float(missing_ratio), 3),
        "imbalance_ratio": round(float(imbalance_ratio), 3) if imbalance_ratio is not None else None,
        "skewed_columns": skewed_columns,
        "high_cardinality_columns": high_cardinality
    }

    return profile