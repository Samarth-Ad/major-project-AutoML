import pandas as pd


def profile_dataset(file_path):
    df = pd.read_csv(file_path)

    num_rows, num_cols = df.shape

    numerical_cols = df.select_dtypes(include="number").shape[1]
    categorical_cols = num_cols - numerical_cols

    missing_ratio = df.isnull().mean().mean()

    # Assume last column is target
    target = df.iloc[:, -1]

    if target.dtype == "object":
        imbalance_ratio = target.value_counts(normalize=True).max()
        task_type = "classification"
    else:
        imbalance_ratio = 0
        task_type = "regression"

    profile = {
        "task_type": task_type,
        "rows": int(num_rows),
        "features": int(num_cols),
        "numerical_features": int(numerical_cols),
        "categorical_features": int(categorical_cols),
        "missing_ratio": round(float(missing_ratio), 3),
        "imbalance_ratio": round(float(imbalance_ratio), 3),
    }

    return profile
