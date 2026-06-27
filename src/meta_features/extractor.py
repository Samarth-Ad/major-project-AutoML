import logging

import pandas as pd

from src.contracts import MetaFeatures, TaskType
from src.meta_features.distributional import compute_distributional
from src.meta_features.information import compute_information
from src.meta_features.landmarking import compute_landmarks
from src.meta_features.simple import compute_simple

logger = logging.getLogger(__name__)


def extract(df: pd.DataFrame, target_col: str, dataset_id: str, task_type: TaskType) -> MetaFeatures:
    try:
        simple = compute_simple(df, target_col, task_type)
    except Exception as exc:
        logger.error("compute_simple failed for '%s': %s", dataset_id, exc)
        raise RuntimeError(f"compute_simple failed for dataset '{dataset_id}'") from exc

    try:
        distributional = compute_distributional(df, target_col)
    except Exception as exc:
        logger.error("compute_distributional failed for '%s': %s", dataset_id, exc)
        raise RuntimeError(f"compute_distributional failed for dataset '{dataset_id}'") from exc

    try:
        information = compute_information(df, target_col, task_type)
    except Exception as exc:
        logger.error("compute_information failed for '%s': %s", dataset_id, exc)
        raise RuntimeError(f"compute_information failed for dataset '{dataset_id}'") from exc

    try:
        landmarks = compute_landmarks(df, target_col, task_type)
    except Exception as exc:
        logger.error("compute_landmarks failed for '%s': %s", dataset_id, exc)
        raise RuntimeError(f"compute_landmarks failed for dataset '{dataset_id}'") from exc

    return MetaFeatures(
        dataset_id=dataset_id,
        task_type=task_type,
        simple=simple,
        distributional=distributional,
        information=information,
        landmarks=landmarks,
    )
