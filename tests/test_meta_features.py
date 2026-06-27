import math
from pathlib import Path

import pandas as pd
import pytest

from src.contracts import MetaFeatures, TaskType
from src.meta_features.extractor import extract

# titanic_survival.csv is the standard 891-row Titanic train set
_TITANIC_CSV = Path(__file__).parent.parent / "Data Processing" / "data" / "titanic_survival.csv"


@pytest.fixture(scope="module")
def titanic_meta() -> MetaFeatures:
    df = pd.read_csv(_TITANIC_CSV)
    return extract(df, target_col="Survived", dataset_id="titanic", task_type=TaskType.BINARY_CLASSIFICATION)


def test_is_valid_instance(titanic_meta: MetaFeatures) -> None:
    assert isinstance(titanic_meta, MetaFeatures)


def test_shape(titanic_meta: MetaFeatures) -> None:
    assert titanic_meta.simple.n_rows == 891
    assert titanic_meta.simple.n_cols == 12


def test_class_balance_ratio(titanic_meta: MetaFeatures) -> None:
    ratio = titanic_meta.simple.class_balance_ratio
    assert ratio is not None
    assert 0.5 <= ratio <= 0.7


def test_decision_stump_score(titanic_meta: MetaFeatures) -> None:
    score = titanic_meta.landmarks.decision_stump_score
    assert not math.isnan(score), "decision_stump_score is NaN"
    assert 0.55 <= score <= 0.85, f"decision_stump_score {score:.3f} outside [0.55, 0.85]"


def test_skewness_finite(titanic_meta: MetaFeatures) -> None:
    for col, val in titanic_meta.distributional.skewness_per_numeric.items():
        assert math.isfinite(val), f"skewness for '{col}' is not finite: {val}"


def test_target_entropy_positive(titanic_meta: MetaFeatures) -> None:
    assert titanic_meta.information.target_entropy is not None
    assert titanic_meta.information.target_entropy > 0


def test_print_json(titanic_meta: MetaFeatures) -> None:
    print(titanic_meta.model_dump_json(indent=2))
