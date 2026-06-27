from pathlib import Path

import pandas as pd
import pytest

from src.conditions.b0_naive import B0Naive
from src.conditions.b1_schema import B1Schema
from src.conditions.b2_metafeature import B2MetaFeatureGuided
from src.contracts import MetaFeatures, TaskType
from src.meta_features.extractor import extract

_TITANIC_CSV = Path(__file__).parent.parent / "Data Processing" / "data" / "titanic_survival.csv"
_TASK = "Build a binary classifier on the supplied tabular data and report held-out balanced accuracy."
_TARGET = "Survived"


@pytest.fixture(scope="module")
def titanic_df() -> pd.DataFrame:
    return pd.read_csv(_TITANIC_CSV)


@pytest.fixture(scope="module")
def titanic_meta(titanic_df: pd.DataFrame) -> MetaFeatures:
    return extract(titanic_df, target_col=_TARGET, dataset_id="titanic", task_type=TaskType.BINARY_CLASSIFICATION)


@pytest.fixture(scope="module")
def df_head(titanic_df: pd.DataFrame) -> pd.DataFrame:
    return titanic_df.head(10)


def test_b0_prompt_content(titanic_meta: MetaFeatures, df_head: pd.DataFrame) -> None:
    prompt = B0Naive().build_prompt(_TASK, titanic_meta, df_head, _TARGET)
    assert _TASK in prompt
    assert _TARGET not in prompt, "B0 prompt must not contain the target column name"
    for col in df_head.columns:
        assert col not in prompt, f"B0 prompt should not contain column name '{col}'"
    assert "meta-feature" not in prompt.lower()
    assert "skewness" not in prompt
    assert "mutual_info" not in prompt


def test_b1_prompt_content(titanic_meta: MetaFeatures, df_head: pd.DataFrame) -> None:
    prompt = B1Schema().build_prompt(_TASK, titanic_meta, df_head, _TARGET)
    assert _TASK in prompt
    assert _TARGET in prompt, "B1 prompt must contain the target column name"
    col_names_in_prompt = [col for col in df_head.columns if col in prompt]
    assert len(col_names_in_prompt) >= 3, f"B1 prompt should contain at least 3 column names, found: {col_names_in_prompt}"
    assert str(titanic_meta.simple.n_rows) in prompt
    assert "skewness" not in prompt
    assert "mutual_info" not in prompt
    assert "decision_stump" not in prompt
    assert "DECISION RULES" not in prompt


def test_b2_prompt_content(titanic_meta: MetaFeatures, df_head: pd.DataFrame) -> None:
    prompt = B2MetaFeatureGuided().build_prompt(_TASK, titanic_meta, df_head, _TARGET)
    assert _TASK in prompt
    assert _TARGET in prompt, "B2 prompt must contain the target column name"
    col_names_in_prompt = [col for col in df_head.columns if col in prompt]
    assert len(col_names_in_prompt) >= 3
    assert "meta-feature" in prompt.lower() or "Meta-Feature" in prompt
    assert "skewness" in prompt
    assert "mutual_info" in prompt
    assert "decision_stump_score" in prompt
    assert "SMOTE" in prompt or "class_weight" in prompt


def test_strict_containment(titanic_meta: MetaFeatures, df_head: pd.DataFrame) -> None:
    b0_prompt = B0Naive().build_prompt(_TASK, titanic_meta, df_head, _TARGET)
    b1_prompt = B1Schema().build_prompt(_TASK, titanic_meta, df_head, _TARGET)
    b2_prompt = B2MetaFeatureGuided().build_prompt(_TASK, titanic_meta, df_head, _TARGET)

    assert _TASK in b1_prompt
    assert _TASK in b2_prompt

    cols_in_b1 = [col for col in df_head.columns if col in b1_prompt]
    assert len(cols_in_b1) > 0, "B1 prompt should contain at least one column name"
    for col in cols_in_b1:
        assert col in b2_prompt, f"Column '{col}' found in B1 but not in B2 (B0 ⊂ B1 ⊂ B2 violated)"

    assert len(b0_prompt) < len(b1_prompt), "B1 must contain strictly more info than B0"
    assert len(b1_prompt) < len(b2_prompt), "B2 must contain strictly more info than B1"


def test_all_prompts_nonempty(titanic_meta: MetaFeatures, df_head: pd.DataFrame) -> None:
    for condition in [B0Naive(), B1Schema(), B2MetaFeatureGuided()]:
        prompt = condition.build_prompt(_TASK, titanic_meta, df_head, _TARGET)
        assert len(prompt) > 50, f"{condition.condition_name} prompt too short: {len(prompt)} chars"
