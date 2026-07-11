"""Tests for the LLM-reasoning trace extraction and mechanical verification."""

from __future__ import annotations

from src.contracts import (
    Decision,
    DistributionalStats,
    InformationStats,
    LandmarkScores,
    MetaFeatures,
    ReasoningTrace,
    SimpleStats,
    TaskType,
)
from src.execution.metrics import extract_reasoning
from src.execution.verification import verify_reasoning


def _meta() -> MetaFeatures:
    return MetaFeatures(
        dataset_id="titanic",
        task_type=TaskType.BINARY_CLASSIFICATION,
        simple=SimpleStats(
            n_rows=712,
            n_cols=11,
            n_numeric=2,
            n_categorical=9,
            missing_ratio_overall=0.088,
            missing_ratio_per_column={"Age": 0.192, "Cabin": 0.775},
            class_balance_ratio=0.622,
            categorical_cardinality={"Name": 712, "Sex": 2},
        ),
        distributional=DistributionalStats(
            skewness_per_numeric={"Age": 0.35, "Fare": 4.63},
            kurtosis_per_numeric={"Age": 0.19, "Fare": 32.28},
            outlier_ratio_per_numeric={"Age": 0.01, "Fare": 0.128},
        ),
        information=InformationStats(
            mutual_info_to_target={"Sex": 0.14, "Name": 0.61},
            mean_abs_correlation=0.107,
            max_pairwise_correlation=0.107,
            target_entropy=0.96,
        ),
        landmarks=LandmarkScores(
            decision_stump_score=0.65,
            naive_bayes_score=0.70,
            one_nn_score=0.55,
            metric_used="balanced_accuracy",
        ),
    )


# ----------------- extract_reasoning ----------------- #

def test_extract_reasoning_parses_valid_line() -> None:
    stdout = (
        "SCORE: 0.784\n"
        'REASONING: {"decisions": [{"step": "scaling", "action": "RobustScaler", '
        '"meta_feature": "distributional.outlier_ratio_per_numeric.Fare", '
        '"observed_value": 0.128, "threshold": 0.05, "rule_id": 7, "applied_to": ["Fare"]}]}\n'
    )
    trace = extract_reasoning(stdout)
    assert trace is not None
    assert len(trace.decisions) == 1
    d = trace.decisions[0]
    assert d.action == "RobustScaler"
    assert d.rule_id == 7


def test_extract_reasoning_missing() -> None:
    assert extract_reasoning("SCORE: 0.5\n") is None


def test_extract_reasoning_malformed_json() -> None:
    assert extract_reasoning("REASONING: {not-json}\n") is None


def test_extract_reasoning_wrong_schema() -> None:
    # Valid JSON, but does not match ReasoningTrace shape.
    assert extract_reasoning('REASONING: {"foo": "bar"}\n') is not None
    # An empty decisions list should still validate.
    trace = extract_reasoning('REASONING: {"decisions": []}\n')
    assert trace is not None and trace.decisions == []


# ----------------- verify_reasoning ----------------- #

_CODE = """
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder
scaler = RobustScaler()
clf = RandomForestClassifier()
"""


def test_verify_faithful_trace() -> None:
    trace = ReasoningTrace(decisions=[
        Decision(
            step="scaling",
            rule_id=7,
            meta_feature="distributional.outlier_ratio_per_numeric.Fare",
            observed_value=0.128,
            threshold=0.05,
            action="RobustScaler",
            applied_to=["Fare"],
        )
    ])
    report = verify_reasoning(trace, _CODE, _meta())
    assert report.faithful is True
    assert report.n_decisions == 1
    assert report.n_faithful == 1
    assert report.verdicts[0].value_matches is True
    assert report.verdicts[0].action_present is True


def test_verify_catches_fabricated_value() -> None:
    trace = ReasoningTrace(decisions=[
        Decision(
            step="scaling",
            meta_feature="distributional.outlier_ratio_per_numeric.Fare",
            observed_value=0.001,  # real is 0.128
            action="RobustScaler",
        )
    ])
    report = verify_reasoning(trace, _CODE, _meta())
    assert report.faithful is False
    v = report.verdicts[0]
    assert v.value_matches is False
    assert v.action_present is True
    assert any("0.001" in r for r in v.reasons)


def test_verify_catches_missing_action() -> None:
    trace = ReasoningTrace(decisions=[
        Decision(step="scaling", action="StandardScaler"),
    ])
    report = verify_reasoning(trace, _CODE, _meta())
    assert report.faithful is False
    v = report.verdicts[0]
    assert v.action_present is False
    assert any("StandardScaler" in r for r in v.reasons)


def test_verify_uncheckable_value_treated_as_faithful() -> None:
    # No meta_feature specified — nothing to verify. Should not fail.
    trace = ReasoningTrace(decisions=[
        Decision(step="model", action="RandomForestClassifier"),
    ])
    report = verify_reasoning(trace, _CODE, _meta())
    assert report.faithful is True
    assert report.verdicts[0].value_matches is None


def test_verify_catches_bad_meta_feature_path() -> None:
    trace = ReasoningTrace(decisions=[
        Decision(
            step="scaling",
            meta_feature="not.a.real.path",
            observed_value=1.0,
            action="RobustScaler",
        )
    ])
    report = verify_reasoning(trace, _CODE, _meta())
    assert report.faithful is False
    assert any("could not resolve" in r for r in report.verdicts[0].reasons)


def test_verify_empty_trace_not_faithful() -> None:
    report = verify_reasoning(ReasoningTrace(decisions=[]), _CODE, _meta())
    assert report.faithful is False
    assert report.n_decisions == 0
    assert "zero decisions" in (report.overall_notes or "")
