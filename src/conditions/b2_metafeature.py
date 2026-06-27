import pandas as pd

from src.contracts import MetaFeatures
from src.conditions.base import PromptCondition

_DECISION_RULES = """\
PREPROCESSING RULES:
1. If missing_ratio_overall > 0.05: prefer IterativeImputer over SimpleImputer(strategy='mean').
2. If missing_ratio_overall > 0.30: consider dropping columns with >50% missing before imputation.
3. If any categorical_cardinality > 20: use TargetEncoder instead of OneHotEncoder for those columns.
4. If n_categorical > n_numeric: consider gradient boosting with native categorical support (e.g., LGBMClassifier(categorical_features=...)).

FEATURE ENGINEERING RULES:
5. If max_pairwise_correlation > 0.95: drop one of the near-duplicate feature pair.
6. If any abs(skewness) > 2.0: apply PowerTransformer(method='yeo-johnson') to those features.
7. If any outlier_ratio > 0.05: use RobustScaler instead of StandardScaler for those features.
8. If n_cols / n_rows > 0.1: apply feature selection (SelectKBest with mutual_info) before modeling.
9. If mean_abs_correlation > 0.5 and n_cols > 50: consider PCA for dimensionality reduction.

CLASS IMBALANCE RULES:
10. If class_balance_ratio < 0.3: apply SMOTE or set class_weight='balanced' in the estimator.
11. If class_balance_ratio < 0.1: combine SMOTE with Tomek links; do NOT use plain accuracy as the metric.

MODEL SELECTION HINTS (from landmarking):
12. If decision_stump_score > 0.75: the problem may be simple — try LogisticRegression or shallow DecisionTreeClassifier first.
13. If one_nn_score - naive_bayes_score > 0.1: data has local structure — prefer tree ensembles (RandomForest, XGBoost) or KNN.
14. If naive_bayes_score - one_nn_score > 0.1: features are relatively independent — linear models or GaussianNB are viable.

IMPORTANT: For each pipeline component you choose, cite which meta-feature or rule informed your decision.\
"""


class B2MetaFeatureGuided(PromptCondition):
    @property
    def condition_name(self) -> str:
        return "B2"

    def build_prompt(
        self,
        task_description: str,
        meta: MetaFeatures,
        df_head: pd.DataFrame,
        target_col: str,
    ) -> str:
        dtype_lines = "\n".join(f"  - {col}: {dtype}" for col, dtype in df_head.dtypes.items())
        schema_section = (
            "## Dataset Schema\n\n"
            f"**Rows:** {meta.simple.n_rows}  |  **Columns:** {meta.simple.n_cols}\n\n"
            f"**Task type:** {meta.task_type.value}\n\n"
            f"**Target column:** {target_col}\n\n"
            f"**Column dtypes:**\n{dtype_lines}\n\n"
            f"**First 3 rows:**\n{df_head.head(3).to_markdown(index=False)}\n"
        )
        meta_section = (
            "## Dataset Meta-Feature Profile\n\n"
            f"```json\n{meta.model_dump_json(indent=2)}\n```\n"
        )
        rules_section = (
            "## Decision Rules (apply where conditions match)\n\n"
            f"{_DECISION_RULES}\n"
        )
        return (
            "You are an AutoML assistant. Generate a complete scikit-learn pipeline for the following task:\n\n"
            f"{task_description}\n\n"
            "Return a Python script that loads the data, preprocesses it, trains a model, and evaluates on a "
            "held-out test set. Use balanced_accuracy_score for classification or root_mean_squared_error for "
            "regression.\n\n"
            f"{schema_section}\n"
            f"{meta_section}\n"
            f"{rules_section}"
        )
