import pandas as pd

from src.contracts import MetaFeatures
from src.conditions.base import PromptCondition


class B0Naive(PromptCondition):
    @property
    def condition_name(self) -> str:
        return "B0"

    def build_prompt(
        self,
        task_description: str,
        meta: MetaFeatures,
        df_head: pd.DataFrame,
        target_col: str,
    ) -> str:
        return (
            "You are an AutoML assistant. Generate a complete scikit-learn pipeline for the following task:\n\n"
            f"{task_description}\n\n"
            "Return a Python script that loads the data, preprocesses it, trains a model, and evaluates on a "
            "held-out test set. Use balanced_accuracy_score for classification or root_mean_squared_error for "
            "regression."
        )
