import pandas as pd

from src.contracts import MetaFeatures
from src.conditions.base import PromptCondition


class B1Schema(PromptCondition):
    @property
    def condition_name(self) -> str:
        return "B1"

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
        return (
            "You are an AutoML assistant. Generate a complete scikit-learn pipeline for the following task:\n\n"
            f"{task_description}\n\n"
            "Return a Python script that loads the data, preprocesses it, trains a model, and evaluates on a "
            "held-out test set. Use balanced_accuracy_score for classification or root_mean_squared_error for "
            "regression.\n\n"
            f"{schema_section}"
        )
