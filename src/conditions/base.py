from abc import ABC, abstractmethod

import pandas as pd

from src.contracts import MetaFeatures


class PromptCondition(ABC):
    @abstractmethod
    def build_prompt(
        self,
        task_description: str,
        meta: MetaFeatures,
        df_head: pd.DataFrame,
        target_col: str,
    ) -> str:
        """Build the full prompt string for the LLM."""
        ...

    @property
    @abstractmethod
    def condition_name(self) -> str:
        """Return 'B0', 'B1', or 'B2'."""
        ...
