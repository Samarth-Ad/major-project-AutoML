from __future__ import annotations

from typing import Optional

import pandas as pd

from src.conditions.base import PromptCondition
from src.conditions.b2_metafeature import B2MetaFeatureGuided
from src.contracts import MetaFeatures


class StrategicLM:
    """Sends a condition-built prompt to an LLM and returns the raw response."""

    def __init__(self, llm_backend: str = "stub") -> None:
        self.llm_backend = llm_backend

    def generate_strategy(
        self,
        task_description: str,
        meta: MetaFeatures,
        df_head: pd.DataFrame,
        target_col: str,
        condition: Optional[PromptCondition] = None,
    ) -> str:
        if condition is None:
            condition = B2MetaFeatureGuided()
        prompt = condition.build_prompt(task_description, meta, df_head, target_col)
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        if self.llm_backend == "stub":
            return f"[STUB RESPONSE — prompt length: {len(prompt)}]"
        raise NotImplementedError(f"LLM backend '{self.llm_backend}' not implemented")
