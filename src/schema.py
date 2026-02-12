from pydantic import BaseModel, Field
from typing import List, Literal


class Preprocessing(BaseModel):
    missing_value_handling: Literal[
        "mean_imputation",
        "median_imputation",
        "drop_column",
        "none"
    ]

    encoding: Literal[
        "one_hot",
        "label_encoding"
    ]

    scaling: Literal[
        "standard_scaler",
        "none"
    ]


class Strategy(BaseModel):
    preprocessing: Preprocessing

    model_candidates: List[
        Literal[
            "LogisticRegression",
            "RandomForest",
            "XGBoost"
        ]
    ]

    reasoning_summary: str = Field(min_length=5)
