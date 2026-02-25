from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class MissingConfig(BaseModel):
    strategy: str  # "none", "mean", "median", "mode"
    threshold: float = 0.4


class EncodingConfig(BaseModel):
    method: str  # "label", "one_hot", "target"
    drop_high_cardinality_ratio: float = 0.7


class ScalingConfig(BaseModel):
    method: str  # "none", "standard", "minmax"


class SkewConfig(BaseModel):
    enabled: bool
    threshold: float = 1.0
    method: str = "log1p"


class CorrelationConfig(BaseModel):
    enabled: bool
    threshold: float = 0.9


class PreprocessingConfig(BaseModel):
    missing: MissingConfig
    encoding: EncodingConfig
    scaling: ScalingConfig
    skew_handling: SkewConfig
    correlation_pruning: CorrelationConfig


class ModelConfig(BaseModel):
    name: str
    parameters: Dict[str, Any]


class SplitConfig(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    stratified: bool = True


class ModelingConfig(BaseModel):
    candidates: List[ModelConfig]
    cross_validation_folds: int = 5
    evaluation_metric: str
    split: SplitConfig   
class Strategy(BaseModel):
    task_type: str  # "classification" or "regression"
    target_column: str
    preprocessing: PreprocessingConfig
    modeling: ModelingConfig
