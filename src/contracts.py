from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum

class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"

class SimpleStats(BaseModel):
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    missing_ratio_overall: float
    missing_ratio_per_column: dict[str, float]
    class_balance_ratio: Optional[float]  # null for regression
    categorical_cardinality: dict[str, int]

class DistributionalStats(BaseModel):
    skewness_per_numeric: dict[str, float]
    kurtosis_per_numeric: dict[str, float]
    outlier_ratio_per_numeric: dict[str, float]  # IQR-based

class InformationStats(BaseModel):
    mutual_info_to_target: dict[str, float]
    mean_abs_correlation: float
    max_pairwise_correlation: float
    target_entropy: Optional[float]  # null for regression

class LandmarkScores(BaseModel):
    decision_stump_score: float
    naive_bayes_score: float
    one_nn_score: float
    metric_used: Literal["balanced_accuracy", "neg_rmse"]

class MetaFeatures(BaseModel):
    """The single source of truth for what we extract from a dataset."""
    dataset_id: str
    task_type: TaskType
    simple: SimpleStats
    distributional: DistributionalStats
    information: InformationStats
    landmarks: LandmarkScores

class RunResult(BaseModel):
    """One pipeline execution result."""
    dataset_id: str
    condition: Literal["B0", "B1", "B2"]
    llm_backend: str
    seed: int
    iteration: int
    success: bool
    error_category: Optional[str]  # None if success
    error_message: Optional[str]
    test_score: Optional[float]
    runtime_seconds: float
    generated_code_path: str

class ErrorCategory(str, Enum):
    """Taxonomy of LLM code-generation failures."""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    API_HALLUCINATION = "api_hallucination"
    SHAPE_MISMATCH = "shape_mismatch"
    TYPE_ERROR = "type_error"
    DEPRECATED_API = "deprecated_api"
    METRIC_MISMATCH = "metric_mismatch"
    TIMEOUT = "timeout"
    RUNTIME_OTHER = "runtime_other"
    RESOURCE_LIMIT = "resource_limit"

class GeneratedPipeline(BaseModel):
    """What the LLM produces — the code to execute."""
    code: str
    condition: Literal["B0", "B1", "B2"]
    llm_backend: str
    dataset_id: str
    seed: int
    iteration: int
