# Agentic AutoML Pipeline — Report

| Property | Value |
|----------|-------|
| **Status** | ❌ FAILED |
| **Pipeline ID** | `6450509b` |
| **Total Time** | 265.11s |
| **Steps Completed** | 6/10 |
| **Generated At** | 2026-03-24 16:21:02 UTC |

## 1. Dataset Analysis

| Property | Value |
|----------|-------|
| **Problem Type** | multiclass_classification |
| **Target Column** | `ocean_proximity` |
| **Rows** | ? |
| **Columns** | ? |
| **Numeric Features** | ? |
| **Null Values** | ? |
| **Class Imbalanced** | ? |

## 2. Pipeline Decisions
### Steps Included

| # | Step | Justification |
|---|------|---------------|
| 1 | `load_dataset` | Load raw data into memory. |
| 2 | `remove_missing_values` | Nulls present (1% in total_bedrooms) need imputation/removal. |
| 3 | `handle_class_imbalance` | Target classes are imbalanced; SMOTE required per checks. |
| 4 | `encode_categorical` | One categorical feature (ocean_proximity) needs encoding. |
| 5 | `handle_skewness` | Several numeric columns are highly skewed; transformation needed. |
| 6 | `feature_engineering` | Correlated numeric features suggest interaction or ratio features could improve model. |
| 7 | `normalize_features` | Logistic regression and SVM require feature scaling. |
| 8 | `select_and_train_models` | Train candidate models. |
| 9 | `evaluate_models` | Assess performance on validation set. |
| 10 | `select_best_model` | Choose model with highest multiclass metric. |
| 11 | `explain_model` | Provide interpretability for final model. |

### Steps Skipped

| Step | Reason |
|------|--------|
| `dimensionality_reduction` | PCA not required per checks |

## 3. Step Execution Details
| # | Step | Status | Time | Attempts |
|---|------|--------|------|----------|
| 1 | `remove_missing_values` | ✅ success | 9.92s | 1 |
| 2 | `handle_class_imbalance` | ✅ success | 13.02s | 1 |
| 3 | `encode_categorical` | ✅ success | 10.79s | 1 |
| 4 | `handle_skewness` | ✅ success | 28.51s | 2 |
| 5 | `feature_engineering` | ✅ success | 27.42s | 1 |
| 6 | `normalize_features` | ✅ success | 19.98s | 1 |
| 7 | `select_and_train_models` | ❌ failed | 128.45s | 4 |
| 8 | `evaluate_models` | ⏭️ skipped | 0.00s | 0 |
| 9 | `select_best_model` | ⏭️ skipped | 0.00s | 0 |
| 10 | `explain_model` | ⏭️ skipped | 0.00s | 0 |

## 4. Model Selection & Validation
**Problem type:** multiclass_classification


**Selection rationale:** Models were selected based on the dataset characteristics (multiclass_classification task). Tree-based models (Random Forest, XGBoost, LightGBM) were preferred for their robustness. Linear models were included as baselines.

## 5. Output Files

| File | Path |
|------|------|
| **Cleaned Dataset** | `outputs\cleaned_data.csv` |
| **Notebook** | `outputs/pipeline.ipynb` |
| **Pipeline Script** | `generated_code/pipeline_script.py` |

## 6. System Architecture

```
CSV Input
    │
    ▼
┌──────────────────────┐
│   Analyzer Agent     │  ← DataUnderstandingAgent (Pass 1)
│   (Data Profiling)   │     Detects: nulls, types, skew, imbalance
└──────────┬───────────┘
           │ data profile
           ▼
┌──────────────────────┐
│   Planner Agent      │  ← DataUnderstandingAgent (Pass 2)
│   (Pipeline Plan)    │     Outputs: structured JSON plan
└──────────┬───────────┘
           │ step list + reasoning
           ▼
┌──────────────────────┐
│   Meta Agent         │  ← MasterAgent + AgentBuilder
│   (Agent Factory)    │     Dynamically creates agents per step
└──────────┬───────────┘
           │ built agents
           ▼
┌──────────────────────────────────────────┐
│   Dynamic Pipeline (Scheduler)           │
│                                          │
│   ┌─────────┐  ┌─────────┐  ┌────────┐  │
│   │ Impute  │→ │ Encode  │→ │ Scale  │  │
│   └─────────┘  └─────────┘  └────────┘  │
│        │            │            │       │
│   ┌─────────┐  ┌─────────┐  ┌────────┐  │
│   │Feature  │→ │ Train   │→ │Evaluate│  │
│   │Engineer │  │ Models  │  │ Models │  │
│   └─────────┘  └─────────┘  └────────┘  │
│                                          │
└──────────────────┬───────────────────────┘
                   │ results
                   ▼
    ┌──────────────────────────────────┐
    │         Output Generation        │
    │  ┌────────────┐ ┌─────────────┐  │
    │  │ Doc Agent  │ │ Notebook    │  │
    │  │ report.md  │ │ Generator   │  │
    │  └────────────┘ └─────────────┘  │
    │  ┌────────────┐ ┌─────────────┐  │
    │  │cleaned_data│ │  model.pkl  │  │
    │  │   .csv     │ │             │  │
    │  └────────────┘ └─────────────┘  │
    └──────────────────────────────────┘
```

---

*Report generated automatically by the Agentic AutoML Pipeline (Pipeline ID: `6450509b`)*