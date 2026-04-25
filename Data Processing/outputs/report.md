# Agentic AutoML Pipeline — Report

| Property | Value |
|----------|-------|
| **Status** | ✅ SUCCESS |
| **Pipeline ID** | `776c2d8e` |
| **Total Time** | 80.21s |
| **Steps Completed** | 4/4 |
| **Generated At** | 2026-04-25 11:38:36 UTC |

## 1. Dataset Analysis

| Property | Value |
|----------|-------|
| **Problem Type** |  |
| **Target Column** | `` |
| **Rows** | ? |
| **Columns** | ? |
| **Numeric Features** | ? |
| **Null Values** | ? |
| **Class Imbalanced** | ? |

## 2. Pipeline Decisions
### Steps Included

| # | Step | Justification |
|---|------|---------------|
| 1 | `load_dataset` | Data-driven decision |
| 2 | `select_and_train_models` | Proceed to model training after loading data |
| 3 | `evaluate_models` | Standard evaluation after training |
| 4 | `select_best_model` | Choose best model from training results |
| 5 | `explain_model` | Provide model explanations |

### Steps Skipped

| Step | Reason |
|------|--------|
| `remove_missing_values` | checks.has_nulls is false |
| `handle_class_imbalance` | checks.needs_smote is false or problem type unknown |
| `encode_categorical` | checks.has_categoricals is false |
| `handle_skewness` | checks.needs_skewness is false |
| `normalize_features` | no scaling needed and model choice unknown |
| `feature_engineering` | checks.needs_feat_eng is false |
| `dimensionality_reduction` | checks.needs_pca is false |

## 3. Step Execution Details
| # | Step | Status | Time | Attempts |
|---|------|--------|------|----------|
| 1 | `select_and_train_models` | ✅ success | 9.58s | 1 |
| 2 | `evaluate_models` | ✅ success | 16.25s | 1 |
| 3 | `select_best_model` | ✅ success | 15.99s | 1 |
| 4 | `explain_model` | ✅ success | 14.84s | 1 |

## 4. Model Selection & Validation
**Problem type:** 


**Selection rationale:** Models were selected based on the dataset characteristics ( task). Tree-based models (Random Forest, XGBoost, LightGBM) were preferred for their robustness. Linear models were included as baselines.

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

*Report generated automatically by the Agentic AutoML Pipeline (Pipeline ID: `776c2d8e`)*