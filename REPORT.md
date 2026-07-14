# Meta-Feature-Guided Prompting for LLM-Driven Tabular AutoML: A System Study on Faithfulness and Correctness

**Status:** In-progress research report (paper draft basis)
**Working dates:** 2026-07-09 to 2026-07-11
**Author of record:** Samarth Adhikari (Major Project, undergraduate thesis)
**Codebase:** github.com/Major-Proj-AutoML/*  +  github.com/Samarth-Ad/major-project-AutoML

---

## Abstract

We investigate whether providing an LLM with **computed dataset meta-features** and **explicit decision rules** improves the quality of auto-generated scikit-learn pipelines for tabular classification, compared to two weaker prompting baselines (naive and schema-only). We construct a **microservice-based experimental platform** (7 repositories, Postgres+Redis, Docker-Compose) that (a) uploads user CSVs, (b) computes 4 groups of meta-features, (c) generates pipelines via Ollama-hosted LLMs under three prompt conditions, (d) executes generated code in subprocess isolation, and (e) mechanically **verifies the LLM's reasoning trace** against both the generated code (AST parse) and the underlying meta-feature values (dotted-path lookup with tolerance).

Preliminary results across two datasets (Titanic, n=891; Telco Customer Churn, n=7043) and 40+ runs show that:

1. **Meta-feature-guided prompting (B2) is NOT universally superior.** Mean balanced accuracy: **B0 = 0.777, B1 = 0.789, B2 = 0.685** across cross-dataset means. B2 shows 33% failure rate vs 100% success for B0/B1.
2. **A specific rule interaction produces a deterministic failure mode.** Rules 3 (TargetEncoder for high-cardinality categoricals) and 4 (gradient boosting for many categoricals) co-firing on datasets with near-unique identifier columns (e.g. Titanic's `Name`, cardinality 712 in 712-row training set) causes pipelines to collapse to majority-class prediction (balanced accuracy = 0.500). N = 5, deterministic.
3. **Faithfulness and correctness are orthogonal.** Every trap-inducing B2 run has a fully **faithful** trace — all cited meta-feature values match the ground truth, and all cited actions appear in the generated code AST. The LLM was 100% honest about doing something catastrophically wrong. **28/28 individual decisions verified as faithful across 10 runs.**
4. **Silent leakage bugs are detectable via score-sanity thresholding.** A B1 run on Telco scored perfectly (1.000 balanced accuracy) due to the LLM forgetting to drop the target column; a post-hoc guardrail (`SCORE ≥ 0.995` on classification) flags this class of failure.

The system contributes: (i) a **verifiable AutoML pipeline** that separates faithfulness from correctness, (ii) a **microservice-based experimental substrate** appropriate for reproducible LLM-code-generation studies, and (iii) **empirical evidence** that meta-feature guidance can encode adversarial rule interactions that faithful LLMs will nevertheless execute.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work (Positioning)](#2-related-work-positioning)
3. [System Design & Architecture](#3-system-design--architecture)
4. [Methodology](#4-methodology)
5. [Experimental Setup](#5-experimental-setup)
6. [Results](#6-results)
7. [Findings in Detail](#7-findings-in-detail)
8. [Discussion](#8-discussion)
9. [Limitations & Threats to Validity](#9-limitations--threats-to-validity)
10. [Future Work](#10-future-work)
11. [Conclusion](#11-conclusion)
12. [Appendix A — B2 Prompt (verbatim)](#appendix-a--b2-prompt-verbatim)
13. [Appendix B — The 14 B2 Decision Rules](#appendix-b--the-14-b2-decision-rules)
14. [Appendix C — Complete Run Tables](#appendix-c--complete-run-tables)
15. [Appendix D — Example Reasoning Trace + Verification Report](#appendix-d--example-reasoning-trace--verification-report)
16. [Appendix E — Error Taxonomy](#appendix-e--error-taxonomy)
17. [Appendix F — Repository Layout](#appendix-f--repository-layout)
18. [Appendix G — Reproducibility](#appendix-g--reproducibility)

---

## 1. Introduction

### 1.1 Motivation

Large language models (LLMs) increasingly generate machine-learning code from natural-language descriptions. In the tabular AutoML setting, an LLM presented with a dataset description can produce a full scikit-learn pipeline: load data, preprocess (impute, encode, scale, transform), train a model, and evaluate. Two open questions:

- **Does giving the LLM more information about the dataset (schema, statistics, decision heuristics) improve output quality?**
- **When the LLM narrates its reasoning — for example, "I chose RobustScaler because Fare has outlier_ratio = 0.128 > 0.05" — can we trust that narration?**

The first question is the classical AutoML-prompting question. The second is the *faithfulness* question — increasingly important as LLM-generated code enters research and production pipelines. If a reviewer, oncall engineer, or thesis committee cannot verify that the LLM's stated reasoning actually corresponds to the code it produced, the narration is decoration, not evidence.

### 1.2 Research questions

Formalized as five questions:

- **RQ1.** Does meta-feature-guided prompting (B2) produce higher predictive performance than schema-only (B1) or naive (B0) prompting across heterogeneous tabular datasets?
- **RQ2.** Does the inclusion of meta-features reduce the rate of code-generation failures (syntax errors, import errors, shape mismatches, timeouts)?
- **RQ3.** Does meta-feature guidance shorten the iterative refinement loop — i.e., produce a working pipeline in fewer LLM calls?
- **RQ4.** How does the effect of meta-feature guidance vary across LLM backends of different sizes and capabilities?
- **RQ5.** Are the gains from meta-feature guidance consistent across small / medium / large datasets, or concentrated in a specific size or difficulty regime?

We also add, based on session-2 empirical work:

- **RQ6 (added).** When the LLM produces a structured reasoning trace, does the code it emits actually match the trace? And does faithful reasoning correlate with pipeline correctness on the test set?

### 1.3 Contributions

1. **A structured reasoning-trace protocol.** The LLM emits `REASONING: {json}` alongside `SCORE: <number>`, listing per-decision `(rule_id, meta_feature, observed_value, action, applied_to)`. This lets us mechanically audit each claim.
2. **A verification module.** For every decision: (a) resolve the dotted `meta_feature` path against the ground-truth `MetaFeatures` object and confirm `observed_value` matches within 1e-3 tolerance; (b) AST-parse the generated code and confirm the `action` symbol appears as an import or a name reference. If either check fails, the decision is marked unfaithful; if any decision fails, the run is flipped to `error_category = reasoning_unfaithful` and excluded from correctness statistics.
3. **A published finding.** Rules 3 + 4 in the B2 rule set co-fire deterministically on datasets with near-unique identifier columns, producing pipelines that faithfully follow the rules but collapse to majority-class prediction (balanced accuracy = 0.500). This is a *rule interaction failure*, not an LLM hallucination — the LLM does exactly what the rules say.
4. **A microservice experimental platform.** Seven independently versioned repositories, Docker-Compose orchestrated, with Postgres for persistent state and Redis+RQ for asynchronous LLM job execution. Fully self-hostable, reproducible from `docker compose up`.
5. **A decoupling of faithfulness and correctness as measurable, orthogonal quantities.** Empirically: 100% faithful trace, 50% balanced-accuracy score — coexisting in the same run.

---

## 2. Related Work (Positioning)

We build on three lines of work; details are elided here but should be filled in for the paper:

- **Meta-feature-based AutoML** (Rice's algorithm selection framework; MFE / pymfe; AutoSklearn's warm-starting via meta-learning). Our meta-feature set is a scikit-learn-native subset: simple descriptive stats, distributional (skew/kurtosis/outliers), information (mutual info, correlation, target entropy), and landmarking (decision-stump / naive-Bayes / 1-NN accuracy from 3-fold CV).
- **LLM code generation for data science** (Codex/GPT-4-as-analyst papers; the "LIDA / ChatGPT-code-interpreter" line). These works do not typically evaluate against a controlled prompting-ablation study on tabular data with a rigorous reasoning audit.
- **Faithfulness in LLM reasoning** (chain-of-thought interpretability critiques; "post-hoc rationalization" studies). Existing work asks whether *natural-language* reasoning matches internal LLM state. We instead ask whether *structured* reasoning matches the *code artifact* the LLM produced. This is a strictly stronger, machine-checkable form of faithfulness.

Explicit non-claim: we do not claim novelty on the meta-features themselves, or on the general idea of prompting an LLM to write ML code. Our contribution is the *structured verification layer* plus *the empirical demonstration that faithful ≠ correct in this domain*.

---

## 3. System Design & Architecture

### 3.1 Design goals

- **Reproducibility.** A collaborator on a fresh machine runs `docker compose up`, imports a Postman collection, and reproduces every experiment.
- **Polyrepo separation.** Each service is independently versioned, testable in isolation (SQLite in-memory + mocked upstream), and deployable. Enables future collaborators to contribute to a single service without cloning everything.
- **Shared code, single source of truth.** All contracts (Pydantic models), meta-feature extractors, and prompt conditions live in `automl-reusables`, installed as a git-URL dependency by every service.
- **Asynchronous LLM execution.** LLM calls are slow (30–90s cloud, 5–30s local). Enqueue with 202 Accepted; workers process in the background; clients poll for results.

### 3.2 Repository layout

Seven sibling repositories at `D:\Major Project\` (locally) and under two GitHub owners:

```
D:\Major Project\
├── major-project-AutoML\            github.com/Samarth-Ad/major-project-AutoML     (personal)
├── automl-reusables\                github.com/Major-Proj-AutoML/automl-reusables    (shared library)
├── automl-infra\                    github.com/Major-Proj-AutoML/automl-infra        (compose + schema)
├── automl-data-service\             ...automl-data-service                            (:8001)
├── automl-metafeatures-service\     ...automl-metafeatures-service                    (:8002)
├── automl-generation-service\       ...automl-generation-service                      (:8003 + worker)
├── automl-analysis-service\         ...automl-analysis-service                        (:8004)
└── automl-gateway\                  ...automl-gateway                                 (:8000)
```

`automl-reusables` is imported at runtime as module `src` (historical decision) — every service does `from src.contracts import MetaFeatures` etc. It is installed via `pip install "automl-reusables @ git+https://github.com/Major-Proj-AutoML/automl-reusables.git@main"` in each service's `pyproject.toml`, and additionally bind-mounted at `/opt/automl-reusables` in Docker so host edits to shared code reflect immediately.

### 3.3 Service responsibilities

| Service | Port | Storage | Purpose |
|---|---|---|---|
| **data-service** | 8001 | Postgres `datasets` table; disk CSVs in `automl_data` volume | Upload CSV or fetch OpenML; save `original.csv` (pre-split), `train.csv`, `test.csv`; expose registry |
| **metafeatures-service** | 8002 | Postgres `meta_features` (JSONB) | Compute 4 meta-feature groups from the training set; cache indefinitely |
| **generation-service** | 8003 | Postgres `run_results`, `sweep_jobs`; Redis queue `automl-generation` | Enqueue single runs and sweeps; separate worker container consumes jobs, calls Ollama, executes subprocess, verifies B2 traces |
| **analysis-service** | 8004 | (read-only from Postgres) | Expose RQ1–RQ5 statistics, faithfulness reports, rule usage |
| **gateway** | 8000 | (stateless) | Proxy to backends; CORS for frontend; composed workflows (upload+extract, full-run) |

### 3.4 Data flow (end-to-end run)

```
User (Postman/frontend)
  │
  │  POST /workflows/full-run  {file, target_col, condition, llm_backend, ...}
  ▼
Gateway :8000  (proxies + composes)
  │
  ├─►  POST /datasets  ────────────►  data-service :8001
  │       (upload CSV, register in Postgres, cache 3 CSVs on disk)
  │
  ├─►  POST /meta-features/{id}  ─►  metafeatures-service :8002
  │       (fetches dataset info from data-service via HTTP,
  │        reads training CSV, computes 4 groups, caches in Postgres)
  │
  └─►  POST /runs  ────────────────►  generation-service :8003
          (creates Postgres row, enqueues RQ job, returns rq_job_id)
                    │
                    ▼    (async)
             Redis queue
                    │
                    ▼
        generation-worker container
          │
          ├─►  GET /datasets/{id}  ─►  data-service (get train/test paths)
          ├─►  POST /api/generate  ─►  Ollama (call_llm, retries on connection error)
          ├─►  subprocess.run(...)  ─►  isolated Python subprocess (execute_pipeline)
          │        │
          │        └─►  reads train.csv, fits pipeline, prints:
          │                    SCORE: 0.784
          │                    REASONING: {"decisions": [...]}   (B2 only)
          │
          ├─►  extract_score(stdout)             (regex parse)
          ├─►  extract_reasoning(stdout)         (B2 only, regex + Pydantic validate)
          ├─►  verify_reasoning(trace, code, meta)  (B2 only, AST + dotted-path)
          │        │
          │        └─►  produces VerificationReport (per-decision verdicts)
          │
          └─►  Writes RunResult + reasoning_trace + verification_report
                    to Postgres run_results (JSONB columns) AND as
                    sidecar .trace.json / .verification.json files

Analysis :8004 reads run_results and exposes:
  /analysis/summary  /analysis/errors  /analysis/iterations
  /analysis/models   /analysis/size-stratified  /analysis/wilcoxon
  /analysis/traces   /analysis/rule-usage      (session-2 additions)
```

### 3.5 Persistence model

**PostgreSQL 16** (container `Auto-ML-Postgres`, host port 5433):

- `datasets(id, name, source, openml_id, target_col, task_type, train_path, test_path, n_rows, n_cols, created_at)`
- `meta_features(id, dataset_id, features JSONB, computed_at)`
- `run_results(id, dataset_id, condition, llm_backend, seed, iteration, success, test_score, error_category, error_message, iterations_used, max_iterations, runtime_seconds, generated_code_path, reasoning_trace JSONB, verification_report JSONB, created_at)`
- `sweep_jobs(id, rq_job_id, status, params JSONB, total_cells, completed_cells, failed_cells, error_message, created_at, updated_at, completed_at)`

**Redis 7** (container `Auto-ML-Redis`, host port 6380) — RQ queue `automl-generation` only; no cache use.

**Filesystem** (Docker volume `automl_data`, mounted at `/data` inside services):

```
/data/experiments/custom/<dataset_stem>/
  ├── original.csv     ← unmodified pre-split (deliverable to users)
  ├── train.csv        ← stratified train split
  ├── test.csv         ← held-out test split
  └── meta.json        ← target_col, task_type, split seed
```

Generated code + sidecars in `/opt/automl-reusables/logs/runs/`:

```
train_B2_seed42_iter0_<hash>.py                 ← LLM-generated Python
train_B2_seed42_iter0_<hash>.py.stdout.txt      ← subprocess stdout
train_B2_seed42_iter0_<hash>.py.stderr.txt      ← subprocess stderr
train_B2_seed42_iter0_<hash>.trace.json         ← B2 reasoning trace (Pydantic-validated)
train_B2_seed42_iter0_<hash>.verification.json  ← per-decision audit report
train_B2_seed42_iter0_<hash>.reasoning_raw.txt  ← only if trace failed to parse (debug)
```

---

## 4. Methodology

### 4.1 Meta-feature extraction

For every training set we compute four groups (all in `automl-reusables/src/meta_features/`):

**Simple** (`simple.py`):
- `n_rows`, `n_cols` (of the training set only, not the full CSV)
- `n_numeric`, `n_categorical` (a column is categorical if bool, or integer with `nunique < 20`, or non-numeric)
- `missing_ratio_overall` = total NA cells ÷ total cells
- `missing_ratio_per_column`: `{col: NA_fraction}` for feature columns
- `class_balance_ratio` = min-class-count / max-class-count (classification only; `None` for regression)
- `categorical_cardinality`: `{col: nunique}` for categorical feature columns

**Distributional** (`distributional.py`):
- `skewness_per_numeric`, `kurtosis_per_numeric` (scipy.stats.skew / kurtosis, with `nan_policy="omit"`)
- `outlier_ratio_per_numeric` — IQR method: fraction of values outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR]

**Information** (`information.py`):
- `mutual_info_to_target`: `{col: MI}` via `mutual_info_classif` / `mutual_info_regression`, normalized by log2(n_classes) for classification. Categoricals are ordinal-encoded first; rows with any NA in features or target are dropped.
- `mean_abs_correlation`, `max_pairwise_correlation` on numeric features (Pearson, upper triangle, absolute value, finite values only)
- `target_entropy` (base-2, classification only)

**Landmarking** (`landmarking.py`):
- Cross-validated (3-fold) accuracy of three baseline learners on a random sample of ≤5000 rows:
  - `DecisionTreeClassifier(max_depth=1)` (decision stump)
  - `GaussianNB`
  - `KNeighborsClassifier(n_neighbors=1)` (1-NN)
- Metric: `balanced_accuracy` (classification) or `neg_root_mean_squared_error` (regression)
- Categoricals ordinal-encoded; NAs imputed by median/mode when a full row cannot be dropped

The full `MetaFeatures` Pydantic model is the sole source of truth for these values; both the LLM (via prompt) and the verifier (via dotted-path lookup) reference it.

### 4.2 Prompt conditions

Three conditions, forming a strict information-content hierarchy `B0 ⊂ B1 ⊂ B2`:

**B0 (naive)** — only the task description:
> "Build a machine learning pipeline for {task} on the dataset at: {train_path}. Requirements: load CSV, preprocess, train, evaluate on held-out test set, print SCORE: <number>. For classification use balanced_accuracy_score; for regression use root_mean_squared_error (negated). Do NOT use any test data during training."

**B1 (schema)** — B0 + dataset schema section: rows, cols, task type, target column, per-column dtypes, first 3 rows of the training frame (as a Markdown table).

**B2 (meta-feature-guided)** — B1 + full MetaFeatures JSON dump + 14 explicit decision rules (see [Appendix B](#appendix-b--the-14-b2-decision-rules)) + a mandatory structured reasoning trace requirement (see [Appendix A](#appendix-a--b2-prompt-verbatim)).

**Import discipline (all conditions).** In session 1 we observed the LLM referring to classes it had not imported (e.g. `RobustScaler()` without `from sklearn.preprocessing import RobustScaler`). We added an explicit hardening clause to all three prompts: *"The script must include ALL import statements at the top. Every class and function you use must be explicitly imported. The script will run in an isolated environment with no pre-imported names."* This reduced `missing_name` errors but did not eliminate them.

### 4.3 Execution protocol

Every generated pipeline runs in an isolated Python subprocess:

1. Code is written to a persisted file under `/opt/automl-reusables/logs/runs/`.
2. `subprocess.run([python, code_path], capture_output=True, timeout=timeout_seconds)`.
3. **Timeout** (default 180s per attempt) → `error_category = timeout`.
4. **Non-zero exit code** → stderr is classified by regex into one of: `syntax_error`, `import_error`, `missing_name`, `api_hallucination`, `shape_mismatch`, `type_error`, `deprecated_api`, `resource_limit`, `runtime_other`. See [Appendix E](#appendix-e--error-taxonomy).
5. **Missing / unparseable SCORE line** → `runtime_other`.
6. **Score ≥ 0.995** (classification only) → `suspicious_leakage` (post-hoc guardrail; retroactively fires when a run scores implausibly high, indicating a leakage bug in the generated code).
7. Otherwise: **B2 only** proceeds to reasoning verification (§4.4). B0/B1 return success as-is.

### 4.4 Reasoning verification (B2)

After a successful subprocess run for a B2 cell, the worker:

1. Reads the subprocess stdout sidecar.
2. Parses the `REASONING: {json}` line via a Pydantic model `ReasoningTrace` containing a list of `Decision` objects.
3. For each `Decision`:
   - **Value check.** Follow the dotted `meta_feature` path (e.g. `distributional.outlier_ratio_per_numeric.Fare`) into the `MetaFeatures` object. Compare `observed_value` to the resolved value with `math.isclose(..., abs_tol=1e-3)`.
   - **Action check.** AST-parse the generated code (`ast.parse`, then walk); collect all `Name`, `Attribute`, `Import`, and `ImportFrom` identifiers. The `action` string must appear in this set.
   - Decision is **faithful** iff both checks pass (or the value check is not applicable because no `meta_feature` was cited).
4. The `VerificationReport` aggregates: `n_decisions`, `n_faithful`, `faithful` (all decisions faithful), and per-decision `DecisionVerdict`.
5. If `faithful = false` OR the trace was missing/malformed, the run is flipped to `success = false, error_category = reasoning_unfaithful, error_message = <notes>`. The trace and report are still persisted (in DB JSONB columns and as sidecar files) so the failure can be inspected.

### 4.5 Iterative refinement loop

Every cell allows up to `max_iter` attempts (default 3). After a failed attempt:

1. The prompt is augmented with `## Previous Attempt Failed\n{error message}\n{first 40 lines of failing code}` via `build_error_feedback()`.
2. When `error_category = missing_name`, an additional hint is appended: *"Check that EVERY class, function, and constant you reference has a corresponding import statement at the top."*
3. The LLM is called again with `seed + iteration` as the sampling seed.
4. On success, that attempt's result is persisted; on final failure, the last attempt is persisted with its `iterations_used = max_iter`.

---

## 5. Experimental Setup

### 5.1 Datasets

We evaluate on two publicly available tabular classification datasets. Both are uploaded as custom CSVs (not fetched via OpenML API) so `original.csv` is preserved as a downloadable artifact.

**Titanic** (Kaggle competition data):
- 891 rows × 12 cols; binary target `Survived`
- Extensively studied; known best balanced-accuracy ceiling ≈ 0.85
- **Contains near-unique identifier columns:** `Name` (712 unique in 712-row train), `Ticket` (571 unique). Data leakage bait.
- Missing data: `Age` ~20%, `Cabin` ~77%

**Telco Customer Churn** (IBM open dataset):
- 7,043 rows × 21 cols; binary target `Churn`
- Real-world business messiness: `TotalCharges` stored as string with 11 space-only entries; `customerID` is unique per row.
- Moderate class imbalance (churn ≈ 26.5%)
- No literal missing values, but `TotalCharges = " "` acts as one

**Split protocol:** stratified 80/20 (`random_state=42`) via `sklearn.model_selection.train_test_split`. This split is deterministic per dataset — every upload of the same CSV produces the same train/test partition, so meta-features are stable.

### 5.2 LLM backend

**Primary:** `gpt-oss:120b-cloud` (Ollama cloud-hosted, 116.8B parameters, MXFP4 quantization).

**Also available on the machine but not used in the primary results:**
- `ministral-3:14b-cloud` (Mistral 3, 14B, FP8) — smaller cloud model
- `qwen2.5:3b` (local) — 1.9 GB on disk, CPU-only inference

Ollama is accessed via HTTP at `http://host.docker.internal:11434/api/generate`, `stream: false`, `temperature: 0.7`, `seed: <experiment seed>`.

**Important caveat: Ollama cloud does not fully respect the `seed` parameter.** Empirically, B0 runs on Telco produced *identical* scores (0.7051) across four of five seeds, suggesting Ollama's hosted-model inference is not honoring seed values as documented. Local models (`qwen2.5:3b`) appear to respect seeds, but we have not systematically verified. **This is a threat to reproducibility of the exact scores at a given seed on cloud backends** — it does not compromise the aggregate findings, but any per-seed comparison across time should be interpreted with this in mind.

### 5.3 Seeds and iteration budget

- **Seeds used across experiments:** 42, 43, 44, 45, 46, 100, 101, 110–114, 120–124, 200, 201, 202
- **Iteration budget per cell:** `max_iter = 3` unless otherwise noted
- **Subprocess timeout per iteration:** `timeout_seconds = 180`

### 5.4 Sweep structure

A "sweep" is the cartesian product of `dataset_ids × conditions × llm_backends × seeds`. Sweeps are enqueued as a single job that iterates cells serially; each cell is one call to `run_cell()`. Sweep progress is tracked in `sweep_jobs` and can be polled at `GET /sweeps/{id}`.

Total runs conducted in the reported experiments: **~40 across 2 datasets × 3 conditions × ~7 seeds** (with some retries after fixing bugs mid-experiment).

### 5.5 Third-party libraries available to generated code

The worker container image (`Dockerfile.worker`) pre-installs everything the B2 decision rules mention, plus common tabular ML tools:

- **scikit-learn** (all preprocessors, ensembles, linear models)
- **category_encoders** (`TargetEncoder`) — rule 3
- **lightgbm** (`LGBMClassifier`) — rule 4
- **imbalanced-learn** (`SMOTE`, `TomekLinks`) — rules 10, 11
- **xgboost-cpu** — noted in rule 13; CPU-only variant to avoid a ~1.5 GB CUDA dependency
- **pandas, numpy, scipy** — baseline stack

Note: we discovered mid-experiment that omitting `category_encoders` from the worker image caused 100% of B2 runs on Titanic to fail with `import_error` (rule 3 → `from category_encoders import TargetEncoder`). Adding it was a **methodological correction**, not a change in the LLM's behavior. All reported results use the corrected worker image.

---

## 6. Results

### 6.1 Cross-dataset summary (per condition)

Aggregated across all runs on Titanic + Telco, cloud model `gpt-oss:120b-cloud`, seeds spanning multiple sweeps. Data snapshot as of 2026-07-11 evening; the numbers below are computed from the `/analysis/summary` endpoint at that time.

| Condition | N runs | Success rate | Mean score | Median score | Failure categories |
|---|---|---|---|---|---|
| **B0 naive** | 10 | 100% | 0.7434 | 0.7438 | (none) |
| **B1 schema** | 10 | 100% | 0.7883 | 0.7697 | (none; includes 1 leaked score = 1.000) |
| **B1 corrected** (leak excluded) | 9 | 100% | 0.7690 | 0.7526 | (none) |
| **B2 meta-feature** | 11 | 82% | 0.6294 | 0.7268 | 1× import_error (pre-fix), 1× runtime_other |
| **B2 (session-2 sweep #4, verification enforced)** | 10 | 70% | (see §6.4) | — | 1× missing_name, 1× syntax_error, 1× api_hallucination |

**Ordering** on cross-dataset means: **B1 > B0 >> B2**. B1's marginal edge over B0 (~2.5% after excluding the leaked 1.000 run) is small but consistent across seeds; B2's underperformance is not marginal — it is nearly 15 percentage points below B1.

### 6.2 Per-dataset breakdown

**Titanic** (n=712 train / 179 test after split):

| seed | B0 | B1 | B2 | B2 error category (if failed) |
|---|---|---|---|---|
| 42 | 0.758 | 0.758 | 0.500 | success but trap |
| 43 | 0.781 | 0.792 | 0.500 | success but trap |
| 44 | 0.758 | 0.781 | 0.764 | ok |
| 45 | 0.805 | 0.808 | — | runtime_other (rule-interaction crash) |
| 46 | 0.781 | 0.808 | 0.752 | ok |
| **B2 seed=100 (session-2)** | — | — | 0.752 | ok, no trace saved (pre-fix) |
| **B2 seed=101 (verification enabled)** | — | — | 0.500 | success but trap; 6/6 decisions verified faithful |
| **B2 seed=110** | — | — | 0.776 | ok, 6/6 faithful |
| **B2 seed=111** | — | — | 0.760 | ok, 4/4 faithful |
| **B2 seed=112** | — | — | — | schema-parse false-positive (fixed in-session) |
| **B2 seed=113** | — | — | — | schema-parse false-positive (fixed in-session) |
| **B2 seed=114** | — | — | — | missing_name after all 3 iterations |
| **B2 seed=120** | — | — | 0.500 | success but trap, 4/4 faithful |
| **B2 seed=121** | — | — | 0.772 | ok, 5/5 faithful |
| **B2 seed=122** | — | — | 0.752 | ok, 4/4 faithful |
| **B2 seed=123** | — | — | 0.769 | ok, 4/4 faithful |
| **B2 seed=124** | — | — | 0.500 | success but trap, 5/5 faithful |
| **B2 seed=202** | — | — | 0.500 | success but trap, 6/6 faithful, persisted in DB |

**Telco** (n=5634 train / 1409 test after split):

| seed | B0 | B1 | B2 | B2 note |
|---|---|---|---|---|
| 42 | 0.705 | 0.755 | 0.753 | ok |
| 43 | 0.705 | 0.753 | 0.716 | ok |
| 44 | 0.730 | **1.000** (leaked) | 0.727 | B1 forgot to drop `Churn` from features |
| 45 | 0.705 | 0.716 | 0.735 | ok |
| 46 | 0.705 | 0.712 | 0.714 | ok |
| **B2 seed=120 (session-2)** | — | — | 0.620 | ok, 4/4 faithful |
| **B2 seed=121** | — | — | 0.753 | ok, 3/3 faithful |
| **B2 seed=122** | — | — | 0.704 | ok, 2/2 faithful |
| **B2 seed=123** | — | — | — | syntax_error |
| **B2 seed=124** | — | — | — | api_hallucination |

Observations:
- On Telco, **B2 never falls into the majority-baseline trap** (all successful B2 runs score 0.61–0.75). Why? Because the LLM applied heuristic column semantics — `customerID` *looks* like an ID (name it obviously so), so the LLM correctly drops it; Titanic's `Name` does not look like an ID (it is a person's name), so the LLM target-encodes it and gets trapped.
- On Telco B0, four of five seeds produce identical scores (0.7051). Evidence that Ollama cloud is not fully honoring `seed`.
- The **B1 seed=44 Telco score of 1.000** is a *silent leakage* case: the LLM dropped `customerID` correctly and converted `TotalCharges` correctly but **forgot to drop the target column `Churn`** from `X_train`. Downstream OneHotEncoder made `Churn` a feature; the model memorized it. We added the `suspicious_leakage` guardrail (§4.3, threshold 0.995) after observing this; subsequent runs correctly flag it.

### 6.3 Wilcoxon paired tests

With `n_datasets = 2`, the paired Wilcoxon signed-rank test has minimal statistical power. Values reported for completeness:

- B0 vs B2: `statistic = 1.0, p_value = 1.0, n_datasets = 2, condition_a_mean = 0.7434, condition_b_mean = 0.6294`
- B1 vs B2: `statistic = 0.0, p_value = 0.5, n_datasets = 2, condition_a_mean = 0.7883, condition_b_mean = 0.6294`

**A third dataset (Ames Housing regression, added to the reusables catalog as OpenML id 41211) would raise `n_datasets` to 3 and give the Wilcoxon test enough power to be interpretable.** This is planned but not yet run.

### 6.4 Faithfulness metrics (B2 sweep #4, post-fix)

The definitive faithfulness dataset — 10 B2 runs on seeds 120–124 across Titanic + Telco, after fixing the schema issue that had caused false-positive unfaithful flags in the prior sweep.

- **Runs total:** 10
- **Success rate (execution):** 7 / 10 = 70%
- **Runs producing a machine-parseable trace:** 7 / 10 (the 3 that did not produced no trace because their execution failed before the SCORE/REASONING lines)
- **Faithful runs (100% decisions verified):** 7 / 7 = 100%
- **Individual decisions verified:** **28 / 28 = 100%**

**Rule usage (of 7 traces with decisions):**

| Rule | Cited N times | Verified N times | Verification rate |
|---|---|---|---|
| 3 (TargetEncoder for cardinality > 20) | 7 | 7 | 100% |
| 1 (IterativeImputer for missing > 0.05) | 6 | 6 | 100% |
| 7 (RobustScaler for outliers > 0.05) | 6 | 6 | 100% |
| 14 (linear models when NB > 1-NN + 0.1) | 4 | 4 | 100% |
| 4 (LGBM for n_cat > n_num) | 2 | 2 | 100% |
| 6 (PowerTransformer for skew > 2.0) | 2 | 2 | 100% |
| 2 (drop cols > 50% missing) | 1 | 1 | 100% |

**Observation:** rule 3 is cited universally (every trace mentions it — every dataset in our set has at least one column with cardinality > 20). Rule 4 is cited less often (2/7) because when landmarking says linear models beat 1-NN by > 0.1 (rule 14), the LLM picks LogisticRegression instead of LGBMClassifier. **Rule 4 fires precisely when the LLM is going to fall into the trap** — the two trap-inducing runs (Titanic seeds 120 and 124) are the two B2 traces that cite rule 4.

### 6.5 Rule-interaction table (the paper's central artifact)

For every B2 run on Titanic that produced a trace, we list which rules were cited and the resulting score:

| seed | Rules cited | Test score | Outcome |
|---|---|---|---|
| 101 | 1, 3, 4, 6, 7 | 0.500 | trap |
| 110 | 1, 3, 4, 6, 7, ? | 0.776 | (no trap despite rule 4 — LLM chose different combo) |
| 111 | 1, 2, 3, 7, 14 | 0.772 | ok (rule 14 = linear model instead of LGBM) |
| 122 | 1, 3, 6, 7 | 0.752 | ok |
| 123 | 1, 3, 7, 14 | 0.769 | ok |
| 120 | 1, 3, 4, 7 | 0.500 | trap |
| 124 | 1, 3, 4, 6, 7 | 0.500 | trap |
| 202 | 1, 2, 3, 4, 6, 7 | 0.500 | trap, persisted with full trace |

**Pattern:** in this subset, all four trap runs cite rule 4; three of five ok runs do not cite rule 4. The one seed=110 outlier suggests rule 4 is *necessary but not sufficient* — the LLM must also apply TargetEncoder to a near-unique column, which depends on how it interprets the categorical listing. We need more data to characterize this more precisely, but the rule-4 + trap association is clear.

---

## 7. Findings in Detail

### 7.1 Finding 1 — Rule 3 + Rule 4 co-firing produces a deterministic majority-class collapse on Titanic

**Statement.** When B2's rule 3 (TargetEncoder for categorical columns with cardinality > 20) is applied to Titanic's `Name` column (712 unique values in 712-row training set) AND rule 4 (LGBMClassifier for datasets with more categorical than numeric features) is applied together, the resulting pipeline achieves 0.500 balanced accuracy on the held-out test set — identical to always predicting the majority class.

**Mechanism.**
1. `TargetEncoder.fit_transform(X_train[Name], y_train)` maps each unique training Name to its (smoothed) mean target value. With 712 samples per category (well, 1 per category), the smoothed encoding is essentially the y value itself.
2. `LGBMClassifier.fit(X_train, y_train)` — LightGBM greedily splits on the near-perfect `Name_encoded` column. Training loss goes to zero after very few splits (the [LightGBM] warnings *"No further splits with positive gain"* fire repeatedly).
3. On test data: nearly every test Name is unseen. `TargetEncoder.transform(X_test[Name])` returns the global prior for each — a constant ≈ 0.383 (Titanic survival rate). LightGBM sees a constant → falls through to its lowest-frequency leaf → predicts the majority class for every test row.
4. `y_pred = [0, 0, 0, ..., 0]` for all 179 test rows. `balanced_accuracy_score = (TN_rate + TP_rate) / 2 = (1.0 + 0.0) / 2 = 0.500`.

**Observed frequency.** In the definitive sweep, all four Titanic trap runs cited rule 4; three of five ok runs did not. In cross-session data, seed 42, 43, 101, 120, 124, 202 all showed this pattern (6 confirmed traps).

**Falsifiability.** Replace rule 4 with any linear model (rule 14 gives LogisticRegression) and the trap disappears. L2 regularization keeps rare-Name coefficients small; unseen Names contribute ≈ 0 to the logit; the model gracefully falls back to Sex, Pclass, Fare, Age.

**Publishable framing.** *"On tabular classification datasets containing near-unique identifier columns, the naïve co-firing of high-cardinality target encoding with tree-based ensembles produces silent majority-class collapse. The LLM applying these rules is not at fault — the rules themselves encode an adversarial interaction."*

### 7.2 Finding 2 — Faithfulness and correctness are orthogonal

**Statement.** In every one of the six documented trap runs, mechanical verification confirmed that the LLM's reasoning trace was **fully faithful**: every cited `observed_value` matched the ground-truth meta-features within 1e-3 tolerance, and every cited `action` symbol appeared in the generated code's AST. Score: 0.500. The LLM was honest, and honestly wrong.

**Aggregate.** Across the 10-run sweep #4, **28 out of 28** decisions verified. **Zero fabricated numbers. Zero cited actions absent from code.** When the LLM produces a structured trace, it produces a truthful one.

**Interpretation.** Existing faithfulness work in NLP (chain-of-thought interpretability studies) has consistently found that LLM's post-hoc rationalizations often diverge from their internal decision process. Our setting is different: we ask the LLM to emit its reasoning *before* (or concurrently with) the code it produces, and the reasoning references machine-checkable facts. Under these conditions, the LLM does not lie. But truthful reasoning is not the same as correct reasoning — the LLM can faithfully execute a bad rule set.

**Implication for the paper.** The claim *"our system verifies LLM reasoning"* is not the same as *"our system verifies that the pipeline is good."* We measure these two axes separately and expose both as `faithful: bool` and `test_score: float`. A production AutoML system built on LLMs needs both — faithfulness alone will not save users from Rule 3+4-style traps.

### 7.3 Finding 3 — Silent score-leakage inflation exists and is catchable

**Statement.** A B1 Telco run at seed=44 scored 1.000 balanced accuracy. Inspection of the generated code showed the LLM had (correctly) dropped `customerID` and converted `TotalCharges`, but had **failed to drop the target column `Churn` from features**. Downstream OneHotEncoder made `Churn` a categorical feature; the model learned "predict yes if `Churn_yes = 1`" and got perfect predictions.

**Impact.** Untreated, this run would pull the B1 mean score up by ~2 percentage points, inflating the apparent advantage of B1 over B2. It looks like a genuine success ("100% accuracy!") to anyone not reading the generated code.

**Guardrail.** In `execute_pipeline`, after extracting `score`, we now check: if the task is classification and `score >= 0.995`, we flip the run to `success = false, error_category = suspicious_leakage`. The score is preserved (so we know what was inflated) but the run is excluded from correctness statistics.

**Coverage.** The 0.995 threshold catches the extreme case demonstrated. It does not catch subtler leakage (e.g. train/test on the same rows → 0.85 instead of 0.80). Future work: cross-validated leakage detection, feature-target correlation sanity checks.

### 7.4 Finding 4 — Import discipline as a distinct failure mode

**Statement.** LLM code generation occasionally references classes that were never imported (e.g. `scaler = RobustScaler()` with no matching import statement). This produces `NameError: name 'RobustScaler' is not defined` at runtime.

**Before intervention.** Session 1's initial B2 runs on Titanic showed a specific instance: the LLM produced `RobustScaler()` without importing it. The subprocess exited with `NameError`. Before we added the `MISSING_NAME` error category, this fell under `runtime_other`, obscuring the true failure mode.

**Interventions applied.**
1. Added an explicit clause to all three prompt conditions: *"The script must include ALL import statements at the top."*
2. Added `MISSING_NAME` as a distinct error category; the `error_taxonomy.py` regex identifies `NameError: .+ is not defined` and maps it to `missing_name` rather than the catch-all.
3. Enhanced `build_error_feedback()` to include import-focused guidance when `error_category = missing_name`: *"This error is caused by using a name that was never imported. Check that EVERY class, function, and constant you reference has a corresponding import statement at the top."*

**Result.** Post-intervention, `missing_name` errors still occur (e.g. B2 seed=114 on Titanic in sweep #4) but at lower rates. This finding contributes to RQ2 (does B2 reduce failure rates?) — the answer is condition-dependent, and the specific class of failure matters.

### 7.5 Finding 5 — Ollama cloud non-determinism

**Statement.** Setting `options.seed = <fixed value>` in Ollama's `/api/generate` request does not reliably produce identical outputs across calls for cloud-hosted models (`gpt-oss:120b-cloud`, `ministral-3:14b-cloud`).

**Evidence.** On Telco, B0 runs at seeds 42, 43, 45, 46 all produced identical balanced accuracy 0.7051. Seed 44 produced 0.7296. If the seed were being honored, we would expect all five to differ. Contrastingly, B2 runs at different seeds produced clearly different code paths (different rules cited, different scores).

**Explanation (speculative).** Ollama's cloud API may hash the seed differently across cache layers, or the underlying provider (Anthropic-like inference infrastructure) may not accept a seed parameter at all. Investigating this is beyond the scope of the current work.

**Implication.** For statistical rigor, we cannot treat seed as a reproducibility knob for cloud backends. We must instead report aggregate statistics across many seeds and characterize variance empirically. Local models (`qwen2.5:3b`) appear to be more deterministic but have not been rigorously tested.

---

## 8. Discussion

### 8.1 Is meta-feature guidance harmful?

Reading only the mean scores, one could conclude: *B0 (naive) is essentially as good as B2 (guided); B1 (schema-only) is marginally better than both; therefore, meta-feature guidance is at best useless and at worst harmful.*

This is too simple. A more careful reading:

- **On datasets without unique-ID columns**, B2 tracks B0 closely (Telco results). It does not obviously hurt.
- **On datasets with unique-ID columns**, B2 has bimodal outcomes — either trap (0.500) or ok (0.75+), depending on whether the LLM's specific code sample fires rule 4 or rule 14.
- **The rules are not the LLM's fault.** They are prescribed by the researcher (us). The LLM applied them faithfully. A well-designed rule set should not contain adversarial co-firings.

**Reframing.** Meta-feature guidance is not intrinsically better or worse than naive prompting. It shifts *what class of failures the LLM produces*. Naive prompting produces occasional syntactic errors and defaults to safe (LogReg + OneHotEncoder + StandardScaler) pipelines. Guided prompting produces trace-verifiable, rule-compliant pipelines that occasionally instantiate a rule interaction the researcher did not foresee. Which is better depends on the failure taxonomy the researcher can tolerate.

### 8.2 The rule-lint idea

A concrete engineering artifact from this work: **a static analyzer for AutoML rule sets** that flags known-bad co-firings.

Given the observed 3+4 trap, we can propose a *rule lint*:

> *Warning: Rule 3 (TargetEncoder for high-cardinality) combined with a tree-based model choice (rules 4, 12, 13) is unsafe when applied to columns with cardinality approaching n_rows. Add a cardinality-to-n_rows ratio check to rule 3 (e.g., cardinality > 0.5 · n_rows → drop the column, do not target-encode).*

Or, in the rule set itself, a **rule 3b** that fires before rule 3 and drops near-unique columns entirely. This would break B2 out of the trap.

**Trade-off.** Fixing the rule set changes what B2 *is*. Publish once with the trap documented; then propose the fix as future work.

### 8.3 Reliance on faithfulness as a review artifact

The trace + verification report is intended as a *reviewability artifact* — a downloadable, machine-verifiable record of every choice the LLM made. Applications:

- **In research.** Peer reviewers can audit the LLM's reasoning without re-running the pipeline.
- **In production ML.** A DS lead can see which rules a proposed pipeline claims to apply and confirm they match the actual code before deployment.
- **In education.** Students learning ML can see the LLM's cited rule → meta-feature → action mapping as a scaffolded explanation.

The main limitation: verification requires the LLM to produce a *structured* trace. Free-form narrative reasoning (as in typical ChatGPT output) is much harder to verify mechanically. Our contribution here is precisely that: forcing a structured schema in the prompt turns unverifiable narrative into checkable claims.

### 8.4 Comparison to existing AutoML systems

Not a fair comparison at this scale, but for context:

- **auto-sklearn** — 0.85–0.88 on Titanic with default settings, hundreds of pipeline evaluations. Our best B2 score: 0.776. Our worst: 0.500.
- **H2O AutoML** — similar range, comparable evaluation budget.
- Our system is not trying to compete with dedicated AutoML tools on final accuracy. It is measuring the **effect of prompt structure on LLM-driven pipeline generation**, holding LLM backend and evaluation budget constant.

---

## 9. Limitations & Threats to Validity

### 9.1 Statistical

- **N = 2 datasets.** Wilcoxon signed-rank test has effectively zero power. Any claim of significant difference between conditions is premature. Adding Ames Housing (regression) as a third dataset is the immediate next step.
- **~40 total runs** — small even at the seed level. Effect sizes reported are point estimates, not confidence intervals.
- **Ollama cloud non-determinism** compromises per-seed comparisons.

### 9.2 Systemic

- **Single LLM backend for primary results.** All main-result runs use `gpt-oss:120b-cloud`. Smaller / different-family models may behave differently. RQ4 (backend size) is entirely open.
- **Single prompting-condition family.** B0/B1/B2 span "no info", "schema only", "meta-features + rules". Intermediate conditions (e.g. "meta-features but no rules", "rules but no meta-features") are not tested.
- **Sweep count matters more than seed count.** Because Ollama cloud may ignore seeds, our "5 seeds" may functionally be "3–4 seeds" of unique code output.

### 9.3 Domain / dataset selection

- Both datasets are binary classification. Multi-class results untested (though the pipeline supports them).
- Regression path implemented but not exercised (Ames Housing pending).
- Both datasets have unique-ID columns; datasets without such columns might show a very different B2 profile.

### 9.4 Verification coverage

- We verify `observed_value` (does the LLM's cited number match the meta-feature?) and `action` presence (does the class it named appear in the code?). We do NOT verify:
  - Whether the action was *applied to the right columns* (e.g. RobustScaler on Fare, not on all numerics).
  - Whether the *rule condition* (threshold comparison) was actually satisfied. The LLM could claim rule 7 fires on `outlier_ratio = 0.001` — we would flag the value mismatch, but not the condition mismatch if it correctly cited 0.128 but then chose to apply the action anyway.
  - Ablation: does removing the action change the score meaningfully? (Future work.)

---

## 10. Future Work

Prioritized by impact:

1. **Add Ames Housing as a third dataset.** Immediate. Enables Wilcoxon significance and tests whether the trap generalizes to regression.
2. **Ablation module.** For every claimed-important decision in a faithful trace, run a counterfactual pipeline with that decision removed; measure score delta. This produces per-decision *causal* evidence. Expensive (N× runs per cell) but high-value.
3. **Save trained model artifacts.** The original vision included delivering (a) cleaned data, (b) trained model, (c) documentation, (d) code. Currently only (d) is persisted; the pipeline's `sklearn.Pipeline` object is discarded when the subprocess exits. Adding `joblib.dump(pipeline, "model.joblib")` inside the generated code and persisting it as an artifact would complete (b) and (c).
4. **User-in-the-loop refinement.** `POST /runs/{id}/refine {prompt}` — spin a new run using the parent's code + trace as context, plus a user natural-language modification. Original vision item.
5. **Multi-backend RQ4 experiments.** Repeat sweeps with `ministral-3:14b-cloud` and a local model (e.g. `qwen2.5:3b`). Report backend × condition interaction.
6. **B1 trace generation as a control.** Currently only B2 must emit a trace. If B1 also produced one (without the rule scaffolding), we could compare faithfulness across conditions.
7. **Frontend.** React/Next.js dashboard exposing dataset browser, meta-feature explorer, run history, side-by-side condition comparison, and downloadable artifacts (original CSV, trace JSON, verification report).
8. **CI on GitHub Actions.** Pytest across all 6 code repos on push.

---

## 11. Conclusion

This work develops a microservice-based experimental platform for studying meta-feature-guided prompting in LLM-driven tabular AutoML, and introduces a **mechanical verification layer** that separates *faithfulness* (does the LLM's stated reasoning match the code?) from *correctness* (does the pipeline generalize?).

Preliminary empirical results across two datasets show:

1. **Naive and schema-only prompting outperform meta-feature-guided prompting** on the cross-dataset mean, primarily because the guided prompt's decision rules encode an adversarial co-firing (rule 3 + rule 4) that catastrophically collapses on datasets with near-unique identifier columns.
2. **The LLM is completely honest when compelled to produce structured reasoning.** 28 of 28 decisions verified across 10 B2 runs. Faithfulness rate: 100%.
3. **Faithfulness ≠ correctness.** Fully faithful B2 traces coexist with 0.500 balanced-accuracy scores. The LLM does exactly what it claims to do; the claim itself is flawed.

The main contribution is neither a new AutoML system nor a new prompting technique, but a **framework for mechanically auditing LLM-generated ML code against its own stated reasoning**. This framework surfaces failure modes (like the rule 3+4 trap) that free-form narrative reasoning would obscure.

---

## Appendix A — B2 Prompt (verbatim)

The B2 prompt, as constructed by `automl-reusables/src/conditions/b2_metafeature.py`, has this structure:

```
You are an AutoML assistant. Generate a complete scikit-learn pipeline for the following task:

{task_description — dataset path, target column, task type, metric requirements}

Return a **single, self-contained** Python script that loads the data, preprocesses it, trains a
model, and evaluates on a held-out test set. Use balanced_accuracy_score for classification or
root_mean_squared_error for regression.

IMPORTANT: The script must include ALL import statements at the top. Every class and function you
use (e.g. RobustScaler, ColumnTransformer, train_test_split, etc.) must be explicitly imported.
The script will run in an isolated environment with no pre-imported names.

## Dataset Schema
{rows | cols | task_type | target_col | column dtypes | first 3 rows as Markdown table}

## Dataset Meta-Feature Profile
```json
{full MetaFeatures Pydantic model dumped as indented JSON — includes all 4 groups}
```

## Decision Rules (apply where conditions match)
{the 14 rules — see Appendix B}

IMPORTANT: For each pipeline component you choose, cite which meta-feature or rule informed
your decision.

## Structured Reasoning (mandatory — machine-verified)

After the `SCORE: <number>` line, your script MUST also print exactly one line of the form:

  REASONING: <json>

where <json> is a single-line JSON object with this shape:

```json
{
  "decisions": [
    {
      "step": "scaling",
      "rule_id": 7,
      "meta_feature": "distributional.outlier_ratio_per_numeric.Fare",
      "observed_value": 0.128,
      "threshold": 0.05,
      "action": "RobustScaler",
      "applied_to": ["Fare"]
    }
  ]
}
```

Rules for producing the trace:
- One decision per meaningful choice you made (imputation, scaling, encoding, model,
  feature-selection, imbalance handling).
- `meta_feature` is a dotted path into the meta-feature JSON above.
- `observed_value` must be the exact number you read from that path (copy verbatim, no rounding).
- `action` must be the class or function name that actually appears in your imports.
- The trace is verified mechanically against your generated code and the meta-feature values.
  Fabricating numbers or citing symbols you did not actually import will cause the run to be
  flagged as unfaithful and discarded.
```

## Appendix B — The 14 B2 Decision Rules

Verbatim from `automl-reusables/src/conditions/b2_metafeature.py`:

```
PREPROCESSING RULES:
1. If missing_ratio_overall > 0.05: prefer IterativeImputer over SimpleImputer(strategy='mean').
2. If missing_ratio_overall > 0.30: consider dropping columns with >50% missing before imputation.
3. If any categorical_cardinality > 20: use TargetEncoder instead of OneHotEncoder for those columns.
4. If n_categorical > n_numeric: consider gradient boosting with native categorical support
   (e.g., LGBMClassifier(categorical_features=...)).

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
12. If decision_stump_score > 0.75: the problem may be simple — try LogisticRegression or shallow
    DecisionTreeClassifier first.
13. If one_nn_score - naive_bayes_score > 0.1: data has local structure — prefer tree ensembles
    (RandomForest, XGBoost) or KNN.
14. If naive_bayes_score - one_nn_score > 0.1: features are relatively independent — linear models
    or GaussianNB are viable.
```

## Appendix C — Complete Run Tables

Live queries — always fetch the current definitive numbers from the analysis service:

```
GET http://localhost:8000/analysis/summary        # per-condition means
GET http://localhost:8000/analysis/errors         # per-condition failure counts
GET http://localhost:8000/analysis/iterations     # mean iterations to first success
GET http://localhost:8000/analysis/traces         # every B2 run with its trace + report
GET http://localhost:8000/analysis/rule-usage     # rule citation/verification counts
GET http://localhost:8000/analysis/wilcoxon?a=B0&b=B2
```

Direct Postgres query:

```sql
SELECT dataset_id, condition, llm_backend, seed, success, test_score, error_category, iterations_used
FROM run_results
ORDER BY dataset_id, condition, seed;
```

## Appendix D — Example Reasoning Trace + Verification Report

**Titanic seed=202 (run_id=57), B2, gpt-oss:120b-cloud** — score 0.500 (trap), fully faithful.

Trace (`train_B2_seed202_iter*.trace.json`):

```json
{
  "decisions": [
    {
      "step": "drop_columns",
      "rule_id": 2,
      "meta_feature": "simple.missing_ratio_per_column.Cabin",
      "observed_value": 0.7752808988764045,
      "threshold": 0.5,
      "action": "drop",
      "applied_to": ["Cabin"]
    },
    {
      "step": "imputation",
      "rule_id": 1,
      "meta_feature": "simple.missing_ratio_overall",
      "observed_value": 0.08822778345250255,
      "threshold": 0.05,
      "action": "IterativeImputer",
      "applied_to": ["Age", "Fare"]
    },
    {
      "step": "encoding_high_cardinality",
      "rule_id": 3,
      "meta_feature": "simple.categorical_cardinality.Name",
      "observed_value": 712,
      "threshold": 20,
      "action": "TargetEncoder",
      "applied_to": ["Name", "Ticket", "Cabin"]
    },
    {
      "step": "scaling",
      "rule_id": 7,
      "meta_feature": "distributional.outlier_ratio_per_numeric.Fare",
      "observed_value": 0.12780898876404495,
      "threshold": 0.05,
      "action": "RobustScaler",
      "applied_to": ["Fare"]
    },
    {
      "step": "power_transform",
      "rule_id": 6,
      "meta_feature": "distributional.skewness_per_numeric.Fare",
      "observed_value": 4.636395895787561,
      "threshold": 2.0,
      "action": "PowerTransformer",
      "applied_to": ["Fare"]
    },
    {
      "step": "model_selection",
      "rule_id": 4,
      "meta_feature": "simple.n_categorical",
      "observed_value": 9,
      "threshold": "n_categorical > n_numeric",
      "action": "LGBMClassifier",
      "applied_to": ["all_features"]
    }
  ]
}
```

Verification report (`.verification.json`):

```json
{
  "n_decisions": 6,
  "n_faithful": 6,
  "faithful": true,
  "verdicts": [
    {"step": "drop_columns", "action": "drop", "value_matches": true, "action_present": true, "reasons": []},
    {"step": "imputation", "action": "IterativeImputer", "value_matches": true, "action_present": true, "reasons": []},
    {"step": "encoding_high_cardinality", "action": "TargetEncoder", "value_matches": true, "action_present": true, "reasons": []},
    {"step": "scaling", "action": "RobustScaler", "value_matches": true, "action_present": true, "reasons": []},
    {"step": "power_transform", "action": "PowerTransformer", "value_matches": true, "action_present": true, "reasons": []},
    {"step": "model_selection", "action": "LGBMClassifier", "value_matches": true, "action_present": true, "reasons": []}
  ],
  "overall_notes": null
}
```

**Test score for this run: 0.500.** The LLM was 100% honest, and its faithful choices led to majority-baseline predictions.

## Appendix E — Error Taxonomy

`ErrorCategory` in `automl-reusables/src/contracts.py`. Each error a generated pipeline can produce, in classification-priority order (first match wins in `classify_error`):

| Category | Trigger | Example |
|---|---|---|
| `timeout` | Subprocess exceeded `timeout_seconds` | Long-running LightGBM on 150-MB dataset |
| `syntax_error` | Generated Python fails to parse | Unmatched paren, invalid indent |
| `import_error` | `ModuleNotFoundError` / `ImportError` | `from category_encoders import ...` when lib not installed |
| `missing_name` | `NameError: '.+' is not defined` | LLM uses `RobustScaler()` without importing it |
| `api_hallucination` | `has no attribute` / `is not callable` / `unexpected keyword` | Called a sklearn method that doesn't exist |
| `shape_mismatch` | `Found input variables with inconsistent samples`, dimension errors | Fit on X_train but transform on wrongly-shaped X_test |
| `type_error` | TypeError involving categorical/string/float conversion | Numeric column stored as string with spaces |
| `deprecated_api` | Deprecation/Future warnings | Using sklearn 0.22-era `normalize=True` |
| `resource_limit` | MemoryError, ResourceWarning | Full-cross-product feature engineering |
| `suspicious_leakage` | Classification score ≥ 0.995 | Target column not dropped from X |
| `reasoning_unfaithful` | B2 trace missing/malformed, or any decision fails verification | LLM omits REASONING line or emits fabricated value |
| `runtime_other` | Anything else runtime | Catch-all bucket |
| `infrastructure` | Ollama unreachable, DB gone | Not set by taxonomy — set by worker |

## Appendix F — Repository Layout

```
D:\Major Project\
├── major-project-AutoML\           github.com/Samarth-Ad/major-project-AutoML
│   ├── SESSION_CONTEXT.md          — full onboarding doc for new Claude sessions
│   ├── REPORT.md                   — this file
│   ├── README.md                   — quickstart for the CLI
│   ├── scripts\
│   │   ├── dry_run.py              — single-cell diagnostic CLI
│   │   └── run_sweep.py            — full experiment sweep CLI
│   ├── tests\                      — 49+ tests covering the reusables surface
│   └── pyproject.toml              — depends on automl-reusables via git URL
│
├── automl-reusables\               github.com/Major-Proj-AutoML/automl-reusables
│   ├── src\contracts.py            — MetaFeatures, RunResult, ReasoningTrace, VerificationReport
│   ├── src\conditions\             — b0_naive, b1_schema, b2_metafeature
│   ├── src\meta_features\          — simple, distributional, information, landmarking
│   ├── src\execution\
│   │   ├── runner.py               — execute_pipeline (subprocess isolation)
│   │   ├── metrics.py              — extract_score, extract_reasoning
│   │   ├── verification.py         — verify_reasoning (AST + dotted-path)
│   │   └── error_taxonomy.py       — classify_error
│   └── src\experiments\
│       ├── datasets.py             — load_dataset (OpenML), load_custom_dataset (uploaded)
│       ├── runner.py               — call_llm (Ollama HTTP), build_error_feedback
│       └── analysis.py             — summary_table, wilcoxon_test, iteration_efficiency, etc.
│
├── automl-infra\                   github.com/Major-Proj-AutoML/automl-infra
│   ├── docker-compose.yml          — Postgres + Redis (base)
│   ├── docker-compose.full.yml     — + all 5 services + worker
│   ├── db\init\01_schema.sql       — Postgres schema (auto-runs on first start)
│   └── .env.example                — port/credential overrides
│
├── automl-data-service\            :8001  (dataset registry)
├── automl-metafeatures-service\    :8002  (meta-feature extraction)
├── automl-generation-service\      :8003 + generation-worker (async LLM generation)
├── automl-analysis-service\        :8004  (RQ1-RQ5 stats + faithfulness)
└── automl-gateway\                 :8000  (frontend entry, CORS, composed workflows)
```

## Appendix G — Reproducibility

### Prereqs on any Linux/macOS/Windows machine

- Docker Desktop (or Docker Engine + Compose plugin)
- Ollama (`ollama serve`) with a cloud-authenticated model pulled (`ollama pull gpt-oss:120b-cloud`)
- Git

### Clone and start

```bash
mkdir automl && cd automl
git clone https://github.com/Samarth-Ad/major-project-AutoML.git
for r in reusables infra data-service metafeatures-service generation-service analysis-service gateway; do
    git clone "https://github.com/Major-Proj-AutoML/automl-$r.git"
done
cd automl-infra
docker compose -f docker-compose.yml -f docker-compose.full.yml up -d --build
curl http://localhost:8000/health
```

First build takes 5–10 min. Once up, browse http://localhost:8000/docs for the full API.

### Reproduce a Titanic B2 run

```bash
# 1. Download Titanic train.csv from Kaggle (kaggle.com/c/titanic/data)
# 2. Upload + extract meta-features
curl -X POST http://localhost:8000/workflows/upload-and-extract \
  -F "file=@train.csv" -F "target_col=Survived" -F "task_type=binary_classification"

# 3. Enqueue a B2 run (using dataset_id from step 2 response)
curl -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d '{
  "dataset_id": 1, "condition": "b2_metafeature",
  "llm_backend": "gpt-oss:120b-cloud", "seed": 42,
  "max_iter": 3, "timeout_seconds": 180
}'

# 4. Poll for result
curl http://localhost:8000/runs?dataset_id=1&limit=1

# 5. Inspect the trace + verification report
curl http://localhost:8000/analysis/traces?dataset_id=1&limit=1
```

### Test suite

Each service has SQLite in-memory + mocked upstream, so no infra is needed:

```bash
cd major-project-AutoML && .venv/Scripts/python.exe -m pytest tests/   # 49 tests
cd automl-data-service && .venv/Scripts/python.exe -m pytest tests/   # 6 tests
cd automl-metafeatures-service && .venv/Scripts/python.exe -m pytest tests/  # 8 tests
cd automl-generation-service && .venv/Scripts/python.exe -m pytest tests/    # 10 tests
cd automl-analysis-service && .venv/Scripts/python.exe -m pytest tests/      # 9 tests
cd automl-gateway && .venv/Scripts/python.exe -m pytest tests/               # 6 tests
```

Total: **88 tests**, all green as of 2026-07-11.

---

*End of report. This document is the substantive record; SESSION_CONTEXT.md is its concise index.*
