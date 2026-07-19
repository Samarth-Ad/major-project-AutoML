# Meta-Feature-Guided Prompting for LLM-Driven Tabular AutoML: A System Study on Faithfulness and Correctness

**Status:** In-progress research report (paper draft basis)
**Working dates:** 2026-07-09 to 2026-07-19 (Stage 1–2: 07-14 → 07-15; Stage 3: 07-15 → 07-16; Stage 4a: 07-19)
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
19. [Appendix H — Stage-1 and Stage-2 Operational Log (2026-07-14 → 2026-07-15)](#appendix-h--stage-1-and-stage-2-operational-log-2026-07-14--2026-07-15)
20. [Appendix I — Stage 3 & Stage 4a Pre-Sweep Validation (2026-07-15 → 2026-07-19)](#appendix-i--stage-3--stage-4a-pre-sweep-validation-2026-07-15--2026-07-19)

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

## Appendix H — Stage-1 and Stage-2 Operational Log (2026-07-14 → 2026-07-15)

This appendix appends to the pre-2026-07-14 report and documents two structured
audit/hardening stages performed in weeks 1–2 of the 7-week thesis-project plan.
Nothing above this line has been modified. All facts below were verified against
the live system (Postgres SELECTs, container inspection, and file-level checks);
none are extrapolated from prior documents. Full working files live under
`review/stage1/` (Stage 1, 18 files) and `../review/stage2/` (Stage 2, 34 files;
outside this repo, at the parent `D:\Major Project\review\stage2\`).

### H.1 Purpose

Two motivations drove this operational work:

1. Verify empirically what the pre-existing docs claimed (`SESSION_CONTEXT.md`,
   `STATUS_README.md`) rather than trusting folklore — several claims were stale
   or incorrect (see H.2.6, H.2.7).
2. Harden the data + code substrate for the remaining stages of the thesis
   timeline: dedupe accumulated experimental noise, add token accounting, and
   confirm the metafeatures pipeline handles regression before adding Ames as a
   third dataset.

### H.2 Stage 1 — Read-only audit + confirmed cleanups (2026-07-14)

Stage 1 began as a strictly read-only infrastructure and DB audit spanning four
tasks (docker volume recovery, current infra state, Postgres inventory, repo
layout verification). After the audit produced a nine-item recommendation list,
the researcher approved batched execution of all recommendations; the resulting
mutations are recorded below.

#### H.2.1 Pre-isolation data recovery

The AutoML stack was renamed and port-shifted on 2026-07-10 (containers
`Auto-ML-Postgres` on `:5433`, `Auto-ML-Redis` on `:6380`). The audit checked
for surviving pre-isolation Postgres volumes and containers on the host:

- 5 docker volumes present; 4 matched the "postgres/pgdata/automl/db" filter.
- The only historical Postgres container (`postgres`, image `postgres:latest`,
  ran 2025-11-07 for 27 min, exited cleanly) had bound anonymous volume
  `eda50670…` at `/var/lib/postgresql`.
- Read-only inspection of that volume via a temporary Alpine container showed:
  `PG_VERSION=18`, three databases with OIDs 1/4/5 (the default `template0`,
  `template1`, `postgres`), and **no user database**. The 2025-11-07 cluster was
  initialized but never written to.

**Conclusion: no pre-isolation experimental data survives on the host.** The
current `automl-infra_postgres_data` volume is the sole source of run history.
The legacy anon volume and container were subsequently removed in H.2.4.

#### H.2.2 Trace persistence backfill

Prior to session-2's persistence commit (`8460054`, 2026-07-12), successful B2
runs wrote `.trace.json` and `.verification.json` sidecar files under
`automl-reusables/logs/runs/` but did not persist the JSON into
`run_results.reasoning_trace` / `verification_report`. As of the Stage 1 audit,
only 3 of 57 runs (IDs 55, 56, 57) had populated trace columns.

A one-off backfill script (`review/stage1/backfill_traces.py`) parsed the 16
sidecar pairs, matched each to a `run_results` row by
`(dataset_name, condition, seed, iteration)`, and executed dollar-quoted
JSONB `UPDATE` statements in a single transaction. 14 of 16 sidecars matched;
2 were unmatched because the persisted row's final iteration differs from the
sidecar's iteration (a later attempt superseded the sidecar). **B2 trace
coverage moved from 3/57 to 17/57 (30%).**

#### H.2.3 Titanic dataset row deduplication

The `datasets` table contained two rows with `(source='custom', name='train',
target_col='Survived')` — a re-upload artifact from 2026-07-10:

- `id=1` (17:07, 1 failed B2 run against it, 1 meta_features row)
- `id=2` (17:28, 30 runs against it, 1 meta_features row) — canonical.

Because both `run_results.dataset_id_fkey` and
`meta_features.dataset_id_fkey` are declared `ON DELETE CASCADE`,
`DELETE FROM datasets WHERE id=1;` in a single transaction transitively
removed the 1 orphaned run and 1 orphaned meta_features row. Post-state:
2 datasets, 2 meta_features, 56 run_results.

#### H.2.4 Legacy Docker residue cleanup

The exited `postgres` container from 2025-11-07 and its empty anon volume
(`eda50670…`) were removed after H.2.1 confirmed they held no data. The
unrelated `eyshit-*` compose project (different unrelated work on the same
host) was not touched.

#### H.2.5 Ollama runtime + RQ4 scope decision

Verified Ollama reachability on the host. `gpt-oss:120b-cloud` and
`ministral-3:14b-cloud` respond to `/api/generate`; `qwen3-coder:480b-cloud`
was not present in the registry. Combined with the empirical observation that
all 57 pre-Stage-1 `run_results` rows use only `gpt-oss:120b-cloud`, **RQ4
(cross-backend variation) has zero data and was formally dropped from the
thesis on 2026-07-15**. The thesis is now scoped to RQ1, RQ2, RQ3, RQ5.
Edits to `README.md` and `SESSION_CONTEXT.md` recording the drop were made in
Stage 1 and staged for commit at the end of Stage 2.

#### H.2.6 STATUS document creation

No status file (`STATUS_README*`) existed in any of the 8 sibling repos prior
to the audit — the Stage 2 task text referenced sections of a non-existent doc.
Two files were created at the parent `D:\Major Project\`:

- `STATUS_README.md` (Stage 1) — first draft of a parent-level status doc with
  the seven caveats surfaced during the audit.
- `STATUS.md` (Stage 2) — supersedes the above; canonical single-source-of-truth
  with 7 sections (thesis, panel state, schema state, sibling HEADs, known
  issues with owner + target stage, file index, maintenance rules). Every
  number in the panel-state section is refreshed from a live `SELECT` at write
  time; no folklore is copied forward.

#### H.2.7 Empirically contradicted prior claims

The audit produced two direct contradictions with the pre-existing docs. Both
are recorded for provenance because they matter for reproducing the pre-Stage-1
state:

- Claim: "`/analysis/traces` and `/analysis/rule-usage` return 404". Reality:
  both return 200 on `:8004` and via `:8000` gateway. The endpoints landed on
  2026-07-12 in `automl-analysis-service` commit `2c0d4fe` and gateway commit
  `15fdccb`.
- Claim: "only `run_id=6` has trace data". Reality: prior to Stage-1 backfill,
  IDs 55, 56, 57 were the only trace-carrying rows. Post-backfill, 17 IDs
  carry traces. `run_id=6` is a B0 run — B0 never emits a REASONING trace.

#### H.2.8 Local settings hygiene

The bash permission allowlist in `.claude/settings.local.json` had accumulated
new entries during the audit. Committed as
`80de7cd chore(claude): accumulate local bash allowlist from stage-1 audit session`
following the project's existing pattern (see prior commit `c08152a`).

### H.3 Stage 2 — Bug fixes + schema hardening (2026-07-15)

Stage 2 was scoped to nine tasks with mandatory backup discipline: full pg_dump
before any write, all mutations in explicit transactions, no changes to
`01_schema.sql`. Task-by-task:

#### H.3.1 Backup + snapshots (Task 0)

- `pg_dump -F c` of the live DB (`review/stage2/pre_stage2.dump`, 21,294 bytes).
- Restore dry-run into a scratch database confirmed `SELECT COUNT(*) FROM
  run_results = 56`, matching the post-Stage-1 baseline. The Stage 2 task text
  had expected 57 (pre-Stage-1 folklore); the delta of 1 is entirely accounted
  for by the Titanic dedupe (H.2.3).
- Git HEAD snapshots for all 8 sibling repos written to
  `review/stage2/pre_stage2_head_*.txt`.
- Working tree stashed: uncommitted RQ4 doc edits on `major-project-AutoML`
  moved to `stash@{0}` for a clean Stage 2 tree.

#### H.3.2 Ollama re-verification (Task 1)

The Stage 1 Ollama verdict was confirmed but with one new development:

- `gpt-oss:120b-cloud` — reachable, generates "ok" to the canary prompt.
- `ministral-3:14b-cloud` — reachable, but the model itself now reports
  `retired at 2026-07-15 00:00 PDT`. The service returns text but that text is
  a retirement notice, not the requested content. Effectively dead.
- `qwen3-coder:480b-cloud` — `ollama pull` returned
  `Error: pull model manifest: file does not exist`. Model does not exist in
  the current Ollama registry. Ignored per Ground Rule 4 (RQ4 was already
  dropped in H.2.5 so this is not blocking).

Reachability from `automl-generation-worker` to `http://host.docker.internal:11434`
was confirmed via a Python `urllib` probe from inside the container (the
container image ships without `curl`, so the earlier bash probe failed —
switched to `python -c "import urllib.request..."` and got 3 models back).

#### H.3.3 Ames Housing dataset registration (Task 6 precondition)

Adding Ames Housing (OpenML id 41211) as a third dataset was blocked before
Stage 2 because uploading it required a validated regression path through the
metafeatures pipeline. Task 6's sanity check preceded the upload:

- `POST /datasets/openml {"openml_id": 41211}` → HTTP 201, dataset registered
  as `datasets.id=4`, `task_type=regression`, target `Sale_Price`, 2930 rows
  × 81 columns.
- `POST /meta-features/4?force=true` → HTTP 201 in ~28 s.

The metafeatures response demonstrated all four feature groups branch on
`task_type` correctly:

- `simple.class_balance_ratio: null` (correctly skipped for regression).
- `information.target_entropy: null` (correctly skipped).
- `information.mutual_info_to_target` computed via `mutual_info_regression`
  for all 80 features.
- `landmarks.metric_used: "neg_rmse"` — the regression branch of
  `landmarking.py:52-57` fired. `decision_stump_score = -39144.71`
  (`DecisionTreeRegressor` RMSE), `naive_bayes_score = -279391.89` (`Ridge`
  reused into that slot), `one_nn_score = -39721.89`
  (`KNeighborsRegressor` RMSE).

**No crash, no NaN-across-the-board, no silent fallback.** Stage 4 may enqueue
Ames B2 cells safely.

One schema oddity worth logging: `landmarks.naive_bayes_score` is a
classification-era name and, for regression, holds `Ridge()` RMSE. The name is
misleading even though the numeric value is correct. Renaming (or aliasing per
`metric_used`) is on the Stage-3 handoff list.

#### H.3.4 UNIQUE constraint on `datasets` (Task 3e)

After H.2.3 removed the Titanic duplicate, a UNIQUE constraint was added to
prevent future recurrence:

```sql
ALTER TABLE datasets
  ADD CONSTRAINT datasets_source_name_target_uniq
  UNIQUE (source, name, target_col);
```

The DDL is not currently mirrored in a fresh-init migration file — fresh DB
setups run only `01_schema.sql`, which does not include this constraint.
Adding `03_datasets_unique.sql` is on the Stage-3 handoff list so the two
paths converge.

#### H.3.5 Duplicate run row — deferred (Task 4)

`run_results` still contains a duplicate tuple:
`(dataset_id=3 [Telco], condition='B1', llm_backend='gpt-oss:120b-cloud', seed=44)`
with rows `id=24` (test_score = **1.0000**, `error_category = NULL`) and
`id=32` (test_score = 0.7533, `error_category = NULL`).

Score delta is 24.7% — above the 20% Ground-Rule threshold — so **neither row
was deleted.** Both remain in the DB pending a research-integrity decision.

The pair is diagnostic of two known issues:

1. Cloud-Ollama non-determinism (SESSION_CONTEXT §7 Finding 4): same seed
   twelve minutes apart, different generated pipelines, different scores.
2. A latent bug in `suspicious_leakage` detection: `id=24` scored exactly 1.0
   on a real Telco run, but `error_category` is NULL. The current guard in
   `error_taxonomy.py` fires at `SCORE >= 0.995` — that should have tripped.
   Investigating the discrepancy is on the Stage-3 handoff list.

#### H.3.6 Schema migration — token accounting (Task 5a)

The token-accounting migration was applied to the live DB and mirrored as a
fresh-init migration file:

```sql
-- automl-infra/db/init/02_add_tokens.sql (idempotent)
ALTER TABLE run_results
  ADD COLUMN IF NOT EXISTS prompt_tokens INT,
  ADD COLUMN IF NOT EXISTS completion_tokens INT,
  ADD COLUMN IF NOT EXISTS total_tokens INT GENERATED ALWAYS AS
    (COALESCE(prompt_tokens,0) + COALESCE(completion_tokens,0)) STORED;
```

`01_schema.sql` was not modified. `total_tokens` is a Postgres-computed
GENERATED column and the ORM model omits it — application code writes only the
two source columns.

#### H.3.7 Worker code change — minimal, additive (Task 5b)

Following Ground Rule 7 (worker change must be minimal, additive), the code
diff spans three files across two repos and does not refactor any existing
code:

- `automl-reusables/src/experiments/runner.py`: new function
  `call_llm_with_usage(prompt, backend, seed) -> tuple[str, int | None, int | None]`.
  Returns `(code_text, prompt_tokens, completion_tokens)`, extracting Ollama's
  `prompt_eval_count` and `eval_count`. The original `call_llm` is untouched;
  scripts that consume `call_llm` (`dry_run.py`, `run_sweep.py`) continue
  working without change.
- `automl-generation-service/app/models.py`: two new columns on
  `RunResultRecord` (`prompt_tokens`, `completion_tokens`). `total_tokens` is
  intentionally omitted from the ORM model — Postgres computes it, SQLAlchemy
  should not.
- `automl-generation-service/app/jobs.py`: import switch to
  `call_llm_with_usage`; per-iteration accumulator variables
  (`prompt_tokens_total`, `completion_tokens_total`, `tokens_captured` flag);
  two new kwargs on `_persist`. When Ollama omits the fields (some cloud
  retry paths), `None` is persisted rather than a fabricated zero.

Full diff at `review/stage2/worker_token_diff.patch` (122 lines).

The worker container was rebuilt (`docker compose up -d --build
generation-worker`, ~90 s) and confirmed to be consuming the queue via a
post-restart log tail plus the canary run in H.3.9.

#### H.3.8 Trace-backfill re-verification (Task 7)

Task 7 asked whether pre-2026-07-12 B2 runs could have their `reasoning_trace`
backfilled. H.2.2 had already done exactly this in Stage 1, so Stage 2 Task 7
became documentation: 32 sidecar files present (16 `.trace.json` + 16
`.verification.json`), 14 of 16 pairs mapped to run rows, 2 remained
unmatched, giving current coverage of 17/35 B2 runs (48.6%). **18 B2 rows
remain trace-less** because no sidecar exists on disk for those runs (the
subprocess did not emit a valid REASONING line, or the sidecar was never
written). Remedy is to re-run those cells in Stage 4.

#### H.3.9 Canary run — end-to-end token verification (Task 9c)

To verify the token-logging code path lands in the DB after the worker
rebuild, a single B2 cell was enqueued against the newly-registered Ames
dataset:

```json
{"dataset_id": 4, "condition": "b2_metafeature",
 "llm_backend": "gpt-oss:120b-cloud", "seed": 999,
 "max_iter": 3, "timeout_seconds": 300}
```

The job ran 3 m 30 s across 3 iterations. The Ames B2 pipeline itself failed
(`success = false`, `error_category = runtime_other`) — expected on the first
B2 attempt for a wide, regression dataset in a 3-iteration budget. Independent
of that failure, the token columns populated as designed:

| column            | value | source |
| ----------------- | ----: | ------ |
| prompt_tokens     | 23653 | sum of 3 iterations' `prompt_eval_count`   |
| completion_tokens |  9532 | sum of 3 iterations' `eval_count`          |
| total_tokens      | 33185 | Postgres GENERATED (23653 + 9532 ✔)        |

**End-to-end token plumbing verified.** All future runs will populate these
columns.

### H.4 Cumulative state deltas (pre-Stage-1 → post-Stage-2)

Every numeric here is a live `SELECT` at the end of Stage 2, cross-checked with
the pre-Stage-1 snapshot embedded in the audit outputs.

| Metric                             | Pre-Stage-1 | Post-Stage-1 | Post-Stage-2 |
| ---------------------------------- | ----------: | -----------: | -----------: |
| `datasets` rows                    |           3 |            2 |            3 |
| `meta_features` rows               |           3 |            2 |            3 |
| `run_results` rows                 |          57 |           56 |           56 |
| `sweep_jobs` rows                  |           4 |            4 |            4 |
| B2 runs with `reasoning_trace`     |           3 |           17 |           17 |
| Duplicate `datasets` tuples        |           1 |            0 |            0 |
| Duplicate `run_results` tuples     |           1 |            1 |            1 (deferred) |
| Runs with `prompt_tokens` non-null |     (col absent) | (col absent) |            1 |
| `datasets` unique constraint       |     absent  |     absent   |    present   |
| `run_results` token columns        |     absent  |     absent   |    present   |

### H.5 Schema evolution

Before Stage 1: `01_schema.sql` (unchanged).

After Stage 2:

- Live DB has three added columns on `run_results` and one added UNIQUE
  constraint on `datasets`.
- New migration file `automl-infra/db/init/02_add_tokens.sql` (idempotent)
  brings fresh DB inits into parity with the live DB for the columns.
- The UNIQUE constraint is **not yet** in a fresh-init migration file — this
  is called out as a Stage-3 handoff (H.7 item 9). Until then, a fresh
  `docker compose down -v && up` would produce a DB without the constraint.

### H.6 Code changes committed / uncommitted at end of Stage 2

Post-canary, the working state is:

| Repo                            | Uncommitted change                                                       |
| ------------------------------- | ------------------------------------------------------------------------ |
| `major-project-AutoML`          | Stage-1 doc edits (RQ4 drop) in `stash@{0}`; this appendix           |
| `automl-infra`                  | new file `db/init/02_add_tokens.sql`                                     |
| `automl-reusables`              | added `call_llm_with_usage` in `src/experiments/runner.py`               |
| `automl-generation-service`     | token columns in `app/models.py`, token plumbing in `app/jobs.py`        |
| Other 4 repos                   | clean                                                                    |

### H.7 Handoff to Stage 3

Ordered by research-integrity impact:

1. **Manual sign-off on the deferred duplicate run** (`run_id=24` vs `run_id=32`).
   Recommended: keep 32, archive 24 as a diagnostic case study.
2. **Fix the `suspicious_leakage` guard boundary** in
   `automl-reusables/src/execution/error_taxonomy.py`. Score exactly 1.0
   should trip the `>= 0.995` guard; investigate why `run_id=24` did not.
3. **Rename the `naive_bayes_score` landmarks slot** (or add a
   `metric_used`-aware alias in `contracts.py`) so regression's Ridge RMSE
   does not read as Naive Bayes.
4. **Backfill script or explicit exclusion decision** for the 18 remaining
   trace-less B2 runs. Sidecars do not exist on disk for these; either accept
   them as pre-verification-era, or re-run those specific cells in Stage 4.
5. **Upload the remaining 27 CC18 datasets** (the catalog exposes them, the
   panel does not include them). Necessary before Stage-4 sweeps that hope to
   answer RQ5.
6. **Add `03_datasets_unique.sql`** so fresh DB inits get the UNIQUE
   constraint added in H.3.4.
7. **Un-stash and commit** the Stage-1 RQ4 doc edits currently in
   `stash@{0}` on `major-project-AutoML`.
8. **Replace or remove `ministral-3:14b-cloud`** references anywhere in
   prompts, README examples, or tests — the model is retired.
9. **Remove `qwen2.5:3b`** from the Ollama registry per SESSION_CONTEXT §13
   (cloud-only policy).

### H.8 Reproducibility notes

- Full Stage 1 audit outputs (18 files) at
  `major-project-AutoML/review/stage1/`.
- Full Stage 2 outputs (34 files) at `D:\Major Project\review\stage2\`
  (outside this repo, at the parent — spans all 8 sibling repos).
- Both stages' report files (`STAGE1_REPORT.md`, `STAGE2_REPORT.md`) are
  self-contained and include restoration instructions if a future replay of
  either stage is needed.
- `STATUS.md` at the parent is the current single-source-of-truth for
  panel state, schema state, and open issues.

---

## Appendix I — Stage 3 & Stage 4a Pre-Sweep Validation (2026-07-15 → 2026-07-19)

This appendix appends to the report and documents two structured stages that
follow Appendix H and precede the Stage 4b full sweep. Nothing above this line
has been modified. Facts below were verified against the live system on the
D:\ machine (`DESKTOP-LK0JISF`) unless a Stage 3 sub-section explicitly notes
that the work was performed on the C:\ machine. Full working files:

- Stage 3 (28 files): `D:\Major Project\review\stage3\` — original host was
  the C:\ machine, artifacts copied over.
- Stage 4a (33 files): `D:\Major Project\review\stage4a\`.

### I.1 Purpose

Two motivations:

1. **Close the gaps Appendix H flagged as Stage-3 handoff** (H.7): duplicate
   run sign-off, leakage-guard investigation, `naive_bayes` landmarker
   rename, CC18 dataset expansion, `03_datasets_unique.sql` migration, cloud
   Ollama model panel verification.
2. **Certify the runway before Stage 4b commits to a 300+ cell sweep.** Two
   things a full sweep cannot tolerate: brittle data paths that vary by
   host, and model panel entries that silently unavailable at run time. Both
   are established here.

### I.2 Stage 3 — Consolidation and audit (2026-07-15 → 2026-07-16, C:\ machine)

Full write-up in `review/stage3/STAGE3_REPORT.md`. This section is a
compressed record for paper-facing readers; the numbered items below map
1:1 to the Stage 3 report's task list.

#### I.2.1 Ollama model panel — as of 2026-07-15

The intended Stage 4 panel was probed 5× each via `/api/generate`:

| model                     | probes | median latency | tokens present | `thinking` present |
|---------------------------|--------|----------------|----------------|--------------------|
| gpt-oss:120b-cloud        | 5/5    | 1.07 s         | yes            | **yes**            |
| gemma4:31b-cloud          | 5/5    | 1.07 s         | yes            | no                 |
| nemotron-3-ultra:cloud    | 5/5    | 1.55 s         | yes            | yes                |

Two paper-relevant findings:

1. **`gpt-oss` emits a non-empty `thinking` field**, not just nemotron. Any
   RQ4 split framed as "reasoning vs non-reasoning models" is muddied by
   this — the axis is not clean at the panel level.
2. **Backup pool** of 4 candidates (`ministral-3:14b-cloud`,
   `glm-5.1:cloud`, `devstral-small-2:24b-cloud`, `nemotron-3-nano:30b-cloud`):
   0/4 pullable at time of Stage 3 audit; all returned 404. Stage 3
   concluded no drop-in replacement existed. **This finding was overtaken
   by Stage 4a** — nemotron-3-nano *is* pullable now (§I.3.4).

#### I.2.2 `suspicious_leakage` guard — root cause

Stage 2's assumption that the guard was buggy (Appendix H, H.3) was a
mis-attribution to the wrong file. The guard lives at
`automl-reusables/src/execution/runner.py:120` (`SCORE >= 0.995`,
classification-only) and is correct in current code. `run_id=24` escaped
because the guard commit `3838881` landed at 2026-07-10 18:29:16 UTC —
11 minutes *after* the row was persisted at 18:18:38 UTC. No code change
required; retroactively flagging the pre-guard row is a data-repair
operation, applied under `review/stage3/leakage_retroactive_transcript.txt`.

#### I.2.3 `naive_bayes` landmarker rename

The `naive_bayes_score` landmarker key is metric-agnostic in code but
metric-specific in interpretation: on regression tasks it holds a Ridge
RMSE, not a Naive Bayes score. Renamed in `contracts.py` and every
downstream reference. Patch: `review/stage3/naive_bayes_rename_diff.patch`.

#### I.2.4 Nemotron divergence probe — INCONCLUSIVE (Stage 3), resolved in
Stage 4a as UNAVAILABLE

The Stage 3 probe of `nemotron-3-ultra:cloud` on Titanic (three cells, one
per B0/B1/B2) crashed inside the worker with `FileNotFoundError`. The CSV
referenced by `datasets.id=2` was missing on the C:\ host — the machine
had never been the one that received the original CSV upload. This is the
architectural flaw that motivates I.3.

The probe was re-attempted in Stage 4a on the D:\ machine and resolved to
FAIL (§I.3.3), but by unavailability, not by content.

#### I.2.5 Schema hardening

`03_datasets_unique.sql` written and applied, closing the Stage 2 H.7 item
that fresh DB inits lacked the UNIQUE constraint on `datasets(source,
name, target_col)`. Migration transcript:
`review/stage3/migration_03_apply.txt`.

### I.3 Stage 4a — Pre-Sweep Validation (2026-07-19, D:\ machine)

Full write-up in `../review/stage4a/STAGE4A_REPORT.md`. This is the
substrate-hardening stage that separates Stage 3 audit findings from the
Stage 4b sweep. Four validations, each with a binary go/no-go outcome that
directly determines the panel Stage 4b enters with.

#### I.3.1 Substrate — CSV/DB stable-path migration

Symptom (inherited from Stage 3): `datasets.train_path` values pointed at
`/opt/automl-reusables/data/experiments/custom/tmp<random>/train.csv`.
The `tmp<random>` folder name is the stem of a Python
`tempfile.NamedTemporaryFile()` handle from the original upload session
and only exists on the host that received that particular upload. This
made the DB row semantically host-scoped even though the paths look
universal.

Migration (Stage 4a Task 1, D:\ machine, all inside one transaction):

1. All 6 CSVs (3 datasets × train + test) confirmed present and readable
   by `automl-generation-worker`.
2. Copied to `/opt/automl-reusables/data/experiments/stable/<dataset_id>/{train,test}.csv`.
3. `UPDATE datasets SET train_path = ..., test_path = ...` for each of 3
   rows, in one `BEGIN/COMMIT` with pre-commit `SELECT` verification.
4. Old copies retained per Ground Rule 7 (fallback if Stage 4b breaks on
   the new path).
5. **Code fix in `automl-data-service/app/service.py`**: added
   `_relocate_to_stable(dataset)` called after both `upload_csv()` and
   `import_openml()` commit. Future dataset registrations land at
   `stable/<id>/` on first save. Container not rebuilt in Stage 4a —
   patch applies at the next `docker compose build data-service`.
   Diff: `review/stage4a/data_service_stable_paths.patch`.
6. Post-migration `automl-generation-worker` verification: 6/6 CSVs
   readable at the new paths (`csv_worker_verification.txt`).

Paper implication: the reproducibility appendix (G) now has an explicit
statement of where dataset CSVs live in the container filesystem and why
that location is stable across hosts. This closes the ambiguity that
Stage 3's crash exposed.

Deviation from Stage 4a plan: the plan proposed
`automl-data-service/data/uploads/<id>/` as the stable target. That path
is not bind-mounted into `automl-generation-worker`; only the
`automl-reusables` directory is. Same architectural intent, reachable
path.

#### I.3.2 Panel — cloud model availability re-check

`ollama pull` and probe-via-API were repeated for the three primaries
and the intended backup:

| model                        | ollama pull | `/api/generate` | tokens | thinking |
|------------------------------|-------------|-----------------|--------|----------|
| gpt-oss:120b-cloud           | already local | ok            | yes    | yes      |
| gemma4:31b-cloud             | success       | ok            | yes    | no       |
| nemotron-3-ultra:cloud       | **404 MANIFEST_UNKNOWN** | **404 not found** | — | — |
| nemotron-3-nano:30b-cloud    | success       | ok            | yes    | yes      |
| nemotron-3-super:cloud       | success       | ok            | yes    | yes      |

Two paper-relevant findings:

1. **`nemotron-3-ultra:cloud` was delisted from the Ollama registry
   between Stage 3 (2026-07-15, present) and Stage 4a (2026-07-19,
   gone).** The library page at `ollama.com/library/nemotron-3-ultra`
   still advertises the `cloud` tag, but direct registry probes for
   `{latest, cloud, ultra, 341b-cloud, ultra-cloud}` all return
   `MANIFEST_UNKNOWN`. This is not a client-version issue — the two
   sibling nemotron variants pulled fine.
2. **The intended-backup nemotron-3-nano is now available** (Stage 3
   marked it as 404). And a sibling 120b variant, `nemotron-3-super:cloud`,
   is also present. The panel has options; the specific model named in the
   task plan does not.

Consequence for Stage 4b: the third primary slot is empty. Reviewer
decides between shipping a 2-backend panel (360 cells) or substituting
`nemotron-3-super` (540 cells) after one Task-2-style divergence probe.
Recorded evidence: `review/stage4a/nemotron_ultra_unavailable.txt`.

#### I.3.3 Task 2 (nemotron divergence) — verdict FAIL by unavailability

Cannot be executed with the specified model. Per plan verdict-branching,
this drops nemotron-ultra from the panel. Not a model-content finding,
but the *rate* of registry churn (a listed cloud model disappearing
within 4 days) is a limitation worth flagging in §9 of any published
version of this work: the panel is not stable across the reproducibility
window.

#### I.3.4 Task 3 (backup qualification) — verdict CONDITIONAL

`nemotron-3-nano:30b-cloud` was pulled, probed 5× (all OK, both token
fields populated, `thinking` present, median ≈1.0 s), and driven through
one B2 test cell on Titanic (seed=9500, max_iter=3, timeout=300):

- Row 59 in `run_results`: `success=false, iterations_used=3,
  prompt_tokens=8900, completion_tokens=44843, has_trace=true,
  error_category=import_error`.
- Root cause: model produced code that uses
  `sklearn.impute.IterativeImputer` without the required
  `from sklearn.experimental import enable_iterative_imputer` — a
  well-known sklearn gotcha the model did not self-correct across three
  retries. `category_encoders` (also imported) is installed and did not
  cause the error.
- Mechanically integrated: gateway → worker → generation-service →
  ollama loop works, tokens capture, trace parser fires.
- Fit as backup (fallback if a primary drops mid-Stage-4b); **not fit as
  a first-choice primary substitute**.

Full write-up: `review/stage4a/backup_qualification.md`.

#### I.3.5 Task 4 (36-cell canary) — verdict PASS (88.9%)

Design B canary: 2 datasets (Titanic + Telco) × 3 conditions × 2 backends
(`gpt-oss:120b-cloud`, `gemma4:31b-cloud`) × 3 seeds = 36 cells. Panel
reduced from the plan's 54 because Task 2 dropped nemotron-ultra.

Wall clock: 21 min 05 s (11:25:43 → 11:46:16 UTC). Success 32/36.
Zero worker restarts, zero queue stalls, zero infrastructure errors.

Per-model summary (source: `review/stage4a/canary_summary.txt`):

| llm_backend         | attempted | succeeded | notes                                        |
|---------------------|-----------|-----------|----------------------------------------------|
| gpt-oss:120b-cloud  | 18        | 18        | Clean sweep; 2 B2 cells needed retries       |
| gemma4:31b-cloud    | 18        | 14        | All 4 failures in B2: 1 `import_error` (Titanic), 3 `reasoning_unfaithful` (Telco) |

Cross-model verifications:

- **Trace parser**: `has_trace = true` on 36/36. Both models, all three
  conditions. Parser is not model-specific. Caveat: `reasoning_trace` is
  populated for B0 and B1 too (not just B2 as a metafeature-linked prompt
  would suggest), so downstream analyses of *metafeature-linked
  reasoning* should filter on `condition = 'B2'` — this is a reporting
  observation, not a bug.
- **Token accounting**: both `prompt_tokens` and `completion_tokens`
  populated on 36/36. Values scale as expected (B2 prompt ≈ 10× B0
  prompt).
- **Leakage guard**: 0 `suspicious_leakage` hits. Note: Telco B2 seed=2002
  on gpt-oss produced test_score = 0.505 (near-random). Not a leakage
  hit (guard fires near 1.0), just a genuine bad-model-choice outcome
  worth reporting in Stage 4b analysis rather than filtering out.

Paper implication (RQ2, RQ6): **gemma4 exhibits a specific B2 failure
profile — all 3 seeds of Telco B2 rejected as `reasoning_unfaithful`**.
This is the guardrail catching a genuine faithfulness violation
(model's cited rules do not appear in its emitted code), consistent with
the report's core finding that faithfulness and correctness are
orthogonal. On this 3-seed slice the effect is deterministic within
gemma4.

Wall-clock extrapolation for Stage 4b Design B (12 datasets):

- 2-backend / 360 cells: 3.5 h naive, 6–10 h realistic.
- 3-backend / 540 cells: 5.3 h naive, 9–15 h realistic.

Both are well under the plan's 60 h alarm bar. The plan's a-priori
estimate of 41 h looks conservative.

Full write-up: `review/stage4a/canary_analysis.md` and
`STAGE4A_REPORT.md`.

### I.4 Cumulative state deltas (post-Stage-2 → post-Stage-4a)

| item                                 | post-Stage-2 | post-Stage-3 | post-Stage-4a |
|--------------------------------------|--------------|--------------|---------------|
| `run_results` row count              | 56           | 57           | **94**        |
| Datasets registered                  | 3            | 3            | 3 (Stage 4b Task 0 imports 9 more) |
| CSV path layout                      | `custom/tmp<rand>/` | `custom/tmp<rand>/` | **`stable/<id>/`** |
| Data-service upload code             | temp path    | temp path    | **stable path (patch, not rebuilt)** |
| Nemotron-ultra in panel              | assumed      | INCONCLUSIVE | **DROPPED (unavailable)** |
| Backup model status                  | 0/4 pullable | 0/4 pullable | **nemotron-nano CONDITIONAL, nemotron-super candidate** |
| Sweep-runner max cells stress-tested | ~1           | ~1           | **36 (0-fault)** |

### I.5 Code changes committed / uncommitted at end of Stage 4a

| Repo                            | Uncommitted change                                                      |
|---------------------------------|-------------------------------------------------------------------------|
| `major-project-AutoML`          | this appendix + ToC + working-dates line                                |
| `automl-data-service`           | `app/service.py`: added `_relocate_to_stable`, hooked into two flows    |
| Other 6 repos                   | clean                                                                   |
| DB                              | 3 `datasets` rows updated with stable paths; +37 `run_results` rows     |

Stage 4a rollback anchor: git tag `stage3-complete` on all 8 repos.

### I.6 Handoff to Stage 4b (reviewer signoff required)

Ordered by decision blocking:

1. **Panel decision on the vacated nemotron-ultra slot.** Three options:
   (a) ship 2-backend (360 cells), (b) substitute `nemotron-3-super:cloud`
   after one Task-2-style probe, (c) substitute `nemotron-3-nano:30b-cloud`
   (not recommended as primary — see I.3.4). Recommendation: (b) if
   appetite exists for the extra probe, else (a).
2. **Rebuild `automl-data-service`** so the `_relocate_to_stable` patch
   takes effect before Stage 4b Task 0 imports 9 OpenML datasets. Without
   the rebuild, those imports land in `custom/tmp<rand>/` again — the
   very failure mode this stage exists to eliminate.
3. **Import the 9 remaining Stage 4b Design B datasets** via
   `POST /datasets/openml`: OpenML IDs 15, 37, 38, 44, 151, 1461, 1487,
   1510, 1590. All are listed in
   `automl-reusables/src/experiments/datasets.py::SELECTED_CLASSIFICATION`.
4. **Optional pre-Stage-4b probe** on a large-dataset B2 cell (e.g. adult,
   openml 1590) to tighten the wall-clock extrapolation for the
   largest-dataset regime. Current canary is small-medium only.

### I.7 Reproducibility notes

- Every Stage 4a mutation was preceded by `pg_dump -F c` and a restore
  test into a scratch DB (see `pre_stage4a.dump` and `backup_verified.txt`;
  baseline restored 57/57 rows exactly).
- Post-stage backup at `review/stage4a/post_stage4a.dump` (94 rows) —
  restores forward from 57 to 94 if Stage 4b needs a clean fall-back
  point before the full sweep begins.
- All 8 repositories were tagged `stage3-complete` at Stage 4a start.
  The tag names Stage 3, not Stage 4a, so it marks *the state the
  reviewer signed off on before the substrate migration*. Any future
  Stage 4a rollback should target this tag.
- Host: `DESKTOP-LK0JISF` (D:\). All Stage 4a paths in the DB assume
  this host until Stage 4b or later formalizes the container-relative
  path (`/opt/automl-reusables/data/experiments/stable/<id>/`).

---

*End of report. This document is the substantive record; SESSION_CONTEXT.md is its concise index.*
