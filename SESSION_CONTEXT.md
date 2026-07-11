# AutoML Project — Full Context Snapshot

**Purpose of this file:** paste this whole document into a new Claude session and Claude will be up to speed. No other context needed.

**Last updated:** 2026-07-11 (session 2)
**Owner:** Samarth Adhikari (`@Samarth-Ad`), GitHub org `Major-Proj-AutoML`

---

## 1. What the project is

**Meta-Feature-Guided Prompting for LLM-Driven Tabular AutoML.**

A research project asking: *does feeding an LLM computed dataset statistics (meta-features + decision rules) produce better auto-generated scikit-learn pipelines than naive prompting?*

Three prompting conditions:

| Condition | What the LLM sees |
|---|---|
| **B0 naive** | Just the task description |
| **B1 schema** | + dataset schema (dtypes, shape, head) |
| **B2 meta-feature** | + full meta-feature JSON + 14 decision rules |

**Research questions:**

- **RQ1** — does B2 outperform B0/B1 on predictive accuracy?
- **RQ2** — does B2 reduce code-generation failures?
- **RQ3** — does B2 need fewer LLM retries to converge?
- **RQ4** — how does the effect vary across LLM backends?
- **RQ5** — does the effect vary by dataset size (small/medium/large)?

**Novel contribution added in session 2:** a mechanical verification system that separates *faithfulness* (does the code match the LLM's claimed reasoning?) from *correctness* (does the pipeline generalize?). Empirically confirmed these are orthogonal — the LLM can produce a fully faithful trace whose pipeline still collapses to baseline.

---

## 2. The 8 sibling repos (polyrepo, NOT monorepo)

All under `D:\Major Project\` locally, pushed to their respective GitHub remotes.

```
D:\Major Project\
├── major-project-AutoML\           github.com/Samarth-Ad/major-project-AutoML     (personal)
├── automl-reusables\               github.com/Major-Proj-AutoML/automl-reusables  ← shared library
├── automl-infra\                   github.com/Major-Proj-AutoML/automl-infra      ← docker-compose + schema
├── automl-data-service\            github.com/Major-Proj-AutoML/automl-data-service       (:8001)
├── automl-metafeatures-service\    github.com/Major-Proj-AutoML/automl-metafeatures-service (:8002)
├── automl-generation-service\      github.com/Major-Proj-AutoML/automl-generation-service   (:8003 + worker)
├── automl-analysis-service\        github.com/Major-Proj-AutoML/automl-analysis-service     (:8004)
└── automl-gateway\                 github.com/Major-Proj-AutoML/automl-gateway              (:8000)
```

Plus infra containers:
- **PostgreSQL 16** on host port `:5433` → container `:5432` (container: `Auto-ML-Postgres`)
- **Redis 7** on host port `:6380` → container `:6379` (container: `Auto-ML-Redis`)

`automl-reusables` is installed as a git dependency by every service's `pyproject.toml`:
```toml
"automl-reusables @ git+https://github.com/Major-Proj-AutoML/automl-reusables.git@main"
```

In containers it's ALSO bind-mounted at `/opt/automl-reusables` (via `docker-compose.full.yml`) so host edits to the shared library reflect immediately after a service restart, no rebuild needed. The importable module name is `src` (not `automl_reusables`) — historical decision to avoid touching import statements.

---

## 3. Architecture at a glance

```
Frontend (later)  ──► Gateway :8000 (CORS, proxy, composed workflows)
                          │
        ┌─────────────────┼─────────────────┬──────────────────┐
        ▼                 ▼                 ▼                  ▼
   data-service     metafeatures-      generation-       analysis-service
   :8001              service :8002    service :8003     :8004
   (CSV / OpenML     (extract +        (async LLM        (RQ1-RQ5 stats,
    upload,           cache meta-       runs via RQ,      rule usage,
    dataset           features)         retry with        faithfulness)
    registry)                           feedback)
        │                 │                 │                  │
        └───► PostgreSQL :5433 ◄────────────┴──────────────────┘
                                            │
                                       Redis :6380 ◄── generation-worker
                                                       (calls Ollama, executes
                                                        generated code in subprocess,
                                                        verifies B2 reasoning trace)
                                                            │
                                                            ▼
                                                     Ollama :11434 (host)
                                                     with cloud auth
```

**Data model on disk:**
- CSVs (uploaded, split into train/test, and `original.csv` pre-split): Docker volume `automl_data` mounted at `/data`
- Generated code + sidecars: bind mount `/opt/automl-reusables/logs/runs/`
  - `train_B2_seed42_iter0_xxx.py` — LLM-generated code
  - `.py.stdout.txt` / `.py.stderr.txt` — subprocess output
  - `.trace.json` — structured decision list (B2 only)
  - `.verification.json` — mechanical audit report (B2 only)
  - `.reasoning_raw.txt` — raw REASONING line when JSON parsing fails (debug)

**Data model in Postgres:**
- `datasets` — dataset registry (source, target_col, task_type, train/test paths, n_rows, n_cols)
- `meta_features` — cached per-dataset meta-features (JSONB)
- `run_results` — every run, including `reasoning_trace` and `verification_report` (JSONB, nullable)
- `sweep_jobs` — sweep progress tracking

---

## 4. Bring it up (fresh machine → running stack)

Prereqs on the host: Docker Desktop, Ollama, Git, ~15 GB free disk.

```powershell
# 1. Clone all 8 repos as siblings (choose any parent directory)
mkdir automl && cd automl
git clone https://github.com/Samarth-Ad/major-project-AutoML.git
for repo in reusables infra data-service metafeatures-service generation-service analysis-service gateway; do
    git clone "https://github.com/Major-Proj-AutoML/automl-$repo.git"
done

# 2. Start Ollama and pull a cloud model
ollama serve       # in another terminal
ollama pull gpt-oss:120b-cloud

# 3. Bring up the full stack (Postgres + Redis + 5 services + worker)
cd automl-infra
docker compose -f docker-compose.yml -f docker-compose.full.yml up -d --build

# 4. Verify
curl http://localhost:8000/health
# Expect: {"status":"ok","upstream":{"data":true,"metafeatures":true,"generation":true,"analysis":true}}
```

First build takes 5–10 min. Subsequent starts ~30 sec.

**Interactive API docs:**
- http://localhost:8000/docs (gateway — this is the one to use)
- Each backend also has its own `/docs` on ports 8001–8004

---

## 5. What each service does (endpoint reference)

### Gateway `:8000`
Frontend entry point. Proxies to backends. Adds two composed workflows:
- `POST /workflows/upload-and-extract` — upload CSV → extract meta-features (returns both)
- `POST /workflows/full-run` — upload + meta-features + enqueue single generation cell

### data-service `:8001`
Dataset registry.
- `POST /datasets` (multipart: `file`, `target_col`, optional `task_type`, `seed`, `test_size`)
- `POST /datasets/openml` — import an OpenML dataset by id
- `GET /datasets`, `GET /datasets/{id}`, `GET /datasets/{id}/preview?n_rows=`, `DELETE /datasets/{id}`
- `GET /datasets/openml/catalog` — the 28 CC18 datasets pre-registered

Every upload saves **three CSVs** in the container's `/data/custom/<name>/`:
- `original.csv` — pre-split, unmodified (what users get as the "clean dataset" artifact)
- `train.csv`, `test.csv` — the stratified split

### metafeatures-service `:8002`
- `POST /meta-features/{dataset_id}[?force=true]` — compute + cache
- `GET /meta-features/{dataset_id}` — fetch cached
- `DELETE /meta-features/{dataset_id}`

Four groups computed: simple, distributional, information (mutual info + correlations), landmarking (decision-stump / NB / 1-NN accuracy from cross-val).

### generation-service `:8003` + `generation-worker`
Async LLM generation.
- `POST /runs` — enqueue one cell: `(dataset_id, condition, llm_backend, seed, max_iter, timeout_seconds)`. Returns `rq_job_id` immediately (202 Accepted).
- `GET /runs`, `GET /runs/{id}` (filter by `dataset_id`/`condition`/`llm_backend`)
- `POST /sweeps` — enqueue cartesian product of `dataset_ids × conditions × llm_backends × seeds`
- `GET /sweeps`, `GET /sweeps/{id}`

Worker (`python -m app.worker`) picks up jobs, fetches dataset from data-service, calls Ollama, executes generated code in a subprocess (via `execute_pipeline` from reusables), retries with `build_error_feedback` up to `max_iter` on failure.

**B2 verification hook** — for successful B2 runs, the worker also:
1. Reads subprocess stdout sidecar
2. Extracts `REASONING: {...}` line
3. Runs `verify_reasoning` (AST + meta-feature check)
4. Persists trace and report in Postgres AND as sidecar JSONs
5. If unfaithful, flips result to `success=false, error_category="reasoning_unfaithful"` (trace/report still saved for inspection)

### analysis-service `:8004`
Read-only queries over `run_results`.
- `GET /analysis/summary` — mean/median score + success rate per (condition, llm_backend)
- `GET /analysis/errors` — failure count per (condition, error_category)
- `GET /analysis/iterations` — mean iterations-to-success per condition (RQ3)
- `GET /analysis/models` — per-backend breakdown (RQ4)
- `GET /analysis/size-stratified` — small/medium/large dataset bucketed
- `GET /analysis/wilcoxon?a=B0&b=B2` — paired Wilcoxon signed-rank test
- `GET /analysis/traces[?dataset_id=&limit=]` — every B2 run with its persisted trace + verification report
- `GET /analysis/rule-usage` — aggregate: for each B2 rule, how many times it was cited and how many verified

---

## 6. B2 reasoning verification — the novel contribution

The B2 prompt (in `automl-reusables/src/conditions/b2_metafeature.py`) instructs the LLM to print, in addition to `SCORE: <number>`, a JSON line:

```
REASONING: {"decisions": [
  {
    "step": "scaling",
    "rule_id": 7,
    "meta_feature": "distributional.outlier_ratio_per_numeric.Fare",
    "observed_value": 0.128,
    "threshold": 0.05,
    "action": "RobustScaler",
    "applied_to": ["Fare"]
  },
  ...
]}
```

`extract_reasoning(stdout)` (in `automl-reusables/src/execution/metrics.py`) parses that line into a `ReasoningTrace` Pydantic model.

`verify_reasoning(trace, code, meta)` (in `automl-reusables/src/execution/verification.py`) checks each decision:
1. **Value check:** resolve the dotted `meta_feature` path against the actual `MetaFeatures` object. Does `observed_value` match within tolerance 1e-3?
2. **Action check:** AST-parse the generated code. Does the `action` symbol appear as an import target OR a called name?

Both checks must pass for the decision to count as "faithful." `verify_reasoning` returns a `VerificationReport` with per-decision `DecisionVerdict`s and an overall `faithful: bool`.

**Live example** (Titanic seed=202, run_id=57): all 6 cited decisions verified, `faithful: true`, but the pipeline still scored 0.5 (majority baseline). The LLM applied TargetEncoder to `Name` (712 unique values) exactly as it claimed — but that decision destroyed generalization.

Every B2 run produces four sidecar files next to the generated code, and the trace/report also land in Postgres as JSONB.

---

## 7. Empirical findings so far (across ~30+ B2 runs on Titanic + Telco)

### Finding 1 — Rule 3 + Rule 4 co-firing is a deterministic trap on Titanic

- **Rule 3:** "if categorical_cardinality > 20, use TargetEncoder"
- **Rule 4:** "if n_categorical > n_numeric, use gradient boosting (LGBMClassifier)"

Whenever both rules fire on Titanic (which has `Name` cardinality 712 in 712-row training set), the pipeline collapses to majority-class prediction (balanced_accuracy = 0.500). LightGBM overfits perfectly on the target-encoded near-unique identifier during training; unseen names in test → global prior → constant prediction.

When Rule 14 (linear models) fires instead of Rule 4, the pipeline scores ~0.75+.

Confirmed across seeds 42, 43, 100, 101, 120, 124, 200, 202 — deterministic co-firing pattern.

### Finding 2 — Faithfulness ≠ Correctness

Every trap run above had a **fully faithful trace** (100% of decisions verified). The LLM correctly documented what it did; what it did was catastrophic.

This is a novel result — existing faithfulness work assumes faithful reasoning implies better outcomes. Our data mechanically disproves that when the rule base contains adversarial co-firings.

### Finding 3 — When the LLM emits a trace, it is completely honest

Across ~28 verified decisions in clean sweep #4, **28/28 verified** (100%). No fabricated `observed_value`s. No cited actions missing from code. The LLM does not lie when compelled to produce structured, machine-checkable output — but it can be honest about doing dumb things.

### Finding 4 — Ollama cloud does not fully respect `seed`

B0 runs on Telco produced identical scores (0.7051) across 4 of 5 seeds. Suggests Ollama's cloud inference doesn't consistently honor the `seed` parameter for hosted models. Mention this in the methodology section — reproducibility of a single-cell run at a given seed is not guaranteed for cloud backends.

---

## 8. Error taxonomy (`ErrorCategory` in `contracts.py`)

| Category | When it fires |
|---|---|
| `syntax_error` | Generated Python fails to parse |
| `import_error` | `ModuleNotFoundError` / `ImportError` |
| `missing_name` | `NameError: 'X' is not defined` (LLM used a class it didn't import) |
| `api_hallucination` | `has no attribute` / `is not callable` / `unexpected keyword` |
| `shape_mismatch` | Sklearn shape errors, "Found input variables with inconsistent samples" |
| `type_error` | TypeError on categoricals or string-to-float conversion |
| `deprecated_api` | DeprecationWarning / FutureWarning |
| `metric_mismatch` | Reserved, not currently emitted |
| `suspicious_leakage` | Classification `SCORE >= 0.995` — real datasets don't hit this without leakage |
| `reasoning_unfaithful` | B2 trace missing, malformed, or contains a decision that fails verification |
| `timeout` | Subprocess exceeded per-cell `timeout_seconds` |
| `runtime_other` | Anything else runtime |
| `resource_limit` | MemoryError, ResourceWarning |
| `infrastructure` | Ollama unreachable, DB connection lost, etc. (set by worker, not taxonomy) |

---

## 9. Test coverage

| Repo | Tests |
|---|---|
| major-project-AutoML | 49 (includes reusables coverage via `import src.*`) |
| automl-data-service | 6 |
| automl-metafeatures-service | 8 |
| automl-generation-service | 10 (2 new for B2 verification path) |
| automl-analysis-service | 9 |
| automl-gateway | 6 |

All tests use SQLite in-memory + monkeypatched external calls (Ollama, HTTP clients, subprocess). No infra required for `pytest`.

Run all:
```powershell
cd "D:\Major Project\major-project-AutoML" && .venv\Scripts\python.exe -m pytest tests/
# repeat cd + pytest for each service
```

---

## 10. Useful commands (workflow cheatsheet)

### Enqueue a single run (bypasses Postman)
```powershell
curl.exe -s -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d '{
  "dataset_id": 2, "condition": "b2_metafeature",
  "llm_backend": "gpt-oss:120b-cloud", "seed": 42,
  "max_iter": 3, "timeout_seconds": 180
}'
```

### Enqueue a sweep
```powershell
curl.exe -s -X POST http://localhost:8000/sweeps -H "Content-Type: application/json" -d '{
  "dataset_ids": [2, 3],
  "conditions": ["b0_naive", "b1_schema", "b2_metafeature"],
  "llm_backends": ["gpt-oss:120b-cloud"],
  "seeds": [42, 43, 44, 45, 46],
  "max_iter": 3, "timeout_seconds": 180
}'
```

### Inspect a specific run's trace
```powershell
curl.exe -s "http://localhost:8000/runs/57" | python -m json.tool
```

### Aggregate rule usage
```powershell
curl.exe -s "http://localhost:8000/analysis/rule-usage" | python -m json.tool
```

### Tail worker logs
```powershell
docker logs -f automl-generation-worker
```

### Peek at Postgres directly
```powershell
docker exec -it Auto-ML-Postgres psql -U automl -d automl -c "SELECT id, dataset_id, condition, success, test_score, error_category FROM run_results ORDER BY id DESC LIMIT 10;"
```

### After editing code

- Files in `automl-reusables/src/` → restart the worker: `docker restart automl-generation-worker` (bind-mounted, no rebuild)
- Files in a service's `app/` folder → `docker cp app/... automl-<service>:/srv/app/... && docker restart automl-<service>` (baked into image, not bind-mounted)
- Compose file changes → `docker compose -f docker-compose.yml -f docker-compose.full.yml up -d --build <service>`

### Full stack down / up
```powershell
cd "D:\Major Project\automl-infra"
docker compose -f docker-compose.yml -f docker-compose.full.yml down
docker compose -f docker-compose.yml -f docker-compose.full.yml up -d
# nuclear (destroys DB + Redis + uploaded CSVs):
docker compose -f docker-compose.yml -f docker-compose.full.yml down -v
```

---

## 11. Session-1 vs session-2 changes

### Session 1 (2026-07-09 to 07-10)
- Extracted `src/` from `major-project-AutoML` into standalone `automl-reusables` sibling repo
- Built the 5 microservices + gateway, all with Dockerfiles and tests
- Set up `automl-infra` with Postgres + Redis + full-stack overlay
- Created GitHub org `Major-Proj-AutoML`, pushed all 7 repos
- Ran first experiments on Titanic + Telco Churn (30 cells across B0/B1/B2)
- Documented 3 failure modes: silent collapse (rule 3+4 trap), rule crash, silent leakage inflation
- Added `suspicious_leakage` guardrail (`SCORE >= 0.995` on classification → flagged)
- Discovered Ollama cloud non-determinism (same seed, different code)

### Session 2 (2026-07-11)
- **Structured reasoning trace system** — LLM must print `REASONING: {json}` alongside SCORE
- **Mechanical verification module** — AST parse of code + dotted-path resolve of meta-features
- **`reasoning_unfaithful` error category** — unverified traces flip to failed, are excluded from analysis
- **`original.csv` pre-split artifact** — saved alongside train.csv/test.csv, available to users
- **JSONB columns in Postgres** — `run_results.reasoning_trace` and `verification_report`, live migration applied
- **Two new analysis endpoints** — `/analysis/traces` and `/analysis/rule-usage`, exposed via gateway
- **Fixed schema forgiveness** — `Decision.threshold` accepts both float and string (LLMs sometimes emit symbolic thresholds like `"n_categorical > n_numeric"`)
- **10 new tests** (5 for extract_reasoning + verify_reasoning, 5 for B2 verification path)
- **Empirical confirmation of the "faithfulness ≠ correctness" claim** — Titanic seed=202 scored 0.5 with 6/6 verified decisions

---

## 12. What's not done yet

- **Frontend** — deliberately deferred; backend now has stable schema + all endpoints a UI would need. Suggested stack: React or Next.js pointing at `http://localhost:8000`.
- **Ames House Prices** — third dataset for Wilcoxon significance (currently n_datasets=2 gives p_value=1.0 always)
- **Ablation module** — for each claimed important decision in a trace, run a counterfactual pipeline without that decision. Measures per-decision *causal* impact on score. Very valuable for the paper but expensive (N× more runs per cell).
- **Reasoning trace for B1** (control condition) — currently only B2 must emit a trace. If B1 also produced one, we could compare faithfulness across conditions.
- **CI on GitHub Actions** — run pytest across the 6 repos on push
- **Interactive refinement API** — the original vision included `POST /runs/{id}/refine` with a user prompt to iteratively improve a pipeline. Not implemented.

---

## 13. User preferences to remember

- **Polyrepo, not monorepo.** Each service = separate GitHub repo. Reference model: `velorithm/*` at their company.
- **Cloud LLM models only** (`gpt-oss:120b-cloud`, `ministral-3:14b-cloud`) — not local ones like `qwen2.5:3b`
- **Postman/Insomnia** for API testing (importable via `/openapi.json` per service)
- **Public repos** for the org
- **Terse Claude responses** — don't restate context; give recommendations, not exhaustive surveys
- **Ask only when a decision is theirs.** Implementation details Claude decides.
- **After each change, run tests + verify one thing works end-to-end.** Sweep bugs get caught early.
- **Docker for reproducibility** — teammate should be able to run `docker compose up` and be productive in 5 min
- **Commit + push often.** Every meaningful change lands on GitHub the same day.

---

## 14. Where the paper story is right now

Strong preliminary result to build a paper around:

> On tabular classification datasets containing near-unique identifier columns (Titanic's `Name`, Telco's `customerID`), meta-feature-guided prompting (B2) produces pipelines that follow the LLM's stated rules faithfully (mechanical verification: 28/28 decisions verified across 10 runs) yet collapse to majority-class prediction with balanced_accuracy = 0.500. The failure is caused by a specific rule co-firing: Rule 3 (TargetEncoder for high cardinality) combined with Rule 4 (gradient boosting for many categoricals) creates a target-encoding leakage pattern that memorizes training identifiers and generalizes to constant predictions on test. Naive prompting (B0) and schema-only prompting (B1) — which have no rule scaffolding — score 0.75+ on the same seeds by defaulting to OneHotEncoder + Logistic/RandomForest, which is L2-regularized enough to gracefully degrade on unseen categories. This disentangles *faithfulness* (does the code match the LLM's cited reasoning?) from *correctness* (does the pipeline generalize?), and demonstrates that meta-feature guidance is not universally beneficial — the rule base itself can encode adversarial interactions that the LLM will faithfully execute.

Publishable finding on 2 datasets so far. Adding Ames as a 3rd would tighten Wilcoxon significance.

---

*End of context. This file is the source of truth — if you're a new Claude session, you can start work directly from here.*
