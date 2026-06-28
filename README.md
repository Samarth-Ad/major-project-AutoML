# Meta-Feature-Guided Prompting for LLM-Driven Tabular AutoML

## Abstract

This project investigates whether injecting computed dataset meta-features (distributional statistics, information-theoretic measures, landmarking scores) into LLM prompts improves the quality and reliability of auto-generated scikit-learn pipelines for tabular classification. We compare three prompting conditions — naive (B0), schema-only (B1), and meta-feature-guided (B2) — across 28 OpenML CC18 datasets and three LLM backends, measuring predictive performance, code-generation failure rate, and iteration efficiency.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) with cloud inference access (for LLM calls)
- ~2 GB free disk space for cached datasets

## Setup

```bash
git clone https://github.com/Samarth-Ad/major-project-AutoML.git
cd major-project-AutoML

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -e .
```

## Download Datasets

```bash
python scripts/run_sweep.py --precache
```

Downloads and caches all 28 OpenML datasets locally under `data/experiments/`.

## Run Tests

```bash
python -m pytest tests/ -v
```

27 unit tests should pass.

## Dry Run (single dataset, single condition)

```bash
python scripts/dry_run.py --dataset-id 37 --condition b2_metafeature --model "qwen3-coder:480b-cloud"
```

Use this for prompt tuning and human diagnosis. The script loads one dataset, builds the prompt for the chosen condition, calls Ollama, runs the generated pipeline in a subprocess, and reports the outcome with stdout/stderr tails.

## Full Sweep

```bash
python scripts/run_sweep.py \
  --models "qwen3-coder:480b-cloud,gpt-oss:120b-cloud,ministral-3:14b-cloud" \
  --seeds "42,43,44" \
  --output results/sweep_results.jsonl
```

The sweep is resume-safe — rerunning skips already-completed `(dataset, condition, model, seed)` cells already present in the output JSONL.

## Project Structure

```
Major-Project/
├─ src/
│  ├─ contracts.py            # Shared dataclasses (TaskType, GeneratedPipeline, …)
│  ├─ strategic_lm.py         # LLM call wrappers / strategy helpers
│  ├─ meta_features/          # Dataset meta-feature extraction
│  │   ├─ simple.py           #   shape, dtype counts, class balance
│  │   ├─ distributional.py   #   skewness, kurtosis, moments
│  │   ├─ information.py      #   entropy, mutual information
│  │   ├─ landmarking.py      #   decision-stump score, simple-learner baselines
│  │   └─ extractor.py        #   aggregator / unified extract() API
│  ├─ conditions/             # Prompting conditions
│  │   ├─ base.py             #   PromptCondition interface
│  │   ├─ b0_naive.py         #   B0 — task description only
│  │   ├─ b1_schema.py        #   B1 — task + column schema
│  │   └─ b2_metafeature.py   #   B2 — task + schema + meta-features
│  ├─ execution/              # Subprocess-isolated pipeline execution
│  │   ├─ runner.py           #   execute_pipeline() in a fresh interpreter
│  │   ├─ error_taxonomy.py   #   classify_error() — syntax / import / shape / timeout / …
│  │   └─ metrics.py          #   extract_score() from generated pipeline stdout
│  └─ experiments/            # Orchestration and analysis
│      ├─ datasets.py         #   OpenML loaders + SELECTED_CLASSIFICATION registry
│      ├─ runner.py           #   call_llm() and single-cell experiment runner
│      └─ analysis.py         #   Summary tables, error breakdowns, stats
├─ scripts/
│  ├─ dry_run.py              # Single (dataset, condition, model) diagnostic run
│  └─ run_sweep.py            # Full dataset x condition x model x seed sweep
├─ tests/                     # 27 unit tests
├─ pyproject.toml
└─ README.md
```

## Research Questions

- **RQ1.** Does meta-feature-guided prompting (B2) produce higher predictive performance than schema-only (B1) or naive (B0) prompting across heterogeneous tabular datasets?
- **RQ2.** Does the inclusion of meta-features reduce the rate of code-generation failures (syntax errors, import errors, shape mismatches, timeouts)?
- **RQ3.** Does meta-feature guidance shorten the iterative refinement loop — i.e., produce a working pipeline in fewer LLM calls?
- **RQ4.** How does the effect of meta-feature guidance vary across LLM backends of different sizes and capabilities?
- **RQ5.** Are the gains from meta-feature guidance consistent across small / medium / large datasets, or are they concentrated in a specific size or difficulty regime?

## License

TBD.
