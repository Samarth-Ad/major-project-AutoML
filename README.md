# Meta-Feature-Guided Prompting for LLM-Driven Tabular AutoML

## Abstract

This project investigates whether injecting computed dataset meta-features (distributional statistics, information-theoretic measures, landmarking scores) into LLM prompts improves the quality and reliability of auto-generated scikit-learn pipelines for tabular classification. We compare three prompting conditions — naive (B0), schema-only (B1), and meta-feature-guided (B2) — across 28 OpenML CC18 datasets and three LLM backends, measuring predictive performance, code-generation failure rate, and iteration efficiency.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) with cloud inference access
- ~2 GB free disk space for cached datasets

## Quick Start

**Step 1.** Clone the repo.

```
git clone https://github.com/Samarth-Ad/major-project-AutoML.git
cd major-project-AutoML
```

**Step 2.** Create a virtual environment.

```
python -m venv .venv
```

**Step 3.** Activate it.

On Windows (PowerShell):

```
.venv\Scripts\Activate.ps1
```

On Linux / macOS:

```
source .venv/bin/activate
```

**Step 4.** Install dependencies.

```
pip install --upgrade pip setuptools
```

```
pip install -e .
```

**Step 5.** Download the OpenML CC18 datasets (one-time, ~2 GB).

```
python scripts/run_sweep.py --precache
```

**Step 6.** Verify the install.

```
python -m pytest tests/ -v
```

All 27 tests should pass.

## Usage

### Dry Run

Run a single `(dataset, condition, model)` cell end-to-end. Use this for prompt tuning and diagnosis.

```
python scripts/dry_run.py --dataset-id 37 --condition b2_metafeature --model "ministral-3:14b-cloud"
```

Flags:

- `--dataset-id` — OpenML dataset id (e.g. `37` = diabetes)
- `--condition` — one of `b0_naive`, `b1_schema`, `b2_metafeature`
- `--model` — Ollama model tag
- `--prompt-only` — print the assembled prompt and exit without calling the LLM

### Full Sweep

Run the full experiment across all datasets × conditions × models × seeds.

```
python scripts/run_sweep.py --models "ministral-3:14b-cloud" --seeds "42,43,44" --output results/sweep_results.jsonl
```

The sweep is **resume-safe** — rerunning skips `(dataset, condition, model, seed)` cells already present in the output JSONL.

## Project Structure

```
src/
├─ meta_features/   Dataset meta-feature extraction (simple, distributional, information, landmarking)
├─ conditions/      Prompting conditions (B0 naive, B1 schema, B2 meta-feature-guided)
├─ execution/       Subprocess-isolated pipeline execution and error taxonomy
└─ experiments/     Dataset loading, LLM orchestration, statistical analysis

scripts/             CLI entry points (dry_run, run_sweep)
tests/               Unit tests (27 total)
```

## Research Questions

- **RQ1.** Does meta-feature-guided prompting (B2) produce higher predictive performance than schema-only (B1) or naive (B0) prompting across heterogeneous tabular datasets?
- **RQ2.** Does the inclusion of meta-features reduce the rate of code-generation failures (syntax errors, import errors, shape mismatches, timeouts)?
- **RQ3.** Does meta-feature guidance shorten the iterative refinement loop — i.e., produce a working pipeline in fewer LLM calls?
- **RQ4.** How does the effect of meta-feature guidance vary across LLM backends of different sizes and capabilities?
- **RQ5.** Are the gains from meta-feature guidance consistent across small / medium / large datasets, or are they concentrated in a specific size or difficulty regime?

## License

TBD.
