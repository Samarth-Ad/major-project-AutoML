# Agentic AutoML Pipeline Builder

An adaptive, LLM-driven AutoML system that reads a CSV dataset, decides the preprocessing and modeling flow, executes it step-by-step, and produces reproducible artifacts.

The pipeline can run through:
- a **CLI** entrypoint (`main.py`)
- a **Streamlit UI** (`app.py`)

It supports both:
- **Ollama** (local LLM; default)
- **Anthropic** (API key based)

## What This Project Generates

After a successful run, the project produces:
- `outputs/cleaned_data.csv` - final processed dataset
- `outputs/model.pkl` - selected trained model (joblib)
- `outputs/metrics.json` - evaluation metrics (if available)
- `outputs/model_comparison.csv` - model comparison table (if available)
- `outputs/pipeline.ipynb` - auto-generated notebook from pipeline script
- `outputs/report.md` - generated documentation report
- `generated_code/pipeline_script.py` - reproducible Python script of pipeline steps
- `logs/pipeline.log` - full runtime logs
- `memory_store/*.jsonl` - pipeline and step execution history

## Core Features

- Dynamic step planning via `DataUnderstandingAgent`
- Adaptive execution orchestrated by `MasterAgent`
- Retry + backoff execution through `Scheduler`
- Generated code observation through `CodeWriterAgent`
- Two-layer memory (in-memory + persistent jsonl logs)
- Optional incremental re-execution support (`orchestrator/incremental.py`)
- LangChain-compatible wrappers (`agents/langchain_wrapper.py`)

## Project Structure

```text
Data Processing/
├─ main.py
├─ app.py
├─ requirements.txt
├─ config/
│  └─ pipeline.yaml
├─ agents/
├─ orchestrator/
├─ execution/
├─ observer/
├─ memory/
├─ builder/
├─ utils/
├─ data/
├─ outputs/
├─ generated_code/
├─ logs/
└─ memory_store/
```

## Requirements

- Python 3.11+
- pip
- (Default backend) Ollama installed and running

Main Python dependencies are listed in `requirements.txt` (pandas, numpy, scikit-learn, xgboost, lightgbm, optuna, shap, streamlit, etc.).

## Quick Start (Windows / PowerShell)

Run these commands from `Data Processing` directory:

```powershell
# 1) Create and activate virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Start Ollama server (in another terminal)
ollama serve

# 4) Pull default model (one-time)
ollama pull gpt-oss:120b-cloud

# 5) Run pipeline
python main.py --data .\data\train.csv
```

## CLI Usage

### Basic Run

```powershell
python main.py --data .\data\train.csv
```

### Useful Variants

```powershell
# Validate config/dependencies without full execution
python main.py --dry-run

# Override Ollama model
python main.py --data .\data\train.csv --ollama-model gpt-oss:120b-cloud

# Use specific backend explicitly
python main.py --data .\data\train.csv --backend ollama

# Use manual step list (legacy/bypass adaptive planning)
python main.py --steps load_dataset remove_missing_values train_model

# Interactive refinement loop
python main.py --data .\data\train.csv --interactive
```

### Full CLI Arguments (from `main.py`)

- `--config` path to YAML config (default: `config/pipeline.yaml`)
- `--steps` list of manual step names
- `--data` input CSV path
- `--prompt` user prompt/objective
- `--interactive` iterative prompt loop after run
- `--backend` `ollama` or `anthropic`
- `--ollama-model` model name override
- `--ollama-url` Ollama base URL override
- `--api-key` Anthropic API key
- `--max-retries` max retries per step
- `--no-abort` continue pipeline after failures
- `--dry-run` validate setup only

## Streamlit UI Usage

Start UI:

```powershell
streamlit run app.py
```

Then:
- Choose backend/model from sidebar
- Upload a CSV file
- Enter pipeline objective prompt
- Click **Run Pipeline**

UI tabs expose metrics, cleaned data, markdown report, and filtered logs.

## Configuration

Main config file: `config/pipeline.yaml`

Key sections:
- `llm`: backend, model names, base URL, retry behavior
- `data`: optional default filepath and target column
- `thresholds`: adaptive trigger values (nulls, imbalance, skew, PCA, tuning)

Example settings in current project:
- default backend: `ollama`
- default model: `gpt-oss:120b-cloud`
- default Ollama URL: `http://localhost:11434`

## Environment Variables

Supported environment variables include:
- `LLM_BACKEND` (`ollama` or `anthropic`)
- `OLLAMA_MODEL`
- `OLLAMA_BASE_URL`
- `ANTHROPIC_MODEL`
- `ANTHROPIC_API_KEY`

If present, `.env` is auto-loaded in `main.py`.

## How the Pipeline Works

1. `load_dataset` reads input data.
2. `DataUnderstandingAgent` profiles data and decides steps/models.
3. `MasterAgent` rebuilds adaptive step list.
4. `Scheduler` runs each step agent with retry logic.
5. `CodeWriterAgent` appends executable step code into `generated_code/pipeline_script.py`.
6. `main.py` saves cleaned data/model/metrics and generates notebook/report.

## Typical Output Files

- `outputs/cleaned_data.csv` - transformed dataset
- `outputs/model.pkl` - best model
- `outputs/metrics.json` - model metrics
- `outputs/model_comparison.csv` - candidate model comparison
- `outputs/pipeline.ipynb` - execution notebook
- `outputs/report.md` - generated run report

## Troubleshooting

- **No tests run with pytest**: this repo currently has no discovered pytest tests.
- **Ollama not reachable**:
  - Start service: `ollama serve`
  - Verify model: `ollama pull gpt-oss:120b-cloud`
- **Streamlit warnings when using plain `python app.py`**:
  - Use `streamlit run app.py` instead.
- **Missing packages**:
  - Reinstall: `pip install -r requirements.txt`

## Development Notes

- Generated/runtime artifacts are intentionally ignored by `.gitignore` (`outputs/`, `generated_code/`, `logs/`, `memory_store/`, etc.).
- Core orchestrator and agent logic lives in:
  - `orchestrator/master_agent.py`
  - `agents/base_agent.py`
  - `agents/data_understanding_agent.py`
  - `execution/scheduler.py`

## Minimal Command Checklist

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# LLM backend
ollama serve
ollama pull gpt-oss:120b-cloud

# Run
python main.py --data .\data\train.csv

# UI
streamlit run app.py
```

