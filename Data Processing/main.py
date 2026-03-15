"""
main.py
-------
Agentic Pipeline Builder System — Main Entry Point

Usage
-----
    python main.py --data .\data\train.csv
    python main.py --data .\data\train.csv --ollama-model gpt-oss:120b-cloud
    python main.py --dry-run
    python main.py --steps load_dataset remove_missing_values train_model

Outputs (written to disk after every run)
------------------------------------------
    outputs/processed_data.csv   <- fully processed DataFrame
    outputs/trained_model.pkl    <- trained model (joblib)
    outputs/metrics.json         <- accuracy, F1, ROC-AUC etc.
    outputs/pipeline.ipynb       <- Jupyter notebook (clean, no sugar)
    generated_code/pipeline_script.py  <- raw Python script
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from orchestrator.master_agent import MasterAgent
from utils.logger import PipelineLogger

_logger = PipelineLogger("main")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUTS_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Agentic Pipeline Builder",
    )
    parser.add_argument("--config",       type=str, default="config/pipeline.yaml")
    parser.add_argument("--steps",        nargs="+", default=None)
    parser.add_argument("--data",         type=str, default=None)
    parser.add_argument("--backend",      type=str, default=None, choices=["ollama","anthropic"])
    parser.add_argument("--ollama-model", type=str, default=None)
    parser.add_argument("--ollama-url",   type=str, default=None)
    parser.add_argument("--api-key",      type=str, default="")
    parser.add_argument("--max-retries",  type=int, default=3)
    parser.add_argument("--no-abort",     action="store_true", default=False)
    parser.add_argument("--dry-run",      action="store_true", default=False)
    return parser


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------

def _configure_llm_backend(args: argparse.Namespace) -> None:
    import yaml as _yaml

    yaml_llm: dict = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            with config_path.open() as f:
                cfg = _yaml.safe_load(f) or {}
            yaml_llm = cfg.get("llm", {})
        except Exception:
            pass

    backend = (
        args.backend
        or yaml_llm.get("backend", None)
        or os.environ.get("LLM_BACKEND", "ollama")
    )
    os.environ["LLM_BACKEND"] = backend

    if backend == "ollama":
        model = (
            args.ollama_model
            or yaml_llm.get("ollama_model", None)
            or os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
        )
        url = (
            args.ollama_url
            or yaml_llm.get("ollama_base_url", None)
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        os.environ["OLLAMA_MODEL"]    = model
        os.environ["OLLAMA_BASE_URL"] = url
        _logger.info(
            f"LLM Backend : OLLAMA\n"
            f"  Model     : {model}\n"
            f"  Server    : {url}"
        )
    elif backend == "anthropic":
        model = yaml_llm.get("anthropic_model") or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        key   = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        os.environ["ANTHROPIC_MODEL"] = model
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
        _logger.info(f"LLM Backend : ANTHROPIC | model={model}")


def _check_dependencies() -> None:
    required = {"pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn", "yaml": "pyyaml"}
    missing  = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        _logger.error(f"Missing packages: {missing}. Run: pip install {' '.join(missing)}")
        sys.exit(1)


def _check_ollama_running(url: str, model: str) -> None:
    import urllib.request, urllib.error, json as _json
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as resp:
            tags = _json.loads(resp.read().decode("utf-8"))
        available = [m.get("name","") for m in tags.get("models", [])]
        if model in available:
            _logger.info(f"Ollama model '{model}' is available and ready.")
        else:
            _logger.warning(
                f"Model '{model}' not found in Ollama manifest.\n"
                f"  Available: {available}\n"
                f"  Pull with: ollama pull {model}"
            )
    except urllib.error.URLError:
        _logger.warning(
            f"Ollama not reachable at {url}\n"
            f"  Start with: ollama serve\n"
            f"  Pull model: ollama pull {model}"
        )


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def _save_outputs(result) -> dict:
    """
    Save all pipeline artifacts to disk.

    Returns dict of saved file paths.
    """
    import pandas as pd

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}

    final = result.final_data

    # ── Unwrap _ExecutionResult if present ────────────────────────────
    model_obj = None
    df_final  = None

    if hasattr(final, "df") and hasattr(final, "model"):
        # ExecutionResult from train_model step
        df_final  = final.df
        model_obj = final.model
    elif isinstance(final, pd.DataFrame):
        df_final = final
    # If final is a string (failed early — initial_data filepath), skip

    # ── 1. Save processed CSV ─────────────────────────────────────────
    if df_final is not None and isinstance(df_final, pd.DataFrame):
        csv_path = OUTPUTS_DIR / "processed_data.csv"
        df_final.to_csv(csv_path, index=False, encoding="utf-8")
        saved["processed_csv"] = str(csv_path)
        _logger.info(
            f"Processed dataset saved -> {csv_path}  "
            f"({df_final.shape[0]} rows x {df_final.shape[1]} cols)"
        )

    # ── 2. Save trained model ─────────────────────────────────────────
    if model_obj is not None:
        try:
            import joblib
            model_path = OUTPUTS_DIR / "trained_model.pkl"
            joblib.dump(model_obj, model_path)
            saved["trained_model"] = str(model_path)
            _logger.info(
                f"Trained model saved -> {model_path}  "
                f"(type: {type(model_obj).__name__})"
            )
        except Exception as exc:
            _logger.warning(f"Could not save model: {exc}")

    # ── 3. Save metrics if available ─────────────────────────────────
    metrics_src = OUTPUTS_DIR.parent / "models" / "metrics.json"
    if metrics_src.exists():
        import shutil
        metrics_dst = OUTPUTS_DIR / "metrics.json"
        shutil.copy(metrics_src, metrics_dst)
        saved["metrics"] = str(metrics_dst)
        _logger.info(f"Metrics saved -> {metrics_dst}")

    return saved


# ---------------------------------------------------------------------------
# Generate Jupyter notebook
# ---------------------------------------------------------------------------

def _generate_notebook(result, saved_paths: dict) -> str:
    """
    Convert the generated pipeline_script.py into a clean Jupyter notebook.

    Each section (imports, step block) becomes its own cell.
    No decorative output — just the exact code that ran, in cells.

    Returns path to the saved .ipynb file.
    """
    import re

    script_path = Path(result.script_path)
    if not script_path.exists():
        _logger.warning("pipeline_script.py not found — skipping notebook generation")
        return ""

    source = script_path.read_text(encoding="utf-8")

    # ── Split script into notebook cells ──────────────────────────────
    cells = []

    def make_code_cell(source_lines: str) -> dict:
        lines = [l + "\n" for l in source_lines.rstrip("\n").split("\n")]
        return {
            "cell_type":       "code",
            "execution_count": None,
            "metadata":        {},
            "outputs":         [],
            "source":          lines,
        }

    def make_md_cell(text: str) -> dict:
        return {
            "cell_type": "markdown",
            "metadata":  {},
            "source":    [text],
        }

    # Cell 1 — notebook title
    cells.append(make_md_cell(
        f"# Agentic Pipeline — Auto-Generated Notebook\n"
        f"**Pipeline ID:** `{result.pipeline_id}`  \n"
        f"**Status:** {'SUCCESS' if result.success else 'FAILED'}  \n"
        f"**Total time:** {result.total_elapsed_s:.2f}s\n"
    ))

    # Cell 2 — imports block
    import_match = re.search(
        r"# ── IMPORTS ─+\n(.*?)(?=\n_PIPELINE_ID)",
        source, re.DOTALL
    )
    if import_match:
        imports_code = import_match.group(1).strip()
        # Add _PIPELINE_ID line
        pipeline_id_match = re.search(r"_PIPELINE_ID = '.*?'", source)
        if pipeline_id_match:
            imports_code += "\n" + pipeline_id_match.group(0)
        cells.append(make_md_cell("## Imports"))
        cells.append(make_code_cell(imports_code))

    # Cells 3–N — one cell per pipeline step
    step_pattern = re.compile(
        r"# (═+)\n# STEP (\d+): (.+?)\n# ═+\n"
        r"(?:# .*?\n)*?"              # comment lines (agent, status, etc.)
        r"# LLM Reasoning:\n"
        r"((?:#.*?\n)*)"              # reasoning lines
        r"# ═+\n\n"
        r"(.*?)(?=\n\n# [═─]|\Z)",   # actual code
        re.DOTALL
    )

    for match in step_pattern.finditer(source):
        step_num  = match.group(2)
        step_name = match.group(3).replace("_", " ").title()
        reasoning = match.group(4).replace("#   ", "").replace("# ", "").strip()
        code_body = match.group(5).strip()

        # Markdown cell with step name + reasoning
        cells.append(make_md_cell(
            f"## Step {step_num}: {step_name}\n\n"
            f"**LLM Reasoning:** {reasoning}\n"
        ))

        if code_body:
            cells.append(make_code_cell(code_body))

    # Final cell — load saved outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    load_outputs_code = "# ── Load pipeline outputs ────────────────────────────\n"
    load_outputs_code += "import pandas as pd\n"

    if "processed_csv" in saved_paths:
        rel = Path(saved_paths["processed_csv"]).name
        load_outputs_code += (
            f"\n# Processed dataset\n"
            f"df_processed = pd.read_csv('outputs/{rel}', encoding='utf-8')\n"
            f"print(f'Processed data: {{df_processed.shape[0]}} rows x {{df_processed.shape[1]}} cols')\n"
            f"df_processed.head()\n"
        )

    if "trained_model" in saved_paths:
        rel = Path(saved_paths["trained_model"]).name
        load_outputs_code += (
            f"\n# Trained model\n"
            f"import joblib\n"
            f"model = joblib.load('outputs/{rel}')\n"
            f"print(f'Model type: {{type(model).__name__}}')\n"
            f"print(f'Model params: {{model.get_params()}}')\n"
        )

    if "metrics" in saved_paths:
        load_outputs_code += (
            f"\n# Model metrics\n"
            f"import json\n"
            f"metrics = json.load(open('outputs/metrics.json'))\n"
            f"print('Metrics:', metrics)\n"
        )

    cells.append(make_md_cell("## Saved Outputs"))
    cells.append(make_code_cell(load_outputs_code))

    # ── Build notebook JSON ───────────────────────────────────────────
    notebook = {
        "nbformat":       4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language":     "python",
                "name":         "python3",
            },
            "language_info": {
                "name":    "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }

    nb_path = OUTPUTS_DIR / "pipeline.ipynb"
    nb_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8"
    )
    _logger.info(f"Jupyter notebook saved -> {nb_path}")
    return str(nb_path)


# ---------------------------------------------------------------------------
# Post-run summary — plain, no sugar
# ---------------------------------------------------------------------------

def _print_summary(result, saved_paths: dict, nb_path: str) -> None:
    import pandas as pd

    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Status       : {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Pipeline ID  : {result.pipeline_id}")
    print(f"  Total time   : {result.total_elapsed_s:.2f}s")
    print(f"  Steps        : {result.report.successful_steps}/{result.step_count} succeeded")
    print()

    # Step breakdown
    for outcome in result.report.outcomes:
        icon = "[OK]  " if outcome.succeeded else "[FAIL]"
        print(f"  {icon} {outcome.step_index}. {outcome.step_name:<35} {outcome.elapsed_s:.2f}s")

    print()
    print("  SAVED FILES:")

    if "processed_csv" in saved_paths:
        p = Path(saved_paths["processed_csv"])
        df = pd.read_csv(p, encoding="utf-8")
        print(f"  [CSV]   {p}  ({df.shape[0]} rows x {df.shape[1]} cols)")
        print(f"          Columns: {list(df.columns)}")

    if "trained_model" in saved_paths:
        print(f"  [MODEL] {saved_paths['trained_model']}")

    if "metrics" in saved_paths:
        with open(saved_paths["metrics"]) as f:
            m = json.load(f)
        print(f"  [METRICS] {saved_paths['metrics']}")
        for k, v in m.items():
            print(f"            {k:<15}: {v}")

    if nb_path:
        print(f"  [NOTEBOOK] {nb_path}")
        print(f"             Open with: jupyter notebook {nb_path}")

    print(f"  [SCRIPT] {result.script_path}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> int:
    _check_dependencies()
    _configure_llm_backend(args)

    if os.environ.get("LLM_BACKEND", "ollama") == "ollama":
        _check_ollama_running(
            url   = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        )

    api_key        = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    resolved_model = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")

    master = MasterAgent(
        api_key          = api_key,
        llm_model        = resolved_model,
        max_retries      = args.max_retries,
        abort_on_failure = not args.no_abort,
    )

    if args.steps:
        pipeline_source = args.steps
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            _logger.error(f"Config not found: {config_path}")
            return 1
        pipeline_source = config_path

    if args.dry_run:
        res = master.dry_run(pipeline_source)
        return 0 if res["valid"] else 1

    initial_data = None
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            _logger.error(f"Data file not found: {data_path}")
            return 1
        initial_data = str(data_path)

    try:
        result = master.run(
            pipeline_config = pipeline_source,
            initial_data    = initial_data,
        )
    except ValueError as exc:
        _logger.error(f"Config error: {exc}")
        return 1
    except KeyboardInterrupt:
        _logger.warning("Interrupted by user.")
        return 1
    except Exception as exc:
        _logger.error(f"Unexpected error: {exc}", exc=exc)
        return 1

    # ── Save outputs ──────────────────────────────────────────────────
    saved_paths = _save_outputs(result)

    # ── Generate notebook ─────────────────────────────────────────────
    nb_path = _generate_notebook(result, saved_paths)

    # ── Print plain summary ───────────────────────────────────────────
    _print_summary(result, saved_paths, nb_path)

    return 0 if result.success else 1


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()