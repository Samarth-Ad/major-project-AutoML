r"""
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
    Handles: single model, multiple models dict, ExecutionResult wrapper.
    Returns dict of saved file paths.
    """
    import pandas as pd

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}

    final     = result.final_data
    model_obj = None
    df_final  = None

    # ── Unwrap output types ───────────────────────────────────────────
    if hasattr(final, "df") and hasattr(final, "model"):
        # _ExecutionResult from train_model step
        df_final  = final.df
        model_obj = final.model

    elif isinstance(final, dict):
        # Could be a multi-model result or metrics dict
        if "best_model" in final:
            model_obj = final["best_model"]
            df_final  = final.get("df", None)
        elif "metrics" in final:
            # Single model result with metrics inline
            model_obj = final.get("model", None)
            df_final  = final.get("df", None)

    elif isinstance(final, pd.DataFrame):
        df_final = final

    # ── 1. Save cleaned DataFrame ─────────────────────────────────────
    if df_final is not None and isinstance(df_final, pd.DataFrame):
        csv_path = OUTPUTS_DIR / "cleaned_data.csv"
        df_final.to_csv(csv_path, index=False, encoding="utf-8")
        saved["processed_csv"] = str(csv_path)
        _logger.info(
            f"Cleaned dataset saved -> {csv_path} "
            f"({df_final.shape[0]} rows x {df_final.shape[1]} cols)"
        )

    # ── 2. Save best model ────────────────────────────────────────────
    if model_obj is not None:
        try:
            import joblib
            model_path = OUTPUTS_DIR / "model.pkl"
            joblib.dump(model_obj, model_path)
            saved["trained_model"] = str(model_path)
            _logger.info(
                f"Model saved -> {model_path} "
                f"(type: {type(model_obj).__name__})"
            )
        except Exception as exc:
            _logger.warning(f"Could not save model: {exc}")

    # ── 3. Copy metrics if available ──────────────────────────────────
    metrics_src = Path("models") / "metrics.json"
    if metrics_src.exists():
        import shutil
        metrics_dst = OUTPUTS_DIR / "metrics.json"
        shutil.copy(metrics_src, metrics_dst)
        saved["metrics"] = str(metrics_dst)

    # ── 4. Save comparison table if available ─────────────────────────
    comparison_src = Path("models") / "comparison.csv"
    if comparison_src.exists():
        import shutil
        comparison_dst = OUTPUTS_DIR / "model_comparison.csv"
        shutil.copy(comparison_src, comparison_dst)
        saved["comparison"] = str(comparison_dst)
        _logger.info(f"Model comparison table saved -> {comparison_dst}")

    return saved


# ---------------------------------------------------------------------------
# Generate Jupyter notebook
# ---------------------------------------------------------------------------

def _generate_notebook(result, saved_paths: dict) -> str:
    """
    Convert the generated pipeline_script.py into a clean Jupyter notebook.

    Uses safe string-based splitting (no regex backtracking) to parse
    the generated script into individual cells.

    Returns path to the saved .ipynb file.
    """
    script_path = Path(result.script_path)
    if not script_path.exists():
        _logger.warning("pipeline_script.py not found — skipping notebook generation")
        return ""

    try:
        source = script_path.read_text(encoding="utf-8")
    except Exception as exc:
        _logger.warning(f"Could not read pipeline script: {exc}")
        return ""

    # ── Helpers ────────────────────────────────────────────────────────
    def make_code_cell(source_lines: str) -> dict:
        lines = [l + "\n" for l in source_lines.rstrip("\n").split("\n")]
        return {
            "cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": lines,
        }

    def make_md_cell(text: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": [text]}

    cells = []

    # Cell 1 — title
    cells.append(make_md_cell(
        f"# Agentic Pipeline — Auto-Generated Notebook\n"
        f"**Pipeline ID:** `{result.pipeline_id}`  \n"
        f"**Status:** {'SUCCESS' if result.success else 'FAILED'}  \n"
        f"**Total time:** {result.total_elapsed_s:.2f}s\n"
    ))

    # ── Safe section splitting (no regex) ─────────────────────────────
    # Split by the heavy-bar separator lines: ══════════ or ──────────
    lines = source.split("\n")
    sections = []          # list of (section_type, section_lines)
    current_lines = []
    current_type  = "preamble"

    for line in lines:
        stripped = line.strip()
        # Detect step section headers: lines starting with '# STEP N:'
        if stripped.startswith("# STEP ") and ":" in stripped:
            # Save accumulated section
            if current_lines:
                sections.append((current_type, current_lines))
            current_lines = [line]
            current_type  = "step_header"
            continue

        # Detect separator lines (═ or ─ decorators)
        if stripped and all(c in "#═─ " for c in stripped) and len(stripped) > 10:
            if current_type == "step_header":
                current_lines.append(line)
            else:
                if current_lines:
                    sections.append((current_type, current_lines))
                current_lines = [line]
                current_type  = "separator"
            continue

        current_lines.append(line)
        if current_type == "separator":
            current_type = "code"
        elif current_type == "step_header":
            current_type = "step_comments"

    if current_lines:
        sections.append((current_type, current_lines))

    # ── Parse sections into notebook cells ────────────────────────────
    # Extract imports from preamble
    preamble_code = []
    for sec_type, sec_lines in sections:
        if sec_type == "preamble":
            for ln in sec_lines:
                s = ln.strip()
                if s.startswith("import ") or s.startswith("from "):
                    preamble_code.append(ln)
                elif s.startswith("_PIPELINE_ID"):
                    preamble_code.append(ln)
            break

    if preamble_code:
        cells.append(make_md_cell("## Imports"))
        cells.append(make_code_cell("\n".join(preamble_code)))

    # Process step blocks: find "# STEP N: NAME" headers and collect
    # the comment block (reasoning) and the code that follows
    i = 0
    while i < len(sections):
        sec_type, sec_lines = sections[i]
        # Look for step header lines
        step_text = "\n".join(sec_lines)
        if "# STEP " in step_text and ":" in step_text:
            # Parse step number and name
            for ln in sec_lines:
                if ln.strip().startswith("# STEP "):
                    parts = ln.strip().lstrip("# ").split(":", 1)
                    step_num = parts[0].replace("STEP ", "").strip()
                    step_name = parts[1].strip() if len(parts) > 1 else f"Step {step_num}"
                    break
            else:
                step_num, step_name = "?", "Unknown"

            # Collect reasoning from comment lines
            reasoning_lines = []
            for ln in sec_lines:
                s = ln.strip()
                if s.startswith("# ") and not s.startswith("# STEP") and not all(c in "#═─ " for c in s):
                    reasoning_lines.append(s.lstrip("# ").strip())

            # Find the code section that follows
            code_lines = []
            j = i + 1
            while j < len(sections):
                next_type, next_lines = sections[j]
                next_text = "\n".join(next_lines)
                if "# STEP " in next_text or "PIPELINE EXECUTION" in next_text:
                    break
                # Collect non-comment, non-separator code
                for ln in next_lines:
                    s = ln.strip()
                    if s and not all(c in "#═─ " for c in s):
                        code_lines.append(ln)
                j += 1

            # Only add cells if there's actual code
            reasoning_text = " ".join(reasoning_lines[:5]) if reasoning_lines else ""
            display_name = step_name.replace("_", " ").title()
            md = f"## Step {step_num}: {display_name}\n"
            if reasoning_text:
                md += f"\n**LLM Reasoning:** {reasoning_text}\n"
            cells.append(make_md_cell(md))

            clean_code = "\n".join(code_lines).strip()
            if clean_code:
                cells.append(make_code_cell(clean_code))

            i = j
            continue
        i += 1

    # Final cell — load saved outputs
    load_code = "# ── Load pipeline outputs ────────────────────────────\n"
    load_code += "import pandas as pd\n"
    if "processed_csv" in saved_paths:
        rel = Path(saved_paths["processed_csv"]).name
        load_code += (
            f"\n# Cleaned dataset\n"
            f"df_clean = pd.read_csv('outputs/{rel}', encoding='utf-8')\n"
            f"print(f'Cleaned data: {{df_clean.shape[0]}} rows x {{df_clean.shape[1]}} cols')\n"
            f"df_clean.head()\n"
        )
    if "trained_model" in saved_paths:
        rel = Path(saved_paths["trained_model"]).name
        load_code += (
            f"\n# Trained model\nimport joblib\n"
            f"model = joblib.load('outputs/{rel}')\n"
            f"print(f'Model type: {{type(model).__name__}}')\n"
        )
    cells.append(make_md_cell("## Saved Outputs"))
    cells.append(make_code_cell(load_code))

    # ── Build notebook JSON ───────────────────────────────────────────
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    nb_path = OUTPUTS_DIR / "pipeline.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
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

    for outcome in result.report.outcomes:
        icon = "[OK]  " if outcome.succeeded else "[FAIL]"
        skip = " [conditional skip]" if outcome.status == "skipped" else ""
        print(f"  {icon} {outcome.step_index}. {outcome.step_name:<35} {outcome.elapsed_s:.2f}s{skip}")

    print()
    print("  SAVED FILES:")

    if "processed_csv" in saved_paths:
        p  = Path(saved_paths["processed_csv"])
        df = pd.read_csv(p, encoding="utf-8")
        print(f"  [CSV]      {p}")
        print(f"             {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"             Columns: {list(df.columns)}")

    if "trained_model" in saved_paths:
        print(f"  [MODEL]    {saved_paths['trained_model']}")

    if "metrics" in saved_paths:
        with open(saved_paths["metrics"]) as f:
            m = json.load(f)
        print(f"  [METRICS]  {saved_paths['metrics']}")
        for k, v in m.items():
            if isinstance(v, (int, float)):
                print(f"             {k:<18}: {v}")

    if "comparison" in saved_paths:
        comp = pd.read_csv(saved_paths["comparison"])
        print(f"  [COMPARISON] {saved_paths['comparison']}")
        print(comp.to_string(index=False))

    if nb_path:
        print(f"  [NOTEBOOK] {nb_path}")
        print(f"             Open: jupyter notebook {nb_path}")

    if "report" in saved_paths:
        print(f"  [REPORT]   {saved_paths['report']}")

    print(f"  [SCRIPT]   {result.script_path}")
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
    config_path    = Path(args.config)

    # Read target column from YAML if not on CLI
    target_column = ""
    if config_path.exists():
        import yaml as _yaml
        try:
            cfg = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            target_column = cfg.get("data", {}).get("target_column", "") or ""
        except Exception:
            pass

    master = MasterAgent(
        api_key          = api_key,
        llm_model        = resolved_model,
        max_retries      = args.max_retries,
        abort_on_failure = not args.no_abort,
        config_path      = str(config_path),
    )

    if args.dry_run:
        res = master.dry_run(config_path)
        return 0 if res["valid"] else 1

    initial_data = None
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            _logger.error(f"Data file not found: {data_path}")
            return 1
        initial_data = str(data_path)

    # pipeline_source: steps list (legacy) or config path (adaptive)
    if args.steps:
        config_path = args.steps   # pass list directly → legacy mode

    # Execute pipeline
    try:
        result = master.run(
            pipeline_config = config_path if not args.steps else args.steps,
            initial_data    = initial_data,
            target_column   = target_column,
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
    try:
        nb_path = _generate_notebook(result, saved_paths)
    except Exception as exc:
        _logger.warning(f"Notebook generation failed: {exc}")
        nb_path = ""

    # ── Generate documentation report ─────────────────────────────────
    try:
        from agents.documentation_agent import DocumentationAgent
        doc_agent = DocumentationAgent()
        report_path = doc_agent.generate_report(result, saved_paths)
        saved_paths["report"] = report_path
    except Exception as exc:
        _logger.warning(f"Report generation failed: {exc}")

    # ── Print plain summary ───────────────────────────────────────────
    _print_summary(result, saved_paths, nb_path)

    return 0 if result.success else 1


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()