"""Diagnostic dry-run script for the AutoML pipeline.

Loads one dataset, builds the prompt for the chosen condition, optionally
prints it and exits (``--prompt-only``), otherwise iterates up to
``--max-iter`` times: calls Ollama, runs the generated code in a subprocess,
captures stdout/stderr/returncode, and reports the outcome.

This script is for human diagnosis and prompt tuning, not for the formal
experiment sweep (which lives in ``src.experiments.runner``).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.conditions.b0_naive import B0Naive
from src.conditions.b1_schema import B1Schema
from src.conditions.b2_metafeature import B2MetaFeatureGuided
from src.conditions.base import PromptCondition
from src.contracts import ErrorCategory, TaskType
from src.execution.error_taxonomy import classify_error
from src.execution.metrics import extract_score
from src.experiments.datasets import build_task_description, load_custom_dataset, load_dataset
from src.experiments.runner import build_error_feedback, call_llm
from src.meta_features.extractor import extract as extract_meta

_CONDITION_MAP: dict[str, type[PromptCondition]] = {
    "b0_naive": B0Naive,
    "b1_schema": B1Schema,
    "b2_metafeature": B2MetaFeatureGuided,
}

_LOGS_DIR = _REPO_ROOT / "logs" / "dry_run"


def _tail(text: str, n: int = 20) -> str:
    if not text:
        return "(empty)"
    lines = text.splitlines()
    if len(lines) <= n:
        return text
    return "\n".join(lines[-n:])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run a single (dataset, condition, model) combination for diagnosis."
    )
    parser.add_argument(
        "--dataset-id", type=int, default=37, help="OpenML dataset id (default: 37, diabetes)."
    )
    parser.add_argument(
        "--csv-path", type=str, default=None,
        help="Path to a custom CSV file (overrides --dataset-id).",
    )
    parser.add_argument(
        "--target-col", type=str, default=None,
        help="Target column name (required when using --csv-path).",
    )
    parser.add_argument(
        "--task-type", type=str, default=None,
        choices=["binary_classification", "multiclass_classification", "regression"],
        help="Task type (auto-detected if omitted).",
    )
    parser.add_argument("--model", type=str, default="llama3.3:70b", help="Ollama model tag.")
    parser.add_argument(
        "--condition",
        type=str,
        choices=list(_CONDITION_MAP.keys()),
        default="b2_metafeature",
        help="Which prompt condition to use.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=3, help="Max LLM refinement iterations.")
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Print the prompt and exit without calling the LLM.",
    )
    return parser.parse_args()


def _print_meta_summary(meta) -> None:
    print("Meta-features summary:")
    print(f"  n_rows                  = {meta.simple.n_rows}")
    print(f"  n_cols                  = {meta.simple.n_cols}")
    print(f"  n_numeric / n_categorical = {meta.simple.n_numeric} / {meta.simple.n_categorical}")
    print(f"  missing_ratio_overall   = {meta.simple.missing_ratio_overall:.4f}")
    cbr = meta.simple.class_balance_ratio
    print(f"  class_balance_ratio     = {cbr if cbr is None else f'{cbr:.4f}'}")
    skew_vals = list(meta.distributional.skewness_per_numeric.values())
    out_vals = list(meta.distributional.outlier_ratio_per_numeric.values())
    max_skew = max((abs(v) for v in skew_vals), default=0.0)
    max_out = max(out_vals, default=0.0)
    print(f"  max |skewness|          = {max_skew:.3f}")
    print(f"  max outlier_ratio       = {max_out:.3f}")
    print(f"  max_pairwise_correlation = {meta.information.max_pairwise_correlation:.3f}")
    print(f"  decision_stump_score    = {meta.landmarks.decision_stump_score:.3f}")
    print(f"  naive_bayes_score       = {meta.landmarks.naive_bayes_score:.3f}")
    print(f"  one_nn_score            = {meta.landmarks.one_nn_score:.3f}")


def _run_code_subprocess(code: str, prefix: str, timeout: int = 600) -> tuple[int, str, str, str, float]:
    """Persist ``code`` to a file under logs/dry_run/ and run it in a subprocess.

    Returns ``(returncode, stdout, stderr, code_path, runtime_seconds)``.
    Mirrors the subprocess block of ``src.execution.runner.execute_pipeline`` but
    keeps stdout/stderr so the diagnostic can display them.
    """
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    safe_prefix = "".join(c if c.isalnum() or c in "-_" else "_" for c in prefix)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix=safe_prefix + "_",
        delete=False,
        dir=str(_LOGS_DIR),
        encoding="utf-8",
    ) as fh:
        fh.write(code)
        code_path = fh.name

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_LOGS_DIR),
        )
        returncode = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        returncode = -15
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr = (stderr + f"\n[TimeoutExpired after {timeout}s]").lstrip()
    runtime = time.monotonic() - start
    return returncode, stdout, stderr, code_path, runtime


def main() -> int:
    args = _parse_args()

    if args.csv_path:
        if not args.target_col:
            print("ERROR: --target-col is required when using --csv-path")
            return 2
        dataset_info = load_custom_dataset(
            args.csv_path, args.target_col, task_type=args.task_type, seed=args.seed,
        )
    else:
        dataset_info = load_dataset(args.dataset_id, seed=args.seed)
    df_train = dataset_info["df_train"]
    df_test = dataset_info["df_test"]
    task_type: TaskType = dataset_info["task_type"]
    target_col: str = dataset_info["target_col"]
    dataset_name: str = dataset_info["dataset_name"]

    if args.csv_path:
        print(f"Dataset:     {dataset_name} (custom CSV: {args.csv_path})")
    else:
        print(f"Dataset:     {dataset_name} (OpenML #{args.dataset_id})")
    print(f"Train shape: {df_train.shape}")
    print(f"Test shape:  {df_test.shape}")
    print(f"Target col:  {target_col}")
    print(f"Task type:   {task_type.value}")
    print()

    meta = extract_meta(
        df_train,
        target_col=target_col,
        dataset_id=dataset_name,
        task_type=task_type,
    )
    _print_meta_summary(meta)
    print()

    condition = _CONDITION_MAP[args.condition]()
    task_description = build_task_description(dataset_info)
    prompt = condition.build_prompt(task_description, meta, df_train.head(3), target_col)

    print(f"Condition:     {condition.condition_name} ({args.condition})")
    print(f"Prompt length: {len(prompt)} chars")
    print()
    print("=" * 70)
    print("PROMPT")
    print("=" * 70)
    print(prompt)
    print("=" * 70)
    print()

    if args.prompt_only:
        return 0

    base_prompt = prompt
    current_prompt = prompt
    best_score: float | None = None
    final_error: str | None = None
    iterations_used = 0

    for i in range(args.max_iter):
        iterations_used = i + 1
        print(f"\n--- Iteration {iterations_used}/{args.max_iter} ---")

        try:
            code = call_llm(current_prompt, args.model, args.seed + i)
        except ConnectionError as exc:
            print("Ollama not reachable at localhost:11434. Start it with: ollama serve")
            print(f"Details: {exc}")
            return 2

        print(f"Generated code ({len(code)} chars):")
        print("-" * 70)
        print(code)
        print("-" * 70)

        run_prefix = f"{dataset_name}_{condition.condition_name}_seed{args.seed + i}_iter{i}"
        returncode, stdout, stderr, code_path, runtime = _run_code_subprocess(
            code, prefix=run_prefix
        )
        print(f"Return code: {returncode}    (runtime: {runtime:.1f}s)")
        print(f"Code saved at: {code_path}")
        print("--- stdout (last 20 lines) ---")
        print(_tail(stdout, 20))
        print("--- stderr (last 20 lines) ---")
        print(_tail(stderr, 20))
        print("------------------------------")

        score = extract_score(stdout, task_type)
        if score is not None:
            print(f"OK Score: {score}")
            best_score = score
            break

        if returncode != 0:
            category, message = classify_error(stderr, returncode)
            error_category = category.value
            final_error = f"{error_category}: {message}"
            print(f"FAIL {error_category}: {message}")
        else:
            error_category = ErrorCategory.RUNTIME_OTHER.value
            final_error = "runtime_other: pipeline ran but did not print a valid SCORE line"
            print(f"FAIL {final_error}")

        current_prompt = base_prompt + build_error_feedback(
            error_message=final_error + "\n\nLast lines of stderr:\n" + _tail(stderr, 10),
            error_category=error_category,
            prev_code=code,
        )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if args.csv_path:
        print(f"Dataset:          {dataset_name} (custom CSV)")
    else:
        print(f"Dataset:          {dataset_name} (OpenML #{args.dataset_id})")
    print(f"Condition:        {condition.condition_name} ({args.condition})")
    print(f"Model:            {args.model}")
    print(f"Best score:       {best_score if best_score is not None else 'FAILED'}")
    print(f"Iterations used:  {iterations_used}/{args.max_iter}")
    print(f"Final error:      {final_error or 'None'}")
    return 0 if best_score is not None else 1


if __name__ == "__main__":
    sys.exit(main())
