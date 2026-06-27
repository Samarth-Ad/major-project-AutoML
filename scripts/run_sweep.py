"""Master orchestration script for the full experiment sweep.

Runs dataset x model x condition x seed (default 28 x 3 x 3 x 3 = 756 runs),
streaming one JSON record per completed cell to ``--output``. Supports resume
(re-running the same command skips cells already present), iterative LLM
refinement (up to ``--max-iter`` attempts per cell, feeding stderr back into
the next prompt), per-run wall time, and graceful handling of Ollama
connection failures.

Intended use::

    # Pre-cache all 28 datasets locally (do this once before the first sweep)
    python scripts/run_sweep.py --precache

    # Smoke test: 9 representative datasets, one model, one seed, B2 only
    python scripts/run_sweep.py \
        --models "qwen3-coder:480b-cloud" \
        --seeds "42" \
        --conditions "b2_metafeature" \
        --dataset-ids "37,40975,38,31,44,1590,1468,1485,182" \
        --output results/smoke_test.jsonl

    # Full sweep
    python scripts/run_sweep.py \
        --models "qwen3-coder:480b-cloud,gpt-oss:120b-cloud,ministral-3:14b-cloud" \
        --seeds "42,43,44" \
        --output results/sweep_results.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.conditions.b0_naive import B0Naive
from src.conditions.b1_schema import B1Schema
from src.conditions.b2_metafeature import B2MetaFeatureGuided
from src.conditions.base import PromptCondition
from src.contracts import GeneratedPipeline
from src.execution.runner import execute_pipeline
from src.experiments.datasets import (
    SELECTED_CLASSIFICATION,
    build_task_description,
    load_dataset,
)
from src.experiments.runner import call_llm
from src.meta_features.extractor import extract as extract_meta

_CONDITION_MAP: dict[str, type[PromptCondition]] = {
    "b0_naive": B0Naive,
    "b1_schema": B1Schema,
    "b2_metafeature": B2MetaFeatureGuided,
}

DEFAULT_MODELS = "qwen3-coder:480b-cloud,gpt-oss:120b-cloud,ministral-3:14b-cloud"
DEFAULT_SEEDS = "42,43,44"
DEFAULT_CONDITIONS = "b0_naive,b1_schema,b2_metafeature"
DEFAULT_OUTPUT = "results/sweep_results.jsonl"
DEFAULT_MAX_ITER = 5
DEFAULT_TIMEOUT = 300
PROGRESS_SUMMARY_EVERY = 50
CONSECUTIVE_INFRA_PAUSE_AT = 3
CONSECUTIVE_INFRA_EXIT_AT = 5
INFRA_PAUSE_SECONDS = 60

logger = logging.getLogger("run_sweep")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full experiment sweep with resume support."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help="Comma-separated Ollama model tags.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help="Comma-separated integer seeds.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=DEFAULT_CONDITIONS,
        help="Comma-separated condition keys (b0_naive, b1_schema, b2_metafeature).",
    )
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default=None,
        help="Comma-separated OpenML ids. Defaults to all 28 from SELECTED_CLASSIFICATION.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to the JSONL results file (append-only).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Max LLM refinement iterations per cell.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Per-subprocess wall-clock timeout in seconds.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume: every (dataset, condition, model, seed) is re-run "
        "even if already in the JSONL file (results are still appended).",
    )
    parser.add_argument(
        "--precache",
        action="store_true",
        help="Download and cache all selected datasets, then exit without "
        "running the sweep.",
    )
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_csv_strs(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _load_completed_keys(path: Path) -> set[tuple[int, str, str, int]]:
    """Return the set of cells already present in the JSONL file.

    A "cell" is identified by ``(dataset_id, condition, model, seed)``. Lines
    that fail to parse or lack required keys are skipped (a malformed line
    should not block restart). Both successful and failed cells count as
    completed -- re-running a known failure wastes compute without changing
    the experimental outcome.
    """
    if not path.exists():
        return set()
    keys: set[tuple[int, str, str, int]] = set()
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                keys.add(
                    (
                        int(rec["dataset_id"]),
                        str(rec["condition"]),
                        str(rec["model"]),
                        int(rec["seed"]),
                    )
                )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning("Skipping malformed JSONL line %d in %s", line_no, path)
    return keys


def _tail(text: str, n: int = 10) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:]) if len(lines) > n else text


def _run_cell_with_refinement(
    dataset_info: dict[str, Any],
    condition_key: str,
    condition: PromptCondition,
    model: str,
    seed: int,
    max_iter: int,
    timeout: int,
) -> dict[str, Any]:
    """Execute one cell with iterative LLM refinement, return the JSONL record.

    Mirrors the loop in scripts/dry_run.py but in compact form: each failure
    appends a short failure summary to the base prompt so the next attempt has
    the error context. Stops at the first success or after ``max_iter``
    attempts. The returned dict is the row written to the sweep's JSONL file
    (the schema is local to this script, not the same as ``RunResult``).
    """
    df_train = dataset_info["df_train"]
    target_col = dataset_info["target_col"]
    task_type = dataset_info["task_type"]
    dataset_name = dataset_info["dataset_name"]
    dataset_id = dataset_info["dataset_id"]

    meta = extract_meta(
        df_train,
        target_col=target_col,
        dataset_id=dataset_name,
        task_type=task_type,
    )
    task_description = build_task_description(dataset_info)
    base_prompt = condition.build_prompt(
        task_description, meta, df_train.head(5), target_col
    )

    cell_start = time.monotonic()
    current_prompt = base_prompt
    best_score: float | None = None
    final_error_category: str | None = None
    final_error_message: str | None = None
    iterations_used = 0

    for i in range(max_iter):
        iterations_used = i + 1
        code = call_llm(current_prompt, backend=model, seed=seed + i)
        pipeline = GeneratedPipeline(
            code=code,
            condition=condition.condition_name,
            llm_backend=model,
            dataset_id=dataset_name,
            seed=seed,
            iteration=i,
        )
        result = execute_pipeline(
            pipeline, task_type=task_type, timeout_seconds=timeout
        )
        if result.success:
            best_score = result.test_score
            final_error_category = None
            final_error_message = None
            break

        final_error_category = result.error_category
        final_error_message = result.error_message
        current_prompt = (
            base_prompt
            + "\n\n## Previous Attempt Failure\n"
            + f"The previous attempt failed: {final_error_category}: "
            + f"{_tail(final_error_message or '', 5)}\n"
            + "Generate corrected code that addresses this specific error."
        )

    wall = time.monotonic() - cell_start
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "condition": condition_key,
        "model": model,
        "seed": seed,
        "score": best_score,
        "iterations_used": iterations_used,
        "max_iterations": max_iter,
        "error_category": final_error_category if best_score is None else None,
        "error_message": final_error_message if best_score is None else None,
        "wall_time_seconds": round(wall, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _precache_datasets(dataset_ids: list[int]) -> int:
    """Download and persist every selected dataset; return process exit code."""
    failures: list[tuple[int, str]] = []
    for idx, ds_id in enumerate(dataset_ids, start=1):
        name = SELECTED_CLASSIFICATION.get(ds_id, f"openml_{ds_id}")
        try:
            info = load_dataset(ds_id, seed=42)
            logger.info(
                "[%d/%d] Cached: %s (train=%s, test=%s)",
                idx,
                len(dataset_ids),
                name,
                info["df_train"].shape,
                info["df_test"].shape,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[%d/%d] FAILED to cache %s (id=%d): %s",
                idx,
                len(dataset_ids),
                name,
                ds_id,
                exc,
            )
            failures.append((ds_id, str(exc)))
    if failures:
        logger.error("Pre-cache finished with %d failures: %s", len(failures), failures)
        return 1
    logger.info("Pre-cache complete: %d datasets ready.", len(dataset_ids))
    return 0


def _format_progress(
    completed: int,
    total: int,
    record: dict[str, Any],
) -> str:
    score = record["score"]
    if score is None:
        outcome = f"FAILED:{record['error_category']}"
    else:
        outcome = f"{score:.4f}"
    return (
        f"[{completed:>3}/{total}] {record['dataset_name']} | "
        f"{record['condition']} | {record['model']} | seed={record['seed']} | "
        f"iter={record['iterations_used']}/{record['max_iterations']} | "
        f"{record['wall_time_seconds']:.1f}s -> {outcome}"
    )


def _format_summary(
    completed_this_session: int,
    total_to_run: int,
    successes: int,
    total_time: float,
) -> str:
    pct = (completed_this_session / total_to_run * 100) if total_to_run else 0.0
    success_pct = (
        (successes / completed_this_session * 100) if completed_this_session else 0.0
    )
    avg = total_time / completed_this_session if completed_this_session else 0.0
    remaining = total_to_run - completed_this_session
    eta_hours = (avg * remaining) / 3600.0
    return (
        f"--- Progress: {completed_this_session}/{total_to_run} ({pct:.1f}%) "
        f"| Success: {successes}/{completed_this_session} ({success_pct:.0f}%) "
        f"| Avg time: {avg:.0f}s/run "
        f"| ETA: ~{eta_hours:.1f} hours ---"
    )


def _make_infra_record(
    ds_id: int,
    dataset_name: str,
    condition_key: str,
    model: str,
    seed: int,
    max_iter: int,
    cell_started: float,
    message: str,
) -> dict[str, Any]:
    return {
        "dataset_id": ds_id,
        "dataset_name": dataset_name,
        "condition": condition_key,
        "model": model,
        "seed": seed,
        "score": None,
        "iterations_used": 0,
        "max_iterations": max_iter,
        "error_category": "infrastructure",
        "error_message": message,
        "wall_time_seconds": round(time.monotonic() - cell_started, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _run_sweep(
    dataset_ids: list[int],
    models: list[str],
    condition_keys: list[str],
    seeds: list[int],
    output_path: Path,
    max_iter: int,
    timeout: int,
    resume: bool,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = _load_completed_keys(output_path) if resume else set()

    total = len(dataset_ids) * len(models) * len(condition_keys) * len(seeds)
    to_run = total - len(completed_keys)
    logger.info(
        "Sweep plan: %d datasets x %d models x %d conditions x %d seeds = %d cells.",
        len(dataset_ids),
        len(models),
        len(condition_keys),
        len(seeds),
        total,
    )
    if resume and completed_keys:
        logger.info(
            "Resuming: %d cells already completed, %d remaining.",
            len(completed_keys),
            to_run,
        )
    elif not resume:
        logger.info("--no-resume: existing JSONL entries will be re-run and appended.")

    if to_run <= 0:
        logger.info("Nothing to do.")
        return 0

    condition_instances = {key: _CONDITION_MAP[key]() for key in condition_keys}

    completed_this_session = 0
    successes_this_session = 0
    wall_total = 0.0
    consecutive_infra_errors = 0

    with output_path.open("a", encoding="utf-8") as fout:
        # Dataset-outermost so a long sweep yields complete per-dataset rows
        # early, which lets downstream analysis start before the run finishes.
        for ds_id in dataset_ids:
            try:
                dataset_info = load_dataset(ds_id, seed=42)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to load dataset %d: %s -- skipping its cells.",
                    ds_id,
                    exc,
                )
                continue

            for model in models:
                for condition_key in condition_keys:
                    condition = condition_instances[condition_key]
                    for seed in seeds:
                        key = (ds_id, condition_key, model, seed)
                        if resume and key in completed_keys:
                            continue

                        cell_started = time.monotonic()
                        try:
                            record = _run_cell_with_refinement(
                                dataset_info=dataset_info,
                                condition_key=condition_key,
                                condition=condition,
                                model=model,
                                seed=seed,
                                max_iter=max_iter,
                                timeout=timeout,
                            )
                            consecutive_infra_errors = 0
                        except ConnectionError as exc:
                            consecutive_infra_errors += 1
                            record = _make_infra_record(
                                ds_id=ds_id,
                                dataset_name=dataset_info["dataset_name"],
                                condition_key=condition_key,
                                model=model,
                                seed=seed,
                                max_iter=max_iter,
                                cell_started=cell_started,
                                message=f"ConnectionError: {exc}",
                            )
                        except Exception as exc:  # noqa: BLE001
                            record = _make_infra_record(
                                ds_id=ds_id,
                                dataset_name=dataset_info["dataset_name"],
                                condition_key=condition_key,
                                model=model,
                                seed=seed,
                                max_iter=max_iter,
                                cell_started=cell_started,
                                message=f"{type(exc).__name__}: {exc}",
                            )

                        fout.write(json.dumps(record) + "\n")
                        fout.flush()

                        completed_this_session += 1
                        wall_total += record["wall_time_seconds"]
                        if record["score"] is not None:
                            successes_this_session += 1
                        logger.info(
                            _format_progress(completed_this_session, to_run, record)
                        )

                        if (
                            completed_this_session
                            and completed_this_session % PROGRESS_SUMMARY_EVERY == 0
                        ):
                            logger.info(
                                _format_summary(
                                    completed_this_session,
                                    to_run,
                                    successes_this_session,
                                    wall_total,
                                )
                            )

                        if consecutive_infra_errors >= CONSECUTIVE_INFRA_EXIT_AT:
                            logger.error(
                                "%d consecutive infrastructure errors -- exiting. "
                                "Progress is preserved in %s; restart with the same "
                                "command to resume.",
                                consecutive_infra_errors,
                                output_path,
                            )
                            return 2
                        if consecutive_infra_errors == CONSECUTIVE_INFRA_PAUSE_AT:
                            logger.warning(
                                "%d consecutive infrastructure errors -- pausing "
                                "%ds before continuing.",
                                consecutive_infra_errors,
                                INFRA_PAUSE_SECONDS,
                            )
                            time.sleep(INFRA_PAUSE_SECONDS)

    logger.info(
        _format_summary(
            completed_this_session, to_run, successes_this_session, wall_total
        )
    )
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()

    if args.dataset_ids:
        dataset_ids = _parse_csv_ints(args.dataset_ids)
    else:
        dataset_ids = list(SELECTED_CLASSIFICATION.keys())

    models = _parse_csv_strs(args.models)
    seeds = _parse_csv_ints(args.seeds)
    condition_keys = _parse_csv_strs(args.conditions)

    unknown = [k for k in condition_keys if k not in _CONDITION_MAP]
    if unknown:
        logger.error(
            "Unknown conditions: %s. Valid keys: %s", unknown, list(_CONDITION_MAP)
        )
        return 2

    if args.precache:
        return _precache_datasets(dataset_ids)

    return _run_sweep(
        dataset_ids=dataset_ids,
        models=models,
        condition_keys=condition_keys,
        seeds=seeds,
        output_path=Path(args.output),
        max_iter=args.max_iter,
        timeout=args.timeout,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    sys.exit(main())
