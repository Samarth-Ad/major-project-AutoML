from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import requests

from src.conditions.base import PromptCondition
from src.contracts import GeneratedPipeline, RunResult
from src.execution.runner import execute_pipeline
from src.experiments.datasets import build_task_description, load_dataset
from src.meta_features.extractor import extract as extract_meta

_OLLAMA_URL = "http://localhost:11434/api/generate"
_FENCE_RE = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_llm(prompt: str, backend: str, seed: int) -> str:
    """POST to a local Ollama server and return the generated text (fences stripped)."""
    body = {
        "model": backend,
        "prompt": prompt,
        "stream": False,
        "options": {"seed": seed, "temperature": 0.7},
    }
    try:
        response = requests.post(_OLLAMA_URL, json=body, timeout=600)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ConnectionError(
            f"Failed to reach Ollama at {_OLLAMA_URL}. Is `ollama serve` running? Original: {exc}"
        ) from exc

    payload = response.json()
    text = payload.get("response", "")
    return _strip_fences(text)


def run_single(
    dataset_info: dict[str, Any],
    condition: PromptCondition,
    llm_backend: str,
    seed: int,
    iteration: int,
    timeout: int = 600,
) -> RunResult:
    """Run one (dataset, condition, llm, seed, iteration) atomic experiment."""
    df_train = dataset_info["df_train"]
    target_col = dataset_info["target_col"]
    task_type = dataset_info["task_type"]
    dataset_id = dataset_info["dataset_name"]

    meta = extract_meta(
        df_train,
        target_col=target_col,
        dataset_id=dataset_id,
        task_type=task_type,
    )
    task_description = build_task_description(dataset_info)
    prompt = condition.build_prompt(task_description, meta, df_train.head(5), target_col)

    code = call_llm(prompt, backend=llm_backend, seed=seed)

    pipeline = GeneratedPipeline(
        code=code,
        condition=condition.condition_name,
        llm_backend=llm_backend,
        dataset_id=dataset_id,
        seed=seed,
        iteration=iteration,
    )
    return execute_pipeline(pipeline, task_type=task_type, timeout_seconds=timeout)


def run_experiment(
    dataset_ids: list[int],
    conditions: list[PromptCondition],
    llm_backends: list[str],
    seeds: list[int],
    n_iterations: int = 1,
    timeout: int = 600,
    output_path: str = "results/experiment_results.jsonl",
) -> list[RunResult]:
    """Full sweep: dataset x condition x llm x seed x iteration. Streams JSONL."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = (
        len(dataset_ids) * len(conditions) * len(llm_backends) * len(seeds) * n_iterations
    )
    completed = 0
    results: list[RunResult] = []

    with out_path.open("a", encoding="utf-8") as fout:
        for dataset_id in dataset_ids:
            dataset_info = load_dataset(dataset_id)
            for condition in conditions:
                for llm in llm_backends:
                    for seed in seeds:
                        for iteration in range(n_iterations):
                            result = run_single(
                                dataset_info,
                                condition=condition,
                                llm_backend=llm,
                                seed=seed,
                                iteration=iteration,
                                timeout=timeout,
                            )
                            fout.write(result.model_dump_json() + "\n")
                            fout.flush()
                            results.append(result)
                            completed += 1
                            print(
                                f"[{completed}/{total}] {dataset_info['dataset_name']} | "
                                f"{condition.condition_name} | {llm} | seed={seed} | "
                                f"success={result.success} | score={result.test_score}"
                            )
    return results
