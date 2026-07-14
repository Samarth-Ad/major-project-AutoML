"""Phase 2: refit each manifest row and dump the fitted estimator to artifacts/."""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "script_manifest.csv"
ARTIFACTS = ROOT / "artifacts"
WRAPPER = Path(__file__).resolve().parent / "_save_wrapper.py"
TIMEOUT_S = 300


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def cell_dir(row: dict) -> Path:
    return ARTIFACTS / slug(row["model"]) / row["dataset_name"] / row["condition"] / f"seed{row['seed']}"


def refit_one(row: dict) -> tuple[bool, str]:
    dest = cell_dir(row)
    dest.mkdir(parents=True, exist_ok=True)
    pipeline_py = dest / "pipeline.py"
    joblib_out = dest / "pipeline.joblib"
    err_path = dest / "refit_error.txt"
    if err_path.exists():
        err_path.unlink()

    shutil.copy2(ROOT / row["script_path"], pipeline_py)
    try:
        proc = subprocess.run(
            [sys.executable, str(WRAPPER), str(pipeline_py), str(joblib_out)],
            capture_output=True, text=True, timeout=TIMEOUT_S, cwd=dest,
        )
    except subprocess.TimeoutExpired as e:
        err_path.write_text(f"TIMEOUT after {TIMEOUT_S}s\n{e.stderr or ''}", encoding="utf-8")
        return False, "timeout"

    if proc.returncode == 0 and joblib_out.exists():
        return True, "ok"
    if joblib_out.exists():
        joblib_out.unlink()  # remove truncated file from failed pickle
    err_path.write_text(
        f"returncode={proc.returncode}\n--- STDOUT ---\n{proc.stdout}\n--- STDERR ---\n{proc.stderr}",
        encoding="utf-8",
    )
    first_stderr_line = (proc.stderr.strip().splitlines() or [""])[-1][:200]
    return False, first_stderr_line or f"rc={proc.returncode}"


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open()))
    saved = 0
    failed: list[tuple[dict, str]] = []
    for i, row in enumerate(rows, 1):
        ok, msg = refit_one(row)
        if ok:
            saved += 1
        else:
            failed.append((row, msg))
        if i % 25 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] saved={saved} failed={len(failed)}", flush=True)

    print(f"\nSaved: {saved}   Failed: {len(failed)}")

    if len(failed) > 10:
        from collections import Counter
        by_cond = Counter(r["condition"] for r, _ in failed)
        by_model = Counter(r["model"] for r, _ in failed)
        by_dataset = Counter(r["dataset_name"] for r, _ in failed)
        by_err = Counter(msg for _, msg in failed)
        print("\nFailure clusters (>10 failures — surfacing patterns, no auto-fix):")
        print("  by condition:", dict(by_cond))
        print("  by model:", dict(by_model))
        print("  by dataset (top 5):", by_dataset.most_common(5))
        print("  top-3 error tails:")
        for e, c in by_err.most_common(3):
            print(f"    {c:4d}  {e}")


if __name__ == "__main__":
    main()
