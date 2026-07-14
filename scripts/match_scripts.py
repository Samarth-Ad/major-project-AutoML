"""Phase 1: match each successful sweep row to its LLM-generated script by mtime."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSONL_FILES = [
    ROOT / "results" / "1_sweep_results.jsonl",
    ROOT / "results" / "2_sweep_results.jsonl",
    ROOT / "results" / "sweep_results.jsonl",
]
RUNS_DIR = ROOT / "logs" / "runs"
MANIFEST = ROOT / "results" / "script_manifest.csv"

COND_TO_TAG = {"b0_naive": "B0", "b1_schema": "B1", "b2_metafeature": "B2"}
FNAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<tag>B[012])_seed(?P<seed>\d+)_iter(?P<it>\d+)_[^.]+\.py$")


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for path in JSONL_FILES:
        with path.open() as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("error_category") is None:
                    r["_source"] = path.name
                    rows.append(r)
    return rows


def index_scripts() -> dict[tuple[str, str, int], list[tuple[Path, float, int]]]:
    idx: dict[tuple[str, str, int], list[tuple[Path, float, int]]] = defaultdict(list)
    for f in RUNS_DIR.iterdir():
        m = FNAME_RE.match(f.name)
        if not m:
            continue
        key = (m["prefix"], m["tag"], int(m["seed"]))
        idx[key].append((f, f.stat().st_mtime, int(m["it"])))
    return idx


def pick(candidates: list[tuple[Path, float, int]], row_ts: str) -> Path:
    target = datetime.fromisoformat(row_ts).timestamp()
    return min(candidates, key=lambda c: abs(c[1] - target))[0]


def main() -> None:
    rows = load_rows()
    idx = index_scripts()
    matched: list[dict] = []
    unmatched: list[dict] = []
    for r in rows:
        tag = COND_TO_TAG[r["condition"]]
        key = (r["dataset_name"], tag, int(r["seed"]))
        cands = idx.get(key)
        if not cands:
            unmatched.append(r)
            continue
        script = pick(cands, r["timestamp"])
        matched.append({
            "dataset_id": r["dataset_id"],
            "dataset_name": r["dataset_name"],
            "condition": r["condition"],
            "model": r["model"],
            "seed": r["seed"],
            "score": r["score"],
            "iterations_used": r["iterations_used"],
            "script_path": str(script.relative_to(ROOT)).replace("\\", "/"),
            "timestamp": r["timestamp"],
            "sweep_source": r["_source"],
        })

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset_id", "dataset_name", "condition", "model", "seed",
        "score", "iterations_used", "script_path", "timestamp", "sweep_source",
    ]
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(matched)

    print(f"Matched: {len(matched)} / Unmatched: {len(unmatched)}")
    if unmatched:
        sample = [(r["dataset_name"], r["condition"], r["seed"]) for r in unmatched[:3]]
        print(f"Sample unmatched (up to 3): {sample}")
        pct = 100 * len(unmatched) / (len(matched) + len(unmatched))
        if pct > 5:
            names = sorted({r["dataset_name"] for r in unmatched})
            print(f"Unmatched > 5% ({pct:.1f}%). Distinct unmatched dataset_names: {names}")


if __name__ == "__main__":
    main()
