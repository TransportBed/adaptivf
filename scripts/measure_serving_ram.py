#!/usr/bin/env python3
"""Measure serving RAM for every completed experiment in a unified, fair way.

For each experiment directory, spawns a **fresh subprocess** that loads only the
artifacts genuinely needed for query serving (no TensorFlow for router methods,
no ground-truth neighbours, no training-only arrays) and reports the process RSS.

Output: one JSON-lines file (stdout) with per-experiment measurements.

Usage:
    cd adaptivf
    PYTHONPATH=src python scripts/measure_serving_ram.py \
        --experiments-root ../experiments/adaptivf \
        --data-root ../data
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset → HDF5 resolution (duplicated intentionally to avoid importing the
# full datasets module which pulls TF / sklearn as side-effects).
# ---------------------------------------------------------------------------
_HDF5_MAP: dict[str, str] = {
    "glove": "glove/glove-100-angular.hdf5",
    "sift": "sift/sift-128-euclidean.hdf5",
    "gist": "gist/gist-960-euclidean.hdf5",
    "deep1m": "deep1m/deep-image-96-angular.hdf5",
    "glove10k": "glove10k/glove10k.hdf5",
}


def _classify_method(config: dict) -> str:
    """Return one of 'faiss', 'router', or 'lira'."""
    method = config.get("method", "").lower()
    if method == "lira":
        return "lira"
    if config.get("index_kind") in ("hnsw", "ivf", "ivfpq"):
        return "faiss"
    return "router"


def _measure_one(exp_dir: Path, data_root: Path) -> dict:
    """Spawn a subprocess that loads serving artifacts and reports RSS."""
    config = json.loads((exp_dir / "config.json").read_text("utf-8"))
    dataset = config["dataset"]
    method = config.get("method", "unknown")
    method_class = _classify_method(config)

    hdf5_rel = _HDF5_MAP.get(dataset)
    if hdf5_rel is None:
        return {"experiment": str(exp_dir), "method": method, "error": f"unknown dataset {dataset}"}
    hdf5_path = (data_root / hdf5_rel).resolve()

    # Build the inner‐subprocess Python code as a string.
    # Each branch loads exactly the serving artifacts and prints RSS.
    inner_code = _build_inner_code(method_class, config, exp_dir, hdf5_path)

    env = dict(__import__("os").environ)
    env.pop("PYTHONPATH", None)  # clean env – no side-effect imports

    try:
        proc = subprocess.run(
            [sys.executable, "-c", inner_code],
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        if proc.returncode != 0:
            return {"experiment": str(exp_dir), "method": method, "error": proc.stderr.strip()[:500]}
        payload = json.loads(proc.stdout.strip())
    except Exception as exc:
        return {"experiment": str(exp_dir), "method": method, "error": str(exc)[:300]}

    payload["experiment"] = str(exp_dir)
    payload["method"] = method
    payload["dataset"] = dataset
    return payload


def _build_inner_code(method_class: str, config: dict, exp_dir: Path, hdf5_path: Path) -> str:
    """Return Python source executed in the subprocess."""
    exp_str = str(exp_dir)
    h5_str = str(hdf5_path)

    preamble = f"""\
import json, math, resource
from pathlib import Path

def _rss_mb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return float(line.split()[1]) / 1024.0
    return float("nan")

rss_baseline = _rss_mb()
"""

    epilogue = """\
rss_serving = _rss_mb()
try:
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
except Exception:
    peak_rss = float("nan")
print(json.dumps({
    "rss_baseline_mb": rss_baseline,
    "rss_serving_mb": rss_serving,
    "rss_peak_mb": peak_rss,
    "method_overhead_mb": max(0.0, rss_serving - rss_baseline) if math.isfinite(rss_serving) and math.isfinite(rss_baseline) else float("nan"),
}))
"""

    if method_class == "faiss":
        body = f"""\
import faiss
index = faiss.read_index("{exp_str}/staging/faiss.index")
"""

    elif method_class == "router":
        reps = config.get("repetitions", 4)
        pq_enabled = config.get("pq_enabled", False)
        body = f"""\
import numpy as np, h5py

# Load base vectors (needed for exact reranking at query time)
with h5py.File("{h5_str}", "r") as f:
    base_vectors = np.array(f["train"], dtype=np.float32)

# Load router MLP weights
weights = np.load("{exp_str}/models/router_weights.npz")

# Load lookup tables
point_to_buckets = np.load("{exp_str}/staging/point_to_buckets.npy")
for i in range({reps}):
    offsets = np.load(f"{exp_str}/staging/lookups/rep{{i}}_offsets.npy")
    ids = np.load(f"{exp_str}/staging/lookups/rep{{i}}_ids.npy")
"""
        if pq_enabled:
            body += f"""\
# PQ artifacts
pq_codes = np.load("{exp_str}/staging/pq_codes.npy")
pq_codebooks = np.load("{exp_str}/staging/pq_codebooks.npy")
"""
        # Load optional IVF artifacts
        body += f"""\
import os
for fname in ("ivf_centroids.npy", "ivf_permutations.npy",
              "ivf_inv_permutations.npy", "ivf_list_ids_index.npy"):
    p = "{exp_str}/staging/" + fname
    if os.path.exists(p):
        np.load(p)
"""

    elif method_class == "lira":
        body = f"""\
import numpy as np, h5py, torch, faiss
from pathlib import Path

# Load base vectors
with h5py.File("{h5_str}", "r") as f:
    base_vectors = np.array(f["train"], dtype=np.float32)

# Load LIRA model + artifacts
model_blob = torch.load("{exp_str}/models/lira_probe.pt", map_location="cpu")
centroids = np.load("{exp_str}/staging/lira_centroids.npy")
scaler = np.load("{exp_str}/staging/lira_distance_scaler.npz")
selected_ids = np.load("{exp_str}/staging/lira_selected_ids.npy")
selected_offsets = np.load("{exp_str}/staging/lira_selected_offsets.npy")

# Load inner FAISS indexes
inner_dir = Path("{exp_str}/staging/lira_inner_indexes")
if inner_dir.exists():
    for idx_file in sorted(inner_dir.glob("bucket_*.index")):
        faiss.read_index(str(idx_file))
"""
    else:
        body = "pass  # unknown method class\n"

    return preamble + body + epilogue


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure serving RAM for completed experiments.")
    parser.add_argument("--experiments-root", default=None, help="Root dir containing dataset subdirs")
    parser.add_argument("--data-root", required=True, help="Root dir containing dataset HDF5 files")
    parser.add_argument("--experiments-file", default=None, help="Optional newline-delimited file of experiment directories to measure")
    parser.add_argument("--output", default=None, help="Output JSON file (default: stdout)")
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()

    results: list[dict] = []
    experiment_dirs: list[Path] = []
    if args.experiments_file:
        file_path = Path(args.experiments_file).expanduser().resolve()
        for line in file_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                experiment_dirs.append(Path(text).expanduser().resolve())
    else:
        if not args.experiments_root:
            raise SystemExit("Either --experiments-root or --experiments-file is required")
        exp_root = Path(args.experiments_root).expanduser().resolve()
        for ds_dir in sorted(exp_root.iterdir()):
            if not ds_dir.is_dir():
                continue
            for exp_dir in sorted(ds_dir.iterdir()):
                if exp_dir.is_dir():
                    experiment_dirs.append(exp_dir)

    for exp_dir in experiment_dirs:
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            continue
        has_results = (exp_dir / "competitiveness_summary.json").exists() or (exp_dir / "metrics.json").exists()
        if not has_results:
            continue

        print(f"Measuring {exp_dir.name} ...", file=sys.stderr, flush=True)
        row = _measure_one(exp_dir, data_root)
        results.append(row)
        print(json.dumps(row), flush=True)

    if args.output:
        Path(args.output).write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nWrote {len(results)} measurements to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
