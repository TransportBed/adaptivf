from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from artifacts import write_csv, write_json
from presets import ABLATION_METHODS, COMPRESSED_METHODS, INITIALIZATION_METHODS, UNCOMPRESSED_METHODS


def _read_json(path: Path) -> dict[str, object] | list[object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(pd.read_csv(path).shape[0])
    except Exception:
        return 0


def _norm_text(value: object) -> str:
    return str(value or "").strip()


def _norm_int(value: object) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    return int(float(text))


def _repo_relative_path(value: object, *, repo_root: Path) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return os.path.relpath(path, repo_root).replace(os.sep, "/")


def _out_relative_path(path: Path, *, out_root: Path) -> str:
    return os.path.relpath(path.resolve(), out_root.resolve()).replace(os.sep, "/")


def _resolve_repo_relative_path(value: object, *, repo_root: Path) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return str(path)


def _sanitize_row_paths(row: dict[str, object], *, repo_root: Path) -> dict[str, object]:
    out = dict(row)
    for key in ("experiment_dir", "experiment"):
        if key in out and _norm_text(out.get(key)):
            out[key] = _repo_relative_path(out.get(key), repo_root=repo_root)
    return out


def _study_sort_key(name: str, row: dict[str, object]) -> tuple[object, ...]:
    if name == "initialization":
        method_order = {method: idx for idx, method in enumerate(INITIALIZATION_METHODS)}
        return (
            _norm_text(row.get("dataset")),
            method_order.get(_norm_text(row.get("method")), 999),
            _norm_int(row.get("probe_depth")),
            _norm_int(row.get("seed")),
        )
    if name == "ablations":
        method_order = {method: idx for idx, method in enumerate(ABLATION_METHODS)}
    else:
        ordered = list(UNCOMPRESSED_METHODS) + list(COMPRESSED_METHODS) + ["AdaptIVF-m80", "AdaptIVF+PQ-m80"]
        method_order = {method: idx for idx, method in enumerate(ordered)}
    return (
        _norm_text(row.get("dataset")),
        method_order.get(_norm_text(row.get("method")), 999),
        _norm_int(row.get("seed")),
    )


def _dedupe_rows(name: str, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    merged: dict[tuple[object, ...], dict[str, object]] = {}
    for row in rows:
        normalized = dict(row)
        normalized["dataset"] = _norm_text(row.get("dataset"))
        normalized["method"] = _norm_text(row.get("method"))
        normalized["seed"] = _norm_int(row.get("seed"))
        if name == "initialization":
            normalized["probe_depth"] = _norm_int(row.get("probe_depth"))
            key = (
                normalized["dataset"],
                normalized["method"],
                normalized["probe_depth"],
                normalized["seed"],
            )
        else:
            key = (
                normalized["dataset"],
                normalized["method"],
                normalized["seed"],
            )
        merged[key] = normalized
    return sorted(merged.values(), key=lambda row: _study_sort_key(name, row))


def _rebuild_study_exports(out_root: Path, name: str, *, repo_root: Path) -> dict[str, object]:
    study_dir = out_root / name
    if not study_dir.exists():
        return {
            "directory": str(study_dir),
            "csv": None,
            "json": None,
            "rows": 0,
        }

    all_rows: list[dict[str, object]] = []
    for dataset_json in sorted(study_dir.glob("*_summary.json")):
        if dataset_json.name == "all_datasets_summary.json":
            continue
        payload = _read_json(dataset_json)
        rows = payload if isinstance(payload, list) else []
        deduped = _dedupe_rows(name, rows)
        deduped = [_sanitize_row_paths(row, repo_root=repo_root) for row in deduped]
        write_json(dataset_json, deduped)
        write_csv(dataset_json.with_suffix(".csv"), deduped)
        all_rows.extend(deduped)

    all_rows = _dedupe_rows(name, all_rows)
    csv_path = study_dir / "all_datasets_summary.csv"
    json_path = study_dir / "all_datasets_summary.json"
    write_csv(csv_path, all_rows)
    write_json(json_path, all_rows)
    return {
        "directory": _out_relative_path(study_dir, out_root=out_root),
        "csv": _out_relative_path(csv_path, out_root=out_root),
        "json": _out_relative_path(json_path, out_root=out_root),
        "rows": len(all_rows),
    }


def _measure_serving_ram(
    *,
    out_root: Path,
    repo_root: Path,
    experiments_root: Path | None,
    data_root: Path | None,
    canonical_experiments: set[str],
) -> dict[str, object] | None:
    if not canonical_experiments or experiments_root is None or data_root is None:
        return None
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "measure_serving_ram.py"
    if not script_path.exists():
        return None

    output_path = out_root / "serving_ram.json"
    experiments_file = out_root / "serving_ram_experiments.txt"
    experiments_file.write_text("\n".join(sorted(canonical_experiments)) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--data-root",
            str(data_root),
            "--experiments-file",
            str(experiments_file),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if proc.stdout:
        print(proc.stdout.strip())
    if proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)

    payload = _read_json(output_path)
    rows = payload if isinstance(payload, list) else []
    filtered = [row for row in rows if _norm_text(row.get("experiment")) in canonical_experiments and not row.get("error")]
    filtered = sorted(filtered, key=lambda row: (_norm_text(row.get("dataset")), _norm_text(row.get("method"))))
    filtered = [_sanitize_row_paths(row, repo_root=repo_root) for row in filtered]
    write_json(output_path, filtered)
    write_csv(out_root / "serving_ram.csv", filtered)
    experiments_file.unlink(missing_ok=True)
    return {
        "json": _out_relative_path(output_path, out_root=out_root),
        "csv": _out_relative_path(out_root / "serving_ram.csv", out_root=out_root),
        "rows": len(filtered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect exported study artifacts into one manifest.")
    parser.add_argument("--out-root", default="paper_exports")
    parser.add_argument("--experiments-root", default=os.environ.get("EXPERIMENTS_ROOT"))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    args = parser.parse_args()

    out_root = Path(args.out_root)
    repo_root = Path(__file__).resolve().parents[1]
    studies = {
        name: _rebuild_study_exports(out_root, name, repo_root=repo_root)
        for name in ("initialization", "competitiveness", "ablations")
    }

    experiments_root = Path(args.experiments_root).expanduser().resolve() if args.experiments_root else None
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None
    canonical_comp = _read_json(out_root / "competitiveness" / "all_datasets_summary.json")
    canonical_rows = canonical_comp if isinstance(canonical_comp, list) else []
    canonical_experiments = {
        _resolve_repo_relative_path(row.get("experiment_dir"), repo_root=repo_root)
        for row in canonical_rows
        if _norm_text(row.get("experiment_dir"))
    }
    serving_ram = _measure_serving_ram(
        out_root=out_root,
        repo_root=repo_root,
        experiments_root=experiments_root,
        data_root=data_root,
        canonical_experiments=canonical_experiments,
    )

    payload = {
        "plans": {
            "initialization": _read_json(out_root / "initialization_plan.json"),
            "competitiveness": _read_json(out_root / "competitiveness_plan.json"),
            "ablation": _read_json(out_root / "ablation_plan.json"),
        },
        "studies": studies,
        "serving_ram": serving_ram,
    }
    path = out_root / "study_manifest.json"
    write_json(path, payload)
    print(path)


if __name__ == "__main__":
    main()
