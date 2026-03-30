from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def paper_exports_root(repo_root: Path) -> Path:
    return ensure_dir(repo_root / "paper_exports")


def experiments_root(repo_root: Path) -> Path:
    return ensure_dir(repo_root / "experiments")


def new_experiment_dir(root: Path, dataset: str, method: str, *, seed: int | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    token = method.lower().replace("+", "p").replace("-", "_")
    if seed is not None:
        token = f"{token}_seed{int(seed)}"
    path = root / dataset / f"{stamp}_{token}"
    return ensure_dir(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path
