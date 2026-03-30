from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from artifacts import write_json


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect exported study artifacts into one manifest.")
    parser.add_argument("--out-root", default="paper_exports")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    studies = {}
    for name in ("initialization", "competitiveness", "ablations"):
        study_dir = out_root / name
        csv_path = study_dir / "all_datasets_summary.csv"
        json_path = study_dir / "all_datasets_summary.json"
        studies[name] = {
            "directory": str(study_dir),
            "csv": str(csv_path) if csv_path.exists() else None,
            "json": str(json_path) if json_path.exists() else None,
            "rows": _row_count(csv_path),
        }

    payload = {
        "plans": {
            "initialization": _read_json(out_root / "initialization_plan.json"),
            "competitiveness": _read_json(out_root / "competitiveness_plan.json"),
            "ablation": _read_json(out_root / "ablation_plan.json"),
        },
        "studies": studies,
    }
    path = out_root / "study_manifest.json"
    write_json(path, payload)
    print(path)


if __name__ == "__main__":
    main()
