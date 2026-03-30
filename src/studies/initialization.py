from __future__ import annotations

import argparse
from pathlib import Path

from artifacts import write_json
from presets import DATASETS, INITIALIZATION_METHODS, INITIALIZATION_METRICS, INITIALIZATION_PROBES, PAPER_DATASETS


def build_initialization_plan(datasets: list[str]) -> dict[str, object]:
    return {
        "study": "initialization",
        "datasets": [DATASETS[key].to_dict() for key in datasets],
        "methods": list(INITIALIZATION_METHODS),
        "probes": list(INITIALIZATION_PROBES),
        "metrics": list(INITIALIZATION_METRICS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the initialization-study plan.")
    parser.add_argument("--datasets", default=",".join(PAPER_DATASETS))
    parser.add_argument("--out-root", default="paper_exports")
    args = parser.parse_args()

    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()]
    payload = build_initialization_plan(datasets)
    path = Path(args.out_root) / "initialization_plan.json"
    write_json(path, payload)
    print(path)


if __name__ == "__main__":
    main()
