from __future__ import annotations

import argparse
from pathlib import Path

from artifacts import write_json
from presets import COMPETITIVENESS_METRICS, COMPRESSED_METHODS, DATASETS, LOAD_BALANCE_METHODS, PAPER_DATASETS, UNCOMPRESSED_METHODS


def build_competitiveness_plan(datasets: list[str]) -> dict[str, object]:
    return {
        "study": "competitiveness",
        "datasets": [DATASETS[key].to_dict() for key in datasets],
        "uncompressed_methods": list(UNCOMPRESSED_METHODS),
        "compressed_methods": list(COMPRESSED_METHODS),
        "load_balance_methods": list(LOAD_BALANCE_METHODS),
        "metrics": list(COMPETITIVENESS_METRICS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the competitiveness-study plan.")
    parser.add_argument("--datasets", default=",".join(PAPER_DATASETS))
    parser.add_argument("--out-root", default="paper_exports")
    args = parser.parse_args()

    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()]
    payload = build_competitiveness_plan(datasets)
    path = Path(args.out_root) / "competitiveness_plan.json"
    write_json(path, payload)
    print(path)


if __name__ == "__main__":
    main()
