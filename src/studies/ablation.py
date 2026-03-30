from __future__ import annotations

import argparse
from pathlib import Path

from artifacts import write_json
from presets import ABLATION_METHODS, ABLATION_METRICS, DATASETS, PAPER_DATASETS


def build_ablation_plan(datasets: list[str]) -> dict[str, object]:
    return {
        "study": "ablation",
        "datasets": [DATASETS[key].to_dict() for key in datasets],
        "methods": list(ABLATION_METHODS),
        "metrics": list(ABLATION_METRICS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AdaptIVF ablation-study plan.")
    parser.add_argument("--datasets", default=",".join(PAPER_DATASETS))
    parser.add_argument("--out-root", default="paper_exports")
    args = parser.parse_args()

    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()]
    payload = build_ablation_plan(datasets)
    path = Path(args.out_root) / "ablation_plan.json"
    write_json(path, payload)
    print(path)


if __name__ == "__main__":
    main()
