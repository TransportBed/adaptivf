from __future__ import annotations

import argparse
from pathlib import Path

from artifacts import write_json
from presets import (
    COMPRESSED_METHODS,
    DATASETS,
    INITIALIZATION_METHODS,
    INITIALIZATION_PROBES,
    PAPER_DATASETS,
    PORTED_METHODS,
    UNCOMPRESSED_METHODS,
)


def build_run_plan(datasets: list[str]) -> dict[str, object]:
    jobs: list[dict[str, object]] = []
    for dataset in datasets:
        for method in INITIALIZATION_METHODS:
            for probe in INITIALIZATION_PROBES:
                jobs.append(
                    {
                        "study": "initialization",
                        "dataset": dataset,
                        "method": method,
                        "probe": int(probe),
                        "ported": method in PORTED_METHODS,
                    }
                )
        for method in UNCOMPRESSED_METHODS:
            jobs.append(
                {
                    "study": "competitiveness",
                    "dataset": dataset,
                    "method": method,
                    "probe": None,
                    "ported": method in PORTED_METHODS,
                }
            )
        for method in COMPRESSED_METHODS:
            jobs.append(
                {
                    "study": "competitiveness",
                    "dataset": dataset,
                    "method": method,
                    "probe": None,
                    "ported": method in PORTED_METHODS,
                }
            )

    return {
        "datasets": [DATASETS[key].to_dict() for key in datasets],
        "jobs": jobs,
        "ported_methods": list(PORTED_METHODS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the full sequential paper-run plan.")
    parser.add_argument("--datasets", default=",".join(PAPER_DATASETS))
    parser.add_argument("--out-root", default="paper_exports")
    args = parser.parse_args()

    datasets = [token.strip() for token in args.datasets.split(",") if token.strip()]
    payload = build_run_plan(datasets)
    path = Path(args.out_root) / "run_plan.json"
    write_json(path, payload)
    print(path)


if __name__ == "__main__":
    main()
