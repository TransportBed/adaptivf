from __future__ import annotations

import json
from pathlib import Path

from artifacts import write_json
from console import banner, info, print_table
from index_manifest import write_index_manifest
from measurement_contract import (
    AVG_COMPUTATIONS_MODE_ROUTER,
    INDEX_OVERHEAD_MODE,
    QUERY_MEM_MODE_ISOLATED,
    SERVING_FOOTPRINT_MODE,
)
from presets import DATASETS

from methods.lira_runtime import LiraConfig, run_lira_selected_online_eval, run_lira_smallscale


class Lira:
    def __init__(self, dataset: str, *, seed: int = 0) -> None:
        self.dataset = dataset
        self.name = "LIRA"
        self.cfg = LiraConfig(dataset=dataset, n_bkt=DATASETS[dataset].partitions, seed=seed)

    def run_competitiveness(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool = False,
    ) -> dict[str, object]:
        banner("method", f"{self.name} | dataset={self.dataset}")
        info(
            f"probe_depth={self.cfg.probe_depth}, n_bkt={self.cfg.n_bkt}, metric={self.cfg.metric}, "
            f"index_full_dataset={self.cfg.index_full_dataset}, selected_part_policy={self.cfg.selected_part_policy}"
        )
        info(
            f"epochs={self.cfg.n_epoch}, batch_size={self.cfg.batch_size}, n_mul={self.cfg.n_mul}, "
            f"repa_step={self.cfg.repa_step}, inner_index={self.cfg.inner_index_type}"
        )
        result = run_lira_smallscale(self.cfg, exp_dir, data_root=data_root, force_prepare=force_prepare)
        if not result.selected_rows:
            raise RuntimeError("LIRA produced no selected threshold rows.")
        target_nprobe = float(self.cfg.probe_depth)
        summary_row = min(
            result.selected_rows,
            key=lambda row: (
                abs(float(row.get("nprobe", float("nan"))) - target_nprobe),
                -float(row.get("recall10_at_10", float("nan"))),
                float(row.get("avg_computations", float("inf"))),
            ),
        )
        online = run_lira_selected_online_eval(
            self.cfg,
            exp_dir,
            data_root=data_root,
            threshold=float(summary_row["threshold"]),
        )
        manifest = write_index_manifest(exp_dir)
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        row: dict[str, object] = {
            "dataset": self.dataset,
            "method": self.name,
            "seed": int(self.cfg.seed),
            "probe_depth": int(self.cfg.probe_depth),
            "recall_at_10": float(online["recall10_at_10"]),
            "avg_computations": float(online["avg_computations"]),
            "computation_min": float(online.get("computation_min", float("nan"))),
            "computation_max": float(online.get("computation_max", float("nan"))),
            "avg_computations_mode": str(online.get("avg_computations_mode", AVG_COMPUTATIONS_MODE_ROUTER)),
            "avg_candidates": float(online["avg_computations"]),
            "avg_candidates_mode": str(online.get("avg_computations_mode", AVG_COMPUTATIONS_MODE_ROUTER)),
            "qps": float(online["qps"]),
            "index_overhead_mb": float(manifest_payload["index_overhead_mb"]),
            "index_overhead_mode": INDEX_OVERHEAD_MODE,
            "serving_footprint_mb": float(manifest_payload["serving_footprint_mb"]),
            "serving_footprint_mode": SERVING_FOOTPRINT_MODE,
            "index_size_mb": float(manifest_payload["index_overhead_mb"]),
            "index_size_mode": INDEX_OVERHEAD_MODE,
            "query_mem_delta_mb": float(online["query_mem_delta_mb"]),
            "query_mem_delta_mode": QUERY_MEM_MODE_ISOLATED,
            "train_s": float(result.metadata.get("timings_s", {}).get("total", 0.0)),
            "experiment_dir": str(exp_dir),
            "threshold": float(summary_row["threshold"]),
            "nprobe": float(online["nprobe"]),
            "selected_part": int(result.selected_part),
        }
        print_table(
            f"{self.name} | competitiveness",
            ["metric", "value"],
            [
                ["Recall@10", f"{float(row['recall_at_10']):.4f}"],
                ["Avg computations", f"{float(row['avg_computations']):.1f}"],
                ["QPS", f"{float(row['qps']):.1f}"],
                ["Index overhead (MB)", f"{float(row['index_overhead_mb']):.3f}"],
                ["Serving footprint (MB)", f"{float(row['serving_footprint_mb']):.3f}"],
                ["Selected part", str(int(row["selected_part"]))],
                ["Threshold", f"{float(row['threshold']):.4f}"],
                ["Online nprobe", f"{float(row['nprobe']):.2f}"],
            ],
        )
        write_json(exp_dir / "competitiveness_summary.json", row)
        return row
