from __future__ import annotations

from pathlib import Path

from methods.router_family import RouterFamily, make_router_method


class Bliss:
    def __init__(self, dataset: str, *, seed: int = 0) -> None:
        self._impl: RouterFamily = make_router_method("BLISS", dataset, seed=seed)
        self.name = self._impl.cfg.method

    def run_initialization_sweep(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        probes: list[int],
        force_prepare: bool = False,
    ) -> list[dict[str, object]]:
        return self._impl.run_initialization_sweep(
            data_root=data_root,
            exp_dir=exp_dir,
            probes=probes,
            force_prepare=force_prepare,
        )

    def run_competitiveness(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool = False,
    ) -> dict[str, object]:
        return self._impl.run_competitiveness(
            data_root=data_root,
            exp_dir=exp_dir,
            force_prepare=force_prepare,
        )


class BlissKMeans:
    def __init__(self, dataset: str, *, seed: int = 0) -> None:
        self._impl: RouterFamily = make_router_method("BLISS-KMeans", dataset, seed=seed)
        self.name = self._impl.cfg.method

    def run_initialization_sweep(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        probes: list[int],
        force_prepare: bool = False,
    ) -> list[dict[str, object]]:
        return self._impl.run_initialization_sweep(
            data_root=data_root,
            exp_dir=exp_dir,
            probes=probes,
            force_prepare=force_prepare,
        )

    def run_competitiveness(
        self,
        *,
        data_root: Path,
        exp_dir: Path,
        force_prepare: bool = False,
    ) -> dict[str, object]:
        return self._impl.run_competitiveness(
            data_root=data_root,
            exp_dir=exp_dir,
            force_prepare=force_prepare,
        )
