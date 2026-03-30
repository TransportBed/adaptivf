from __future__ import annotations

from pathlib import Path

from methods.router_family import RouterFamily, make_router_method


class _AdaptIVFVariant:
    def __init__(self, method: str, dataset: str, *, seed: int = 0, m_max: int | None = None) -> None:
        self._impl: RouterFamily = make_router_method(method, dataset, seed=seed, m_max=m_max)
        self.name = self._impl.cfg.method

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


class AdaptIVF:
    def __init__(self, dataset: str, *, seed: int = 0, m_max: int | None = None) -> None:
        self._variant = _AdaptIVFVariant("AdaptIVF", dataset, seed=seed, m_max=m_max)
        self.name = self._variant.name

    def run_competitiveness(self, *, data_root: Path, exp_dir: Path, force_prepare: bool = False) -> dict[str, object]:
        return self._variant.run_competitiveness(
            data_root=data_root,
            exp_dir=exp_dir,
            force_prepare=force_prepare,
        )


class AdaptIVFPQ:
    def __init__(self, dataset: str, *, seed: int = 0, m_max: int | None = None) -> None:
        self._variant = _AdaptIVFVariant("AdaptIVF+PQ", dataset, seed=seed, m_max=m_max)
        self.name = self._variant.name

    def run_competitiveness(self, *, data_root: Path, exp_dir: Path, force_prepare: bool = False) -> dict[str, object]:
        return self._variant.run_competitiveness(
            data_root=data_root,
            exp_dir=exp_dir,
            force_prepare=force_prepare,
        )


def make_adaptivf_ablation(method: str, dataset: str, *, seed: int = 0, m_max: int | None = None) -> _AdaptIVFVariant:
    return _AdaptIVFVariant(method, dataset, seed=seed, m_max=m_max)
