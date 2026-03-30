from __future__ import annotations

from dataclasses import dataclass, replace

from methods.router_family import RouterConfig, RouterFamily, make_router_method


@dataclass(frozen=True)
class AdaptIVFConfig:
    seed: int = 0
    confidence_threshold: float = 0.75
    max_assignments: int = 3
    m_base: int = 5
    m_max: int = 10
    entropy_scale: float = 10.0
    reassign_interval: int = 5
    pq_enabled: bool = False
    pq_m: int = 16
    pq_bits: int = 8


def _base_router_config(dataset: str, config: AdaptIVFConfig) -> RouterConfig:
    method = "AdaptIVF+PQ" if config.pq_enabled else "AdaptIVF"
    base = make_router_method(method, dataset, seed=config.seed).cfg
    return replace(
        base,
        method=method,
        pq_enabled=bool(config.pq_enabled),
        pq_m=int(config.pq_m),
        pq_bits=int(config.pq_bits),
        assignment_strategy="confidence_threshold",
        assignment_threshold=float(config.confidence_threshold),
        max_assignments=int(config.max_assignments),
        probing_strategy="entropy_adaptive",
        m_base=int(config.m_base),
        m_max=int(config.m_max),
        entropy_scale=float(config.entropy_scale),
        reassign_interval=int(config.reassign_interval),
        seed=int(config.seed),
    )


def make_adaptivf(dataset: str, *, config: AdaptIVFConfig | None = None) -> RouterFamily:
    resolved = config or AdaptIVFConfig()
    return RouterFamily(_base_router_config(dataset, resolved))


__all__ = ["AdaptIVFConfig", "make_adaptivf"]
