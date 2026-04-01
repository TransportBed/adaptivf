# Next Directions

This note reflects the finished `2026-04-01` paper run. Three directions now
stand out clearly above the rest.

## 1. Calibrated Probe Control

This is the highest-value follow-up. The strongest AdaptIVF results come from
the `m80` competitiveness variants rather than the default `m10` controller:

- `GloVe`: `0.736 -> 0.913`
- `SIFT`: `0.825 -> 0.956`
- `GIST`: `0.568 -> 0.921`
- `Deep1M`: `0.822 -> 0.954`

That gap is too large to treat as minor tuning. It implies that the current
entropy-to-budget schedule is still too conservative. The best next work is
therefore better controller calibration on the same backbone:

- dataset-specific `m_base`, `m_max`, and `lambda`
- piecewise or banded probe schedules instead of one global exponential map
- held-out calibration objectives that jointly value recall and query cost

## 2. Throughput-First Serving Optimization

The finished run suggests that AdaptIVF's remaining weakness is not training
cost or index size, but serving efficiency. The current numbers come from a
CPU-only Python path, so they should not be read as a hard ceiling. The most
direct next improvements are implementation-level:

- C/C++ query-serving kernels for routing, aggregation, and deduplication
- batched shortlist handling and cheaper reranking admission
- GPU-backed execution for the online path

The goal is straightforward: preserve the high-cap recall regime while moving
meaningfully closer to the classical throughput baselines.

## 3. Controller-Aware Compression

PQ is now clearly dataset-dependent:

- strong on `SIFT`
- strong on `Deep1M`
- acceptable on `GloVe`
- still weak on `GIST`

That pattern suggests compression should be controlled, not merely enabled.
Promising directions are:

- entropy-aware shortlist sizing before exact reranking
- dataset-specific PQ settings rather than one global recipe
- tighter coupling between uncertainty, shortlist length, and compressed scan
  depth

## What Not To Prioritize Next

The finished run does not justify another round of baseline expansion or a
return to heavier multi-index designs. The current repository already shows
that the compact single-backbone idea is viable. The highest-value next step is
to make that same idea better calibrated and faster.
