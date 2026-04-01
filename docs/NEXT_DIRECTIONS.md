# Next Directions

This note reflects the finished paper run, not the earlier partial reruns. The
main pattern is now clear:

- the compact single-backbone AdaptIVF design is validated
- the low-cap default controller is not yet the strongest instantiation
- the main remaining gap is controller calibration and serving efficiency, not
  training cost

## 1. Calibrated Probe Control

This is the highest-value follow-up.

The strongest AdaptIVF results in the finished run come from the `m80`
competitiveness variants, not the default `m10` controller:

- `GloVe`: `0.736 -> 0.913`
- `SIFT`: `0.825 -> 0.956`
- `GIST`: `0.568 -> 0.921`
- `Deep1M`: `0.822 -> 0.954`

That is too large a gap to ignore. It means the current entropy-to-budget
mapping is not yet a good default controller. The next step is not another
index structure; it is a better probe-allocation policy on the same backbone.

Priority subdirections:

- learn or validate `m_base`, `m_max`, and `lambda` per dataset rather than
  fixing them globally
- treat entropy as a calibration signal, not only as a monotonic budget knob
- explore piecewise or banded probe schedules instead of the current single
  exponential mapping
- evaluate whether the controller should optimize for recall, QPS, or a mixed
  objective on a held-out calibration set

## 2. Throughput-First Serving Optimization

The finished run confirms that recall headroom exists, but QPS remains the main
systems bottleneck once the probe cap is raised.

Examples:

- `AdaptIVF-m80` is strong on recall, but still far below `IVF` / `IVFPQ` in
  raw throughput
- `AdaptIVF+PQ-m80` preserves much of the recall gain on `SIFT` and `Deep1M`,
  but its QPS remains low

The best next work is therefore on the serving path, not on retraining another
router architecture.

Priority subdirections:

- faster candidate deduplication and reranking
- batched probe-depth bands instead of fully per-query branching
- cheaper shortlist admission before exact scoring
- faster ADC and shortlist handling for the PQ path

The success criterion is simple: preserve the `m80` recall regime while
improving QPS by a meaningful constant factor.

## 3. Controller-Aware Compression

The PQ story is now clearly dataset-dependent.

- `SIFT`: high-cap PQ remains excellent
- `Deep1M`: high-cap PQ remains strong
- `GloVe`: PQ is acceptable but not neutral
- `GIST`: PQ is still the hardest case

This suggests compression should be coupled to uncertainty and dataset
difficulty rather than treated as one global mode.

Priority subdirections:

- entropy-gated exact reranking vs PQ reranking
- dataset-specific PQ settings rather than a single uniform compression recipe
- compressed shortlist kernels that reduce QPS loss, not just index size

## 4. Stronger Default Variant Selection

The ablation results show that `AdaptIVF-Static` beats default `AdaptIVF` on
most datasets at the current low-cap setting, while `AdaptIVF-A4` does not fix
that gap consistently. So the repository now has three distinct roles:

- `Static`: strong conservative baseline on the same backbone
- `m10`: current adaptive default, but under-calibrated
- `m80`: strongest recall-oriented competitiveness setting

The next development cycle should make this relationship explicit and then
collapse it where possible:

- either improve the adaptive default until it matches the stronger variants
- or expose a principled “compact / balanced / recall-oriented” controller menu

## 5. Memory and Resource Semantics

The repository now exports both query RAM and isolated serving RAM, but the
reporting surface can still be improved.

Priority subdirections:

- keep `query_mem_delta_mb` and `rss_serving_mb` clearly separated in the paper
  and code
- preserve `index_overhead_mb` as the primary compactness metric, but continue
  reporting serving footprint and RAM separately
- document the operational meaning of each systems metric directly in the
  export bundle

This matters because the long-term value of AdaptIVF is not just recall. It is
the claim that a compact learned IVF backbone can be easier to train, easier to
store, and easier to reason about operationally than heavier learned partition
systems.

## What Not To Prioritize Next

The finished run does **not** justify spending the next cycle on:

- more repetitions or moving back toward multi-index designs
- expanding the baseline list further
- redesigning the router architecture before the current controller is
  calibrated and the serving path is faster

The method already has enough novelty. The highest-value next step is to make
the strongest version of the existing idea more self-calibrating and faster.
