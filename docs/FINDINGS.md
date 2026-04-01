# Experimental Findings

This document summarizes the completed `2026-04-01` paper run now exported in
[`paper_exports/`](../paper_exports).

## Stable Conclusions

The finished run supports five stable conclusions.

- **Initialization matters, but not monotonically.** On `SIFT`, geometry-aware
  initialization dominates hash-seeded BLISS (`0.928` for `BLISS-KMeans`,
  `0.927` for `MLP-IVF`, `0.676` for `BLISS` at `m=10`). On `GIST` and
  `Deep1M`, hash-seeded BLISS remains competitive or better (`0.834` and
  `0.922` at `m=10`), so there is no dataset-agnostic initialization rule.
- **The low-cap default AdaptIVF controller is underprobed.** Default
  `AdaptIVF` (`m_max=10`) finishes at `0.736 / 0.825 / 0.568 / 0.822`
  Recall@10 on `GloVe / SIFT / GIST / Deep1M`, which is materially below the
  same backbone at `m_max=80`.
- **The `m_max=80` variants are the strongest learned results on three of four
  datasets while remaining compact.** `AdaptIVF-m80` reaches `0.913` on
  `GloVe`, `0.921` on `GIST`, and `0.954` on `Deep1M`, with only
  `19.3 / 17.8 / 150.2 MB` index overhead respectively.
- **LIRA is the compute-efficient learned baseline, but it is structurally
  expensive.** It keeps average computations near `820 / 815 / 883 / 814`
  across the four datasets while its index overhead grows to
  `1.53 / 1.51 / 7.86 / 12.63 GB`.
- **The main unresolved problem is controller calibration and serving
  efficiency, not training cost.** AdaptIVF training is consistently cheap
  (`~0.3–0.5 ks` on `glove/sift/gist` and `~0.4 ks` on `deep1m`), but the
  low-cap default controller leaves too much recall on the table and the
  high-cap controller pays heavily in candidate volume and QPS.

## 1. Initialization Study

At `m=10`, the initialization study isolates the same dataset split that now
drives the paper narrative.

| Dataset | BLISS | BLISS-KMeans | MLP-IVF | Main takeaway |
|---|---:|---:|---:|---|
| GloVe | 0.8188 | 0.8252 | 0.8218 | Minor but real gain from geometry-aware initialization. |
| SIFT | 0.6758 | 0.9281 | 0.9272 | Hash-seeded BLISS is fragile; geometry-aware initialization is decisively better. |
| GIST | 0.8342 | 0.7561 | 0.7518 | Repeated learned set partitions can beat a single coarse geometric scaffold. |
| Deep1M | 0.9222 | 0.8961 | 0.9001 | The same reversal persists at larger scale. |

The paper-safe conclusion is therefore narrow and important: learned
partitioning is **initialization-sensitive**, but the preferred initialization
depends on the dataset rather than obeying a single global rule.

## 2. Final Competitiveness Results

### GloVe

- Best overall recall: `AdaptIVF-m80` at `0.9127`
- Best learned compactness: `AdaptIVF-m80` at `19.29 MB`
- Best learned compute-efficiency: `LIRA` at `819.6` avg computations
- Main contrast: default `AdaptIVF` is weak (`0.7363`), while `m80` recovers
  the strongest learned recall without increasing index overhead materially

### SIFT

- Best overall recall: `HNSW` at `0.9955`
- Best learned recall: `LIRA` at `0.9720`
- Best compact learned method: `AdaptIVF` at `15.75 MB`
- Main contrast: `AdaptIVF-m80` reaches `0.9557`, but `LIRA` remains the best
  learned recall result and `MLP-IVF` remains strong at `0.9288`

### GIST

- Best overall recall: `AdaptIVF-m80` at `0.9212`
- Best learned compactness: `AdaptIVF-m80` at `17.85 MB`
- Best learned compute-efficiency: `LIRA` at `882.6` avg computations
- Main contrast: this is the clearest proof that the default controller is too
  conservative: `AdaptIVF` rises from `0.5679` to `0.9212` when the cap moves
  from `10` to `80`

### Deep1M

- Best overall recall: `HNSW` at `0.9810`
- Best learned recall: `AdaptIVF-m80` at `0.9535`
- Best learned compactness: `AdaptIVF` / `AdaptIVF-m80` at `~150 MB`
- Main contrast: `AdaptIVF-m80` overtakes `LIRA` (`0.9535` vs `0.9317`) while
  staying roughly two orders of magnitude smaller in index overhead

Across the four datasets, the finished run supports a clearer story than the
earlier partial reruns did: the strongest AdaptIVF result is not the low-cap
default controller, but the **high-cap `m80` variant**, which turns the same
single-backbone design into the best learned-recall method on `GloVe`, `GIST`,
and `Deep1M`.

## 3. Compression and PQ

PQ results are now also clearer.

- On `SIFT`, PQ preserves most of the `m80` gain:
  `AdaptIVF-m80 = 0.9557`, `AdaptIVF+PQ-m80 = 0.9545`
- On `Deep1M`, PQ remains strong at high cap:
  `AdaptIVF-m80 = 0.9535`, `AdaptIVF+PQ-m80 = 0.9345`
- On `GloVe`, PQ is tolerable but not neutral:
  `0.9127 -> 0.8006`
- On `GIST`, PQ remains the hardest case:
  `0.9212 -> 0.6609`

So the compression story is not “PQ works” or “PQ fails.” It is:
**PQ is competitive on SIFT and Deep1M, acceptable on GloVe, and still weak on
the hardest high-dimensional GIST setting.**

## 4. Ablation Study

The ablation results are now stable and important.

- `AdaptIVF-Static` beats default `AdaptIVF` on `SIFT`, `GIST`, and `Deep1M`
- `AdaptIVF` is only slightly better on `GloVe`
- `AdaptIVF-A4` does not rescue the low-cap default; it usually increases
  computations without closing the recall gap to the static variant
- The same ordering largely persists under PQ

This means the current entropy-adaptive controller is **not yet the source of
the strongest gains** at the low-cap operating point. The validated part of the
method today is the compact single-backbone design plus learned selective
assignment; the query-time controller still needs better calibration.

## 5. Training and Memory

Training time is not the bottleneck.

- `AdaptIVF` and `AdaptIVF+PQ` train in roughly `300–520 s` on
  `GloVe / SIFT / GIST` and `~400 s` on `Deep1M`
- `BLISS` and `MLP-IVF` are consistently slower (`~1.25–2.05 ks`)
- `LIRA` is dramatically slower and heavier (`1.2 ks` on `GIST`,
  `2.5 ks` on `SIFT`, `12.0 ks` on `Deep1M`, and `22.9 ks` on `GloVe`)

The query-memory measurements also support the systems story.

- `query_mem_delta_mb` is smallest for the compressed learned methods and
  largest for the FAISS graph / flat baselines
- `AdaptIVF` stays low among learned methods:
  `12.1 / 17.0 / 33.6 / 30.2 MB` on `GloVe / SIFT / GIST / Deep1M`
- `AdaptIVF-m80` raises query RAM moderately, not catastrophically:
  `38.2 / 41.0 / 27.0 / 47.3 MB`

The new serving-RAM measurements tell the same story from the steady-state
process side. `AdaptIVF` lands at roughly `498 / 531 / 3705 / 3830 MB`
method-overhead RSS on `GloVe / SIFT / GIST / Deep1M`, versus
`990 / 1022 / 4224 / 4281 MB` for `LIRA`; `IVFPQ` remains the smallest serving
path at about `74 / 70 / 74 / 325 MB`. So the compact single-backbone design is
not only easy to train; it also stays materially lighter than the heaviest
learned baseline after the serving process has fully loaded its artifacts.

So the method is already easy to train and reasonably light on query-path RAM.
The real bottleneck is the quality-versus-work calibration of the probing
policy.

## 6. Paper-Level Takeaway

The finished run changes the best paper framing slightly.

- The contribution is **not** that the default low-cap controller wins
  everywhere; it does not.
- The contribution **is** that a compact single-backbone learned IVF design can
  reach very strong recall with much lower structural overhead than heavier
  learned baselines.
- The main scientific split is now clear:
  - the backbone and selective assignment design are validated
  - the entropy-driven controller remains promising but under-optimized at the
    low-cap default setting

That is still a strong paper. It simply means the strongest AdaptIVF result in
this repository is currently the `m80` competitiveness variant, while the
default `m10` variant functions as the conservative deployment setting and the
ablation target for future controller work.
