# Next Directions

This note narrows follow-up work to the three directions most justified by our
completed large-scale research and the current streamlined benchmark reruns.
The main pattern is already stable, even though the final `deep1m`
competitiveness block is still running:

- AdaptIVF consistently improves over BLISS- and MLP-IVF-style learned routing
  on our four benchmark datasets while using materially lower learned/index
  overhead.
- The strongest remaining weakness is not recall, but serving efficiency:
  runtime and candidate expansion still lag far behind FAISS baselines.
- The uncertainty mechanisms are clearly useful, but a single fixed setting is
  not optimal across easy cosine workloads and harder high-dimensional L2
  workloads.

Across our broader four-dataset research, default AdaptIVF reaches Recall@10 of
roughly `0.908` on Glove, `0.956` on SIFT, `0.948` on GIST, and `0.954` on
Deep1M. The more aggressive A4 variant raises that ceiling further to roughly
`0.928`, `0.969`, `0.966`, and `0.967`, but with substantially higher candidate
cost. Current streamlined reruns on Glove, SIFT, and GIST preserve the same
ranking pattern while keeping learned/index overhead low (`19.3 MB` on Glove,
`15.7 MB` on SIFT, and `17.8 MB` on GIST for AdaptIVF). At the same time, GIST
also makes the current bottleneck unmistakable: very strong recall can still
arrive with extremely large candidate sets and weak QPS.

## 1. Throughput-First Serving Optimization

This is the highest-value direction.

Our research already shows that the method has enough recall headroom to be
interesting. The main systems gap is that no-PQ AdaptIVF still operates far
below FAISS baselines in QPS, especially on the harder datasets. On the current
streamlined reruns, AdaptIVF achieves strong recall on Glove and SIFT but still
trails HNSW and IVF by orders of magnitude in throughput. The completed GIST
rerun makes the same point even more sharply: AdaptIVF reaches the best
finished recall, but only with a very large candidate budget and single-digit
QPS. In other words, the next bottleneck is not finding good buckets; it is
serving those buckets efficiently.

The most promising subdirections are:

- **Batch-adaptive probing.** Keep entropy-based decisions, but group queries
  into a small number of probe-depth bands so the actual scan kernels operate at
  fixed depths per batch rather than fully per-query control flow.
- **Faster reranking and candidate handling.** Focus on candidate deduplication,
  tighter list layouts, fewer temporary allocations, and better SIMD/BLAS paths
  for exact scoring.
- **Bucket-level pruning before full scan.** Use cheap bucket summaries,
  centroid checks, or learned admission rules to cut candidate expansion before
  exact reranking starts.
- **PQ kernel quality, not just PQ availability.** The compressed variant is
  already useful; the next gain is to make ADC and reranking fast enough that PQ
  buys real end-to-end throughput rather than only smaller artifacts.

The success criterion here is straightforward: preserve the current recall
advantage over learned IVF baselines while improving QPS by a meaningful
constant factor at matched operating points.

## 2. Self-Calibrating Uncertainty Control

The current uncertainty mechanisms work, but our research says they should be
calibrated rather than treated as one global fixed recipe.

Two findings make this clear. First, the more aggressive A4 assignment/probing
policy helps substantially on the harder datasets in our broader runs, but it
is not uniformly worth its extra candidate cost on easier workloads. Second, PQ
loss is strongly dataset-dependent: on the current streamlined reruns it is
large on Glove (`0.914 -> 0.810`), small on SIFT (`0.957 -> 0.953`), and harsh
again on GIST (`0.924 -> 0.656`). That is a strong signal that one global
uncertainty policy is leaving performance on the table.

The most promising subdirections are:

- **Dataset-calibrated probe control.** Learn or validate `lambda`, `m_base`,
  `m_max`, and clipping bounds from a small held-out set instead of fixing them
  globally.
- **Adaptive assignment budget.** Choose between the default assignment window
  and a more aggressive setting such as A4 based on validation difficulty,
  rather than hard-coding one variant for every dataset.
- **Entropy-gated compression.** Use uncertainty to decide when exact reranking
  is worth paying for and when PQ is acceptable, instead of applying one
  compression mode uniformly to all queries.
- **Entropy as an operational signal.** Use routing entropy not only to set
  probe budgets, but also to trigger fallback policies, bucket pruning, and
  distribution-shift monitoring in production environments where recall is not
  observable online.

The goal is not to make the method more complex for its own sake. The goal is
to preserve the single-backbone design while making its uncertainty controls
respond to dataset difficulty and query difficulty more intelligently.

## 3. Production-Grade Resource Semantics and Maintenance

The next major improvement is to make the operational story as strong as the
recall story.

Our current research already shows that AdaptIVF has a compelling learned/index
overhead advantage, but it also reveals two practical questions that will keep
coming up: how to report memory fairly across method families, and how the
index should evolve when data changes over time. Those are now more important
than inventing another routing variant.

The most important subdirections are:

- **Dual storage reporting.** Keep the current realized-layout overhead metric,
  but pair it with a companion structure-only view where appropriate so storage
  comparisons are explicit rather than implicit.
- **Unified serving RAM measurement.** Standardize on one isolated post-load or
  peak-RSS definition across all methods, including LIRA, instead of mixing
  pre-load and post-load measurements.
- **Lightweight insert/delete path.** Exploit the single-backbone layout for
  incremental maintenance, with periodic refresh or compaction rather than full
  retraining as the default response to modest data drift.

This direction matters because the paper's long-term value is not just that
AdaptIVF retrieves well, but that it offers a simpler and more controllable
systems design than heavier learned partitioning pipelines. The strongest
production story is not “highest recall at any cost,” but “strong recall with a
compact single-backbone design and an online uncertainty signal that remains
available even when recall cannot be measured.”

## What Not to Prioritize Next

The evidence does **not** currently justify spending the next cycle on:

- adding more repetitions or moving back toward multi-index designs
- expanding the baseline list further before the current run is complete
- introducing new architectural branches before the serving path is faster and
  the uncertainty controls are better calibrated

The method already has enough novelty and enough empirical support. The next
step is to make the strongest version of the existing idea faster, more
self-tuning, and easier to reason about operationally.
