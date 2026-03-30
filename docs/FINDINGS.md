# Experimental Findings

> **Status**: Living document. The streamlined competitiveness rerun has
> finished `glove`, `sift`, and `gist`; `deep1m` is still in progress. The
> statements below combine those completed reruns with our broader earlier
> four-dataset research when noted.

## Stable Conclusions

The current evidence already supports four stable conclusions.

- **AdaptIVF is a strong learned-routing method on recall.** On the completed
  competitiveness reruns, it beats `MLP-IVF` and `BLISS` on all three finished
  datasets.
- **AdaptIVF is structurally compact.** Its learned/index overhead remains far
  below `BLISS`, `MLP-IVF`, and especially `LIRA`.
- **LIRA is query-efficient but structurally heavy.** It achieves very low
  average scans by combining learned partition selection with partition-local
  inner indexes, but pays heavily in training time, memory, and storage.
- **Serving efficiency is now the main weakness of AdaptIVF.** The method is
  not candidate-efficient on the harder datasets, and the QPS story is weaker
  than the recall and footprint stories.

## 1. Initialization Study

The initialization study still supports the same mechanistic conclusion:
partition quality matters before any uncertainty-aware routing can help.

At low probe depth (`m=5`) the results are already dataset-dependent:

| Dataset | BLISS | BLISS-KMeans | MLP-IVF | Main takeaway |
|---|---:|---:|---:|---|
| Glove | 0.7140 | 0.7282 | 0.7225 | IVF/KMeans initialization helps, but the gap is modest. |
| SIFT | 0.6374 | 0.8174 | 0.8198 | Hash-style BLISS is fragile; IVF-style initialization is much better. |
| GIST | 0.6757 | 0.5917 | 0.6059 | Initialization sensitivity remains strong in high dimension. |
| Deep1M | 0.8297 | 0.7926 | 0.7989 | No single router wins everywhere; initialization quality gates routing quality. |

Two conclusions matter for the paper:

- The gains do **not** come from “learning” alone; they depend strongly on the
  quality of the coarse partition scaffold.
- Hash-style initialization is especially brittle on the harder L2 datasets.

## 2. Competitiveness Results on Completed Datasets

### Glove

| Method | Recall@10 | Avg computations | QPS | Index overhead (MB) | Train (s) |
|---|---:|---:|---:|---:|---:|
| HNSW | 0.8982 | 4,876.0 | 35,538.3 | 758.6 | — |
| IVF | 0.7494 | 12,340.5 | 56,371.9 | 460.9 | — |
| IVFPQ | 0.3785 | 12,329.3 | 94,350.1 | 27.6 | — |
| BLISS | 0.7980 | 30,561.9 | 41.5 | 45.0 | 1,282.8 |
| MLP-IVF | 0.8217 | 25,001.6 | 55.0 | 45.0 | 1,267.1 |
| MLP-IVFPQ | 0.7597 | 25,007.5 | 6.0 | 63.1 | 1,231.6 |
| LIRA | 0.8733 | 817.1 | 360.0 | 1,531.3 | 46,116.7 |
| AdaptIVF | 0.9142 | 52,352.1 | 49.2 | 19.3 | 312.1 |
| AdaptIVF+PQ | 0.8102 | 54,696.8 | 4.5 | 37.5 | 310.8 |

`AdaptIVF` is the strongest finished learned result on `glove`: higher recall
than `MLP-IVF`, `BLISS`, and `LIRA`, with the smallest learned/index overhead
and far lower training time than `LIRA`.

### SIFT

| Method | Recall@10 | Avg computations | QPS | Index overhead (MB) | Train (s) |
|---|---:|---:|---:|---:|---:|
| HNSW | 0.9957 | 3,009.7 | 45,558.3 | 747.8 | — |
| IVF | 0.8746 | 10,756.3 | 55,243.7 | 496.4 | — |
| IVFPQ | 0.5462 | 10,756.3 | 245,128.6 | 23.5 | — |
| BLISS | 0.7020 | 38,747.2 | 16.0 | 39.6 | 1,284.0 |
| MLP-IVF | 0.9273 | 20,176.5 | 75.5 | 39.6 | 1,281.8 |
| MLP-IVFPQ | 0.9270 | 20,075.6 | 13.1 | 55.0 | 1,293.1 |
| LIRA | 0.9717 | 813.1 | 466.2 | 1,509.4 | 49,528.1 |
| AdaptIVF | 0.9566 | 33,805.4 | 64.3 | 15.7 | 318.9 |
| AdaptIVF+PQ | 0.9533 | 29,952.7 | 11.4 | 31.2 | 317.2 |

On `sift`, `AdaptIVF` again clearly beats `MLP-IVF` and `BLISS` on recall while
remaining much smaller than every other learned method. `LIRA` is still higher
on raw recall, but only with a dramatically heavier pipeline.

### GIST

| Method | Recall@10 | Avg computations | QPS | Index overhead (MB) | Train (s) |
|---|---:|---:|---:|---:|---:|
| HNSW | 0.9193 | 3,769.0 | 9,061.3 | 3,921.6 | — |
| IVF | 0.7040 | 13,966.4 | 5,611.7 | 3,673.5 | — |
| IVFPQ | 0.1710 | 13,966.4 | 56,540.8 | 27.6 | — |
| BLISS | 0.8261 | 35,501.0 | 13.1 | 46.1 | 2,057.2 |
| MLP-IVF | 0.7505 | 35,871.1 | 13.4 | 46.1 | 2,061.3 |
| MLP-IVFPQ | 0.5885 | 35,938.4 | 5.6 | 62.3 | 2,048.6 |
| LIRA | 0.9101 | 869.0 | 144.4 | 7,857.5 | 57,814.6 |
| AdaptIVF | 0.9242 | 103,607.7 | 9.2 | 17.8 | 513.7 |
| AdaptIVF+PQ | 0.6563 | 100,094.4 | 4.1 | 34.1 | 503.9 |

`GIST` is the clearest statement of the current trade-off:

- `AdaptIVF` has the best finished recall.
- `AdaptIVF` remains very compact in learned/index overhead.
- `AdaptIVF` is also extremely expensive in candidate volume, and its QPS is
  much weaker than the recall result alone would suggest.

This is the strongest evidence that the main next problem is serving
efficiency, not retrieval quality.

## 3. What Is Already Stable Before Deep1M Finishes

Even before the current `deep1m` rerun completes, the broader earlier
four-dataset research already points in the same direction:

- default `AdaptIVF` reached roughly `0.908` on `glove`, `0.956` on `sift`,
  `0.948` on `gist`, and `0.954` on `deep1m`
- the more aggressive `A4` variant pushed that to roughly `0.928`, `0.969`,
  `0.966`, and `0.967`, but with substantially larger candidate sets

So the big picture is already stable:

- `AdaptIVF` is recall-strong
- `AdaptIVF` is structurally compact
- `AdaptIVF` is not candidate-efficient in its current form

## 4. Compression Story

The compressed story is now clearly dataset-dependent rather than uniformly
positive or uniformly negative.

- On `glove`, `AdaptIVF+PQ` still beats `MLP-IVFPQ` on recall (`0.810` vs
  `0.760`) while staying smaller.
- On `sift`, the PQ loss is minimal (`0.957 -> 0.953`), and the compressed
  variant remains very competitive.
- On `gist`, compression is much harsher (`0.924 -> 0.656`), so the current PQ
  path is not yet a strong result on the hardest high-dimensional setting.

This strongly suggests that a single uniform compression policy is not ideal.
PQ should be treated as uncertainty- and dataset-dependent rather than global.

## 5. Metric Interpretation That Should Stay Explicit

Two metric caveats are now important enough to document explicitly.

### Index overhead

`index_overhead_mb` is a realized-layout metric, not a pure structure-only
metric.

- externally stored shared vectors such as `train.npy` and `index.npy` are
  excluded
- bytes embedded inside a method's realized index structure remain counted

This is fair as a systems metric, but it is not the same thing as “structure
bytes only.” Paper text should state that clearly.

### Serving RAM

Serving RAM should be described with a unified isolated-process definition, not
with a mix of pre-load and post-load RSS values. The qualitative memory story is
already clear:

- `IVFPQ` is the lightest memory path
- `LIRA` is the heaviest learned method because of its partition-local inner
  indexes
- `AdaptIVF` is the leanest learned-router method, but it is not the lightest
  method overall because the shared vector payload still dominates flat-vector
  approaches on large datasets

## 6. Pending Work

At the time of this update:

- initialization is complete on all four datasets
- competitiveness is complete on `glove`, `sift`, and `gist`
- competitiveness is still running on `deep1m`
- the full ablation stage is still pending

So the document should now be read as:

- **stable on the main method story**
- **not yet final on the last dataset and ablation tables**
