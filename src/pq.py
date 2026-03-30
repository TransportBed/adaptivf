from __future__ import annotations

import numpy as np


def pad_to_m(x: np.ndarray, m: int) -> tuple[np.ndarray, int]:
    d = x.shape[1]
    dsub = int(np.ceil(d / m))
    d_pad = dsub * m
    if d_pad == d:
        return x.astype(np.float32, copy=False), d_pad
    pad = np.zeros((x.shape[0], d_pad - d), dtype=np.float32)
    return np.hstack([x.astype(np.float32, copy=False), pad]), d_pad


def train_global_pq_codebooks(
    residual_sample: np.ndarray,
    m: int,
    bits: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    if bits > 8:
        raise ValueError("bits > 8 is not supported in v1.")
    x_pad, d_pad = pad_to_m(residual_sample, m)
    dsub = d_pad // m
    ksub = 1 << bits
    codebooks = np.zeros((m, ksub, dsub), dtype=np.float32)
    for i in range(m):
        sub = x_pad[:, i * dsub : (i + 1) * dsub]
        if sub.shape[0] < ksub:
            reps = int(np.ceil(ksub / max(1, sub.shape[0])))
            tiled = np.tile(sub, (reps, 1))[:ksub]
            codebooks[i] = tiled.astype(np.float32, copy=False)
            continue
        try:
            from sklearn.cluster import MiniBatchKMeans

            km = MiniBatchKMeans(
                n_clusters=ksub,
                random_state=seed + i,
                batch_size=min(4096, max(256, sub.shape[0] // 8)),
                max_iter=80,
                n_init=1,
                verbose=0,
            )
            km.fit(sub)
            codebooks[i] = km.cluster_centers_.astype(np.float32, copy=False)
        except Exception:
            rng = np.random.default_rng(seed + i)
            pick = rng.choice(sub.shape[0], size=ksub, replace=False)
            codebooks[i] = sub[pick].astype(np.float32, copy=False)
    return codebooks, d_pad


def encode_pq_codes(residuals: np.ndarray, codebooks: np.ndarray, d_pad: int) -> np.ndarray:
    m, _ksub, dsub = codebooks.shape
    if residuals.shape[1] != d_pad:
        if residuals.shape[1] > d_pad:
            residuals = residuals[:, :d_pad]
        else:
            pad = np.zeros((residuals.shape[0], d_pad - residuals.shape[1]), dtype=np.float32)
            residuals = np.hstack([residuals, pad])
    codes = np.empty((residuals.shape[0], m), dtype=np.uint8)
    for i in range(m):
        sub = residuals[:, i * dsub : (i + 1) * dsub]
        centers = codebooks[i]
        sub_norm = np.sum(sub * sub, axis=1, keepdims=True)
        ctr_norm = np.sum(centers * centers, axis=1)[None, :]
        dists = sub_norm + ctr_norm - 2.0 * (sub @ centers.T)
        codes[:, i] = np.argmin(dists, axis=1).astype(np.uint8, copy=False)
    return codes


def adc_table(residual_q: np.ndarray, codebooks: np.ndarray, d_pad: int) -> np.ndarray:
    m, _ksub, dsub = codebooks.shape
    q = residual_q.reshape(1, -1).astype(np.float32, copy=False)
    if q.shape[1] != d_pad:
        if q.shape[1] > d_pad:
            q = q[:, :d_pad]
        else:
            pad = np.zeros((1, d_pad - q.shape[1]), dtype=np.float32)
            q = np.hstack([q, pad])
    table = np.empty((m, codebooks.shape[1]), dtype=np.float32)
    for i in range(m):
        q_sub = q[:, i * dsub : (i + 1) * dsub]
        centers = codebooks[i]
        diff = centers - q_sub
        table[i] = np.sum(diff * diff, axis=1)
    return table


def adc_distances(codes: np.ndarray, table: np.ndarray) -> np.ndarray:
    m = table.shape[0]
    out = np.zeros((codes.shape[0],), dtype=np.float32)
    for i in range(m):
        out += table[i, codes[:, i]]
    return out
