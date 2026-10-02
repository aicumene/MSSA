from __future__ import annotations
import numpy as np

def _as_channels_samples(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("Expected (channels, samples).")
    return x

def block_hankel(x, window_length: int, delay_samples: int = 1):
    x = _as_channels_samples(x)
    c, n = x.shape
    L, tau = int(window_length), int(delay_samples)
    if L < 2:
        raise ValueError("window_length must be >= 2.")
    if tau < 1:
        raise ValueError("delay_samples must be >= 1.")
    span = (L - 1) * tau
    M = n - span
    if M <= 0:
        raise ValueError("Time series too short.")
    blocks = []
    for lag in range(L):
        start = span - lag * tau
        blocks.append(x[:, start:start+M])
    return np.vstack(blocks)

def diagonal_average_block_hankel(H, n_channels: int, n_samples: int, window_length: int, delay_samples: int = 1):
    H = np.asarray(H, dtype=float)
    c, n, L, tau = int(n_channels), int(n_samples), int(window_length), int(delay_samples)
    if H.shape[0] != c * L:
        raise ValueError("Unexpected number of block-Hankel rows.")
    span = (L - 1) * tau
    M = n - span
    if H.shape[1] != M:
        raise ValueError("Unexpected number of block-Hankel columns.")
    out = np.zeros((c, n), float)
    counts = np.zeros((c, n), float)
    for lag in range(L):
        block = H[lag*c:(lag+1)*c]
        start = span - lag*tau
        for j in range(M):
            idx = start + j
            out[:, idx] += block[:, j]
            counts[:, idx] += 1
    mask = counts > 0
    out[mask] /= counts[mask]
    return out
