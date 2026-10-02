from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .hankel import block_hankel, diagonal_average_block_hankel

@dataclass(frozen=True)
class MSSAResult:
    hankel: np.ndarray
    eigenvalues: np.ndarray
    singular_values: np.ndarray
    eigenvectors: np.ndarray
    n_channels: int
    n_samples: int
    window_length: int
    delay_samples: int

class MSSA:
    def __init__(self, window_length: int, delay_samples: int = 1):
        self.window_length = int(window_length)
        self.delay_samples = int(delay_samples)
        self.hankel_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or not np.all(np.isfinite(x)):
            raise ValueError("x must be finite with shape (channels, samples).")
        c, n = x.shape
        H = block_hankel(x, self.window_length, self.delay_samples)
        M = H.shape[1]
        scatter = (H @ H.T) / M
        eigvals, eigvecs = np.linalg.eigh(scatter)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order]
        self.hankel_ = H
        self.eigenvalues_ = eigvals
        self.singular_values_ = np.sqrt(eigvals * M)
        self.eigenvectors_ = eigvecs
        self.n_channels_ = c
        self.n_samples_ = n
        return self

    def _check(self):
        if self.hankel_ is None:
            raise RuntimeError("Call fit() first.")

    def result(self):
        self._check()
        return MSSAResult(
            self.hankel_, self.eigenvalues_, self.singular_values_,
            self.eigenvectors_, self.n_channels_, self.n_samples_,
            self.window_length, self.delay_samples
        )

    def signal_subspace(self, rank: int):
        self._check()
        r = int(rank)
        if r < 1 or r > self.eigenvectors_.shape[1]:
            raise ValueError("rank outside valid range.")
        return self.eigenvectors_[:, :r]

    def elementary_hankel(self, component: int):
        self._check()
        i = int(component)
        s = self.singular_values_[i]
        if s == 0:
            return np.zeros_like(self.hankel_)
        u = self.eigenvectors_[:, i]
        v = (self.hankel_.T @ u) / s
        return s * np.outer(u, v)

    def grouped_hankel(self, components):
        self._check()
        H = np.zeros_like(self.hankel_)
        for i in components:
            H += self.elementary_hankel(int(i))
        return H

    def reconstruct(self, components):
        H = self.grouped_hankel(components)
        return diagonal_average_block_hankel(
            H, self.n_channels_, self.n_samples_,
            self.window_length, self.delay_samples
        )
