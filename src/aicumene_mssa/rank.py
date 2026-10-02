from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

@dataclass(frozen=True)
class RankEstimate:
    rank: int
    threshold: float
    noise_variance: float
    aspect_ratio: float
    mp_median: float
    eigenvalues: np.ndarray

@lru_cache(maxsize=256)
def marchenko_pastur_median(gamma_rounded: float) -> float:
    gamma = float(gamma_rounded)
    if not (0.0 < gamma < 1.0):
        raise ValueError("MP median currently requires 0 < gamma < 1.")
    s = np.sqrt(gamma)
    a = (1.0 - s)**2
    b = (1.0 + s)**2
    def density(x):
        if x <= a or x >= b:
            return 0.0
        return np.sqrt((b-x)*(x-a)) / (2*np.pi*gamma*x)
    def cdf(x):
        val, _ = quad(density, a, x, limit=300)
        return val
    return float(brentq(lambda x: cdf(x)-0.5, a+1e-12, b-1e-12))

def estimate_rank_bai_yin(H, finite_sample_correction: float = 0.0):
    H = np.asarray(H, dtype=float)
    if H.ndim != 2:
        raise ValueError("H must be 2D.")
    n, M = H.shape
    gamma = n/M
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"Requires n/M < 1; got gamma={gamma:.3f}.")
    scatter = (H @ H.T) / M
    eig = np.linalg.eigvalsh(scatter)
    eig = np.sort(np.maximum(eig, 0.0))[::-1]
    mp_med = marchenko_pastur_median(round(float(gamma), 12))
    emp_med = float(np.median(eig))
    floor = np.finfo(float).eps * max(1.0, float(eig[0])) * max(n, M)
    if emp_med <= floor:
        sigma2 = 0.0
        threshold = floor
    else:
        sigma2 = emp_med/mp_med
        threshold = sigma2*(1+np.sqrt(gamma))**2*(1+float(finite_sample_correction))
    return RankEstimate(
        rank=int(np.sum(eig > threshold)),
        threshold=float(threshold),
        noise_variance=float(sigma2),
        aspect_ratio=float(gamma),
        mp_median=float(mp_med),
        eigenvalues=eig,
    )
