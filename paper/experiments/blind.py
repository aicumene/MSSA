"""Blind spatial separation used as a baseline: FastICA and SOBI (numpy only).

X has shape (n_samples, n_channels). Both return (sources, mixing, unmixing); row k of `mixing` is the
spatial pattern of source k."""
from typing import Optional, Sequence, Tuple
import numpy as np
from numpy.typing import ArrayLike


def _symmetric_decorr(W: np.ndarray) -> np.ndarray:
    """Orthogonalize the rows of W: W <- (W W^T)^{-1/2} W."""
    eig, U = np.linalg.eigh(W @ W.T)
    eig = np.maximum(eig, 1e-12)
    return (U / np.sqrt(eig)) @ U.T @ W


def fastica(
    X: ArrayLike,
    n_components: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FastICA with symmetric decorrelation and cubic non-linearity (numpy-only).

    Spatial unmixing that exploits non-Gaussianity: given mixed observations
    ``X[t, c] = sum_k A[c, k] s_k[t]`` with non-Gaussian sources s_k, recover
    sources up to sign and permutation.

    Parameters
    ----------
    X : array (n_samples, n_features)
    n_components : int
        Number of independent components to extract (<= min(n_samples, n_features)).
    max_iter, tol :
        Iteration cap and convergence tolerance on the decorrelated weight matrix.
    random_state : int | None
        Seed for the random initialisation.

    Returns
    -------
    sources : array (n_samples, n_components)
    mixing : array (n_components, n_features)
        Per-source spatial pattern (row k is source k's channel weights).
    unmixing : array (n_components, n_features)
        sources ≈ (X - X.mean(0)) @ unmixing.T.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    T, C = X.shape
    k = int(n_components)
    if k < 1 or k > min(T, C):
        raise ValueError(f"n_components must be in [1, {min(T, C)}].")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s, 1e-12)
    Xw = U[:, :k] * np.sqrt(T)  # PCA-whitened
    rng = np.random.default_rng(random_state)
    W = _symmetric_decorr(rng.standard_normal(size=(k, k)))
    for _ in range(max_iter):
        Y = Xw @ W.T
        # G(y) = y^3, G'(y) = 3 y^2  -> update W <- E[g(y) x_w] - E[g'(y)] W
        W_new = _symmetric_decorr((Y ** 3).T @ Xw / T - 3 * W)
        lim = float(np.max(np.abs(np.abs(np.einsum("ij,ij->i", W_new, W)) - 1.0)))
        W = W_new
        if lim < tol:
            break
    sources = Xw @ W.T
    K_whiten = (Vt[:k] / s[:k, None]) * np.sqrt(T)  # (k, C)
    unmixing = W @ K_whiten                          # (k, C)
    mixing = np.linalg.pinv(unmixing)                # (C, k)
    return sources, mixing.T, unmixing               # mixing.T (k, C) per-source rows


def _jade_jd(Cs: np.ndarray, *, max_sweeps: int = 100, tol: float = 1e-8) -> np.ndarray:
    """
    Cardoso joint approximate diagonalisation of M symmetric matrices via Givens
    rotations.

    Cs : array (M, k, k), each Cs[m] symmetric.

    Returns V (k, k) orthogonal such that V^T Cs[m] V is approximately diagonal
    for every m. The optimal pair-angle is theta = (1/4) atan2(num, den) where
    num = 4 sum_m ddiag_m off_m, den = sum_m (ddiag_m^2 - 4 off_m^2); for a
    single matrix this reduces to the standard Jacobi rotation.
    """
    Cs = np.asarray(Cs, dtype=float)
    M, k, _ = Cs.shape
    A = Cs.copy()
    V = np.eye(k)
    for _ in range(max_sweeps):
        max_change = 0.0
        for p in range(k - 1):
            for q in range(p + 1, k):
                ddiag = A[:, p, p] - A[:, q, q]
                offd = A[:, p, q]
                # Minimise sum_m (new_off_m)^2 where new_off = (1/2)sin(2θ)ddiag + cos(2θ)off
                # gives tan(4θ) = -4·SC / (SD - 4·SO); for M=1 this reduces to the
                # standard Jacobi rotation zeroing the off-diagonal.
                num = -4.0 * float(np.sum(ddiag * offd))
                den = float(np.sum(ddiag ** 2 - 4.0 * offd ** 2))
                theta = 0.25 * np.arctan2(num, den)
                if abs(theta) < tol:
                    continue
                c, s = np.cos(theta), np.sin(theta)
                # G^T A G where G = I except [[c, s], [-s, c]] in rows/cols (p, q)
                cp = A[:, :, p].copy(); cq = A[:, :, q].copy()
                A[:, :, p] = c * cp - s * cq
                A[:, :, q] = s * cp + c * cq
                rp = A[:, p, :].copy(); rq = A[:, q, :].copy()
                A[:, p, :] = c * rp - s * rq
                A[:, q, :] = s * rp + c * rq
                vp = V[:, p].copy(); vq = V[:, q].copy()
                V[:, p] = c * vp - s * vq
                V[:, q] = s * vp + c * vq
                max_change = max(max_change, abs(theta))
        if max_change < tol:
            break
    return V


def sobi(
    X: ArrayLike,
    n_components: int,
    *,
    lags: Optional[Sequence[int]] = None,
    max_sweeps: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Second-Order Blind Identification (SOBI): spatial unmixing via joint
    diagonalisation of time-lagged covariance matrices.

    Suited to narrowband / temporally-smooth sources (e.g. alpha oscillators)
    where second-order statistics already disambiguate the spatial mixture.
    Deterministic (no random init, unlike FastICA).

    Parameters
    ----------
    X : array (n_samples, n_features)
    n_components : int
    lags : sequence[int] | None
        Lags (in samples) at which to compute the covariance matrices; defaults
        to a logarithmic spread [1, 2, 4, 8, 16, 32].
    max_sweeps, tol :
        Joint-diagonalisation iteration cap and tolerance.

    Returns
    -------
    sources, mixing, unmixing : same shapes/conventions as :func:`fastica`.
    """
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    T, C = X.shape
    k = int(n_components)
    if k < 1 or k > min(T, C):
        raise ValueError(f"n_components must be in [1, {min(T, C)}].")
    if lags is None:
        lags = [l for l in (1, 2, 4, 8, 16, 32) if l < T // 2]
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s, 1e-12)
    Xw = U[:, :k] * np.sqrt(T)
    Cs = []
    for l in lags:
        cv = Xw[:-l].T @ Xw[l:] / (T - l)
        Cs.append((cv + cv.T) / 2)
    if not Cs:
        raise ValueError("All requested lags exceed T/2.")
    V = _jade_jd(np.stack(Cs, axis=0), max_sweeps=max_sweeps, tol=tol)
    sources = Xw @ V                                  # rotation in whitened space
    K_whiten = (Vt[:k] / s[:k, None]) * np.sqrt(T)
    unmixing = V.T @ K_whiten
    mixing = np.linalg.pinv(unmixing)
    return sources, mixing.T, unmixing


