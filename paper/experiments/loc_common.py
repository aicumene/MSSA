"""Shared code for the source-localisation experiments.

Pipeline: block-Hankel embedding -> signal subspace -> poles by shift invariance (ESPRIT)
-> per-channel complex amplitudes by least squares -> signed spatial loading of every mode
-> matching of the loading to attenuation profiles of candidate source positions.
Notation follows the paper: m channels, N samples, window L, K = N - L + 1.
"""
import os, sys, numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import eigh
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))          # the aicumene_mssa package of this repository
FS = 100.0


# ---------------------------------------------------------------- embedding and subspace
def block_hankel(X, L):
    """Channel-major block-Hankel matrix (mL x K); block j is the Hankel matrix of channel j."""
    m, N = X.shape
    W = sliding_window_view(X, L, axis=1)                    # W[j, k, i] = X[j, k + i]
    return np.ascontiguousarray(W.transpose(0, 2, 1)).reshape(m * L, N - L + 1)


def signal_subspace(H, r):
    """Leading r left singular vectors and singular values, through the lagged scatter."""
    S = H @ H.T
    n = S.shape[0]
    w, U = eigh(S, subset_by_index=[n - r, n - 1])
    return U[:, ::-1], np.sqrt(np.maximum(w[::-1], 0.0))


def esprit_poles(U, m, L):
    """Poles from the shift invariance inside every channel block of the signal subspace."""
    r = U.shape[1]
    B = U.reshape(m, L, r)
    up = B[:, :-1, :].reshape(m * (L - 1), r)
    dn = B[:, 1:, :].reshape(m * (L - 1), r)
    return np.linalg.eigvals(np.linalg.lstsq(up, dn, rcond=None)[0])


def mode_amplitudes(X, z):
    """Least-squares complex amplitudes C (m x r) of x_j(t) = sum_p C[j, p] z_p^t, t = 1..N.

    Returns C, the noise variance estimated from the residual, and diag((V^H V)^-1)."""
    m, N = X.shape
    V = z[None, :] ** np.arange(1, N + 1)[:, None]
    C = np.linalg.lstsq(V, X.T.astype(complex), rcond=None)[0]
    res = X.T - (V @ C).real
    s2 = float(np.sum(res ** 2) / (m * (N - len(z))))
    v = np.real(np.diag(np.linalg.inv(V.conj().T @ V)))
    return C.T, s2, v


def positive_modes(z, tol=1e-6):
    """Indices of the poles with positive frequency, sorted by frequency."""
    idx = [i for i in range(len(z)) if z[i].imag > tol]
    return sorted(idx, key=lambda i: np.angle(z[i]))


def signed_loading(c):
    """Common phase of a complex loading; returns in-phase part a, quadrature part b (both real)."""
    th = 0.5 * np.angle(np.sum(c ** 2))
    w = c * np.exp(-1j * th)
    a, b = 2 * w.real, 2 * w.imag
    if a.sum() < 0:
        a, b = -a, -b
    return a, b


def mssa_modes(X, L, rank):
    """Poles, complex loadings and noise estimate of the modes of a recording."""
    m, N = X.shape
    U, s = signal_subspace(block_hankel(X, L), rank)
    z = esprit_poles(U, m, L)
    C, s2, v = mode_amplitudes(X, z)
    out = []
    for i in positive_modes(z):
        a, b = signed_loading(C[:, i])
        out.append(dict(z=z[i], f=np.angle(z[i]) * FS / (2 * np.pi), c=C[:, i], a=a, b=b,
                        var_a=2 * v[i] * s2,                          # variance of every entry of a and b
                        T=float(b @ b / (2 * v[i] * s2))))            # ~ chi2(m-1) for one in-phase source
    return out, U, s, s2


def phase_test(T, m, alpha=0.05):
    """True when the quadrature part is too large for a single in-phase source."""
    return T > chi2.ppf(1 - alpha, m - 1)


def svd_pair_loading(U, s, m, L, pair):
    """Naive loading: energy of a pair of singular components in every channel block."""
    B = U.reshape(m, L, -1)
    return np.sqrt(sum((s[i] ** 2) * np.sum(B[:, :, i] ** 2, axis=1) for i in pair))


def bandpower_loading(X, f, half=1.0):
    """Classical energy fingerprint: amplitude from the RMS of a narrow band around a known frequency."""
    sos = butter(2, [max(f - half, 0.2), f + half], btype='band', fs=FS, output='sos')
    return np.sqrt(2.0) * sosfiltfilt(sos, X, axis=1).std(axis=1)


# ---------------------------------------------------------------- attenuation models
class Power:
    def __init__(self, gamma=1.0): self.gamma = gamma
    def __call__(self, rho): return rho ** (-self.gamma)
    def d(self, rho): return -self.gamma * rho ** (-self.gamma - 1)


class Exponential:
    def __init__(self, mu=3.0): self.mu = mu
    def __call__(self, rho): return np.exp(-self.mu * rho)
    def d(self, rho): return -self.mu * np.exp(-self.mu * rho)


RHO_MIN = 1e-6          # a candidate position is never closer than this to a sensor


def profiles(Q, P, g):
    """Attenuation profiles of the positions Q (n x d) at the sensors P (m x d): n x m."""
    Q = np.atleast_2d(Q)
    return g(np.maximum(np.linalg.norm(Q[:, None, :] - P[None, :, :], axis=2), RHO_MIN))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class Matcher:
    """Fingerprint matching: grid search for the best cosine, then least squares in (q, alpha)."""
    def __init__(self, P, g, grid, lo, hi):
        self.P, self.g, self.grid, self.lo, self.hi = P, g, grid, np.asarray(lo, float), np.asarray(hi, float)
        G = profiles(grid, P, g)
        self.Gn = G / np.linalg.norm(G, axis=1, keepdims=True)

    def fit(self, a):
        q0 = self.grid[np.argmax(self.Gn @ a)]
        g0 = profiles(q0, self.P, self.g)[0]
        al0 = max(a @ g0 / (g0 @ g0), 1e-12)
        r = least_squares(lambda p: a - p[-1] * profiles(p[:-1], self.P, self.g)[0], np.r_[q0, al0],
                          bounds=(np.r_[self.lo, 0.0], np.r_[self.hi, np.inf]))
        q, al = r.x[:-1], r.x[-1]
        return q, al, 1.0 - cosine(a, profiles(q, self.P, self.g)[0]) ** 2

    def scan2(self, a, b):
        """Two sources under one pole: MUSIC scan of the plane span{a, b}, then joint refinement."""
        W = np.linalg.qr(np.c_[a, b])[0]                                  # orthonormal basis of the plane
        J = 1.0 - np.sum((self.Gn @ W) ** 2, axis=1)                      # distance of g_q to the plane
        n = int(round(np.sqrt(len(self.grid))))
        Jm = J.reshape(n, n)
        mins = []
        for i in range(n):
            for k in range(n):
                nb = Jm[max(i - 1, 0):i + 2, max(k - 1, 0):k + 2]
                if Jm[i, k] <= nb.min():
                    mins.append((Jm[i, k], i * n + k))
        mins.sort()
        if len(mins) < 2:
            mins = sorted((J[i], i) for i in np.argsort(J)[:2])
        q0 = np.r_[self.grid[mins[0][1]], self.grid[mins[1][1]]]
        d = self.P.shape[1]
        Y = np.c_[a, b]

        def resid(p):
            G = profiles(p.reshape(2, d), self.P, self.g).T               # m x 2
            return (Y - G @ np.linalg.lstsq(G, Y, rcond=None)[0]).ravel()
        r = least_squares(resid, q0, bounds=(np.r_[self.lo, self.lo], np.r_[self.hi, self.hi]))
        return r.x.reshape(2, d)


def jacobian(q, P, g):
    """d g_q / d q (m x d)."""
    diff = q[None, :] - P
    rho = np.maximum(np.linalg.norm(diff, axis=1), RHO_MIN)
    return (g.d(rho) / rho)[:, None] * diff


def predicted_cov(q, alpha, P, g, var_a):
    """First-order covariance of the position estimate for loading noise of variance var_a per sensor."""
    gq = profiles(q, P, g)[0]
    J = jacobian(q, P, g)
    Pp = np.eye(len(gq)) - np.outer(gq, gq) / (gq @ gq)
    return var_a / alpha ** 2 * np.linalg.inv(J.T @ Pp @ J)


# ---------------------------------------------------------------- simulation
def sources(freqs, phases, N, damping=None):
    """Exponentially modulated sinusoids s_k(t), t = 1..N (rows)."""
    t = np.arange(1, N + 1) / FS
    damping = np.zeros(len(freqs)) if damping is None else np.asarray(damping)
    return np.exp(-damping[:, None] * t[None, :]) * np.sin(2 * np.pi * np.asarray(freqs)[:, None] * t[None, :]
                                                             + np.asarray(phases)[:, None])


def add_noise(Xs, snr_db, rng):
    """White sensor noise of equal variance; SNR is the mean over sensors of signal power to noise power."""
    sigma = np.sqrt(np.mean(Xs ** 2) / 10 ** (snr_db / 10.0))
    return Xs + sigma * rng.standard_normal(Xs.shape), sigma


RHO0 = 0.5              # reference distance for the definition of SNR


def noise_sigma(g, snr0_db, alpha=1.0):
    """Sensor-noise level for a given SNR of a source of strength alpha seen from the reference distance."""
    return alpha * float(g(RHO0)) / np.sqrt(2.0 * 10 ** (snr0_db / 10.0))


def grid2d(lo, hi, n):
    u = np.linspace(lo, hi, n)
    return np.array([(x, y) for x in u for y in u])


def random_source(rng, P, lo=0.1, hi=0.9, clearance=0.12, d=2):
    while True:
        q = rng.uniform(lo, hi, d)
        if np.min(np.linalg.norm(P - q, axis=1)) >= clearance:
            return q


GRID3 = np.array([(x, y) for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])     # 3 x 3 array on the unit square
