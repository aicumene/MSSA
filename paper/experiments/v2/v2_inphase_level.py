"""Empirical level of the in-phase test T = |b|^2 / (2 v sigma^2) against the chi-square(m-1) 95 % point, for one in-phase
source (so that the null holds), with the pole estimated by ESPRIT from the block-Hankel subspace and, for comparison,
with the true pole.  Setting of Section 6.1: 3x3 grid (m = 9), fs = 100 Hz, N = 500, L = 40, g = 1/rho, SNR_0 at rho_0 = 0.5.
Also two in-quadrature sources (the alternative) to report the power.  NumPy only."""
import numpy as np, sys
rng = np.random.default_rng(9)
T_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
P = np.array([[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)]); m = len(P); fs, N, L = 100.0, 500, 40; K = N - L + 1; t = np.arange(N)/fs
def g_of(q): return np.linalg.norm(P - q, axis=1)**(-1.0)
def block_hankel(X): return np.vstack([np.lib.stride_tricks.sliding_window_view(X[j], L).T for j in range(X.shape[0])])
def esprit(H, r):
    U = np.linalg.svd(H, full_matrices=False)[0][:, :r]; B = U.reshape(m, L, r)
    up = B[:, :-1, :].reshape(m*(L-1), r); dn = B[:, 1:, :].reshape(m*(L-1), r); return np.linalg.eigvals(np.linalg.lstsq(up, dn, rcond=None)[0])
def amplitudes(X, z):
    V = z[None, :]**np.arange(1, N+1)[:, None]; C = np.linalg.lstsq(V, X.T.astype(complex), rcond=None)[0]; res = X.T - (V @ C).real
    s2 = float(np.sum(res**2)/(m*(N - len(z)))); v = np.real(np.diag(np.linalg.inv(V.conj().T @ V))); return C.T, s2, v
def signed(c):
    th = 0.5*np.angle(np.sum(c**2)); w = c*np.exp(-1j*th); return 2*w.real, 2*w.imag
def chi2_95(k): return k*(1 - 2/(9*k) + 1.6449*np.sqrt(2/(9*k)))**3
q95 = chi2_95(m - 1)
def random_source():
    while True:
        q = rng.uniform(0.1, 0.9, 2)
        if np.min(np.linalg.norm(P - q, axis=1)) >= 0.12: return q
print(f"nominal level 5 %: chi2_{m-1} 95 % point {q95:.2f}; {T_TRIALS} trials per cell")
print("SNR_0  | one in-phase source: rejection rate, pole by ESPRIT | pole known | two sources in quadrature (power), ESPRIT")
for snr in (0, 10, 20):
    sigma2 = (0.5**(-2))/(2*10**(snr/10)); rej = {"esprit": 0, "known": 0, "alt": 0}
    for _ in range(T_TRIALS):
        q0 = random_source(); f0 = rng.uniform(8, 15); phi = rng.uniform(0, 2*np.pi); a0 = g_of(q0)
        X = a0[:, None]*np.sin(2*np.pi*f0*t[None] + phi) + np.sqrt(sigma2)*rng.standard_normal((m, N))
        z = esprit(block_hankel(X), 2); z = z[np.argmax(z.imag)]; C, s2, v = amplitudes(X, np.array([z, np.conj(z)]))
        a, b = signed(C[:, 0]); rej["esprit"] += (b @ b/(2*v[0]*s2) > q95)
        zk = np.exp(2j*np.pi*f0/fs); C, s2, v = amplitudes(X, np.array([zk, np.conj(zk)])); a, b = signed(C[:, 0]); rej["known"] += (b @ b/(2*v[0]*s2) > q95)
        q1 = random_source(); a1 = g_of(q1)
        X2 = a0[:, None]*np.sin(2*np.pi*f0*t[None] + phi) + a1[:, None]*np.sin(2*np.pi*f0*t[None] + phi + np.pi/2) + np.sqrt(sigma2)*rng.standard_normal((m, N))
        z = esprit(block_hankel(X2), 2); z = z[np.argmax(z.imag)]; C, s2, v = amplitudes(X2, np.array([z, np.conj(z)])); a, b = signed(C[:, 0]); rej["alt"] += (b @ b/(2*v[0]*s2) > q95)
    print(f"{snr:4d} dB | {100*rej['esprit']/T_TRIALS:5.1f} %                                   | {100*rej['known']/T_TRIALS:5.1f} %    | {100*rej['alt']/T_TRIALS:5.1f} %")
