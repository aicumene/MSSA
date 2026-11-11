"""Numerical checks for the additions of version 2 (Theorem on the number of modes, end-to-end accuracy).
Standard library + NumPy only."""
import numpy as np
rng = np.random.default_rng(7)

def block_hankel(X, L):
    m, N = X.shape; K = N - L + 1
    return np.vstack([np.lib.stride_tricks.sliding_window_view(X[j], L).T for j in range(m)])  # (mL) x K

def sources(N, freqs, fs=100.0, phases=None):
    t = np.arange(N) / fs
    phases = rng.uniform(0, 2*np.pi, len(freqs)) if phases is None else phases
    return np.vstack([np.sin(2*np.pi*f*t + p) for f, p in zip(freqs, phases)])

# ---- (a) noise floor of the block-Hankel matrix: E[H H^T/K] = sigma^2 I, deviation ~ 1/sqrt(K)
print("(a) noise floor rho = ||H_E||^2/K for white noise sigma=1, m=8, L=20:")
for K in (250, 1000, 4000, 16000):
    vals = []
    for _ in range(20):
        E = rng.standard_normal((8, K + 19)); H = block_hankel(E, 20)
        vals.append(np.linalg.norm(H, 2)**2 / K)
    print(f"   K={K:6d}: rho mean {np.mean(vals):.3f}, excess over sigma^2 {np.mean(vals)-1:.3f}  (x sqrt(K) = {(np.mean(vals)-1)*np.sqrt(K):.2f})")

# ---- (b) exact recovery of the number of modes by the Weyl interval
print("(b) rank recovery: m=8, r=2 undamped sources (10 and 13 Hz), K=2000; condition lambda_2r > 4 rho")
m, r, K = 8, 2, 2000
for L in (10, 20, 40, 80):
    N = K + L - 1
    for snr_db in (-20, -15, -10, -5):
        ok = 0; cond = 0; ratio = []
        for trial in range(30):
            A = rng.standard_normal((m, r)); A /= np.linalg.norm(A, axis=0)        # unit-norm loadings
            S = sources(N, [10.0, 13.0]); X = A @ S
            sigma = np.sqrt(10**(-snr_db/10))                                        # per-channel noise sd, ||a_k||=1
            E = sigma * rng.standard_normal((m, N))
            HS, HE, HX = block_hankel(X, L), block_hankel(E, L), block_hankel(X + E, L)
            lam = np.linalg.svd(HS, compute_uv=False)**2 / K                          # signal eigenvalues
            rho = np.linalg.norm(HE, 2)**2 / K
            lam2r = lam[2*r-1]; ratio.append(lam2r / (L/4))                          # predicted lambda ~ ||a||^2 L/4
            if lam2r > 4*rho:
                cond += 1
                tau2 = 0.5*(rho + (np.sqrt(lam2r) - np.sqrt(rho))**2)               # midpoint of the admissible interval
                d_hat = int(np.sum(np.linalg.svd(HX, compute_uv=False)**2 / K > tau2))
                ok += (d_hat == 2*r)
        print(f"   L={L:3d} SNR={snr_db:4d} dB: condition holds in {cond:2d}/30 trials, exact rank when it holds {ok}/{cond}; "
              f"lambda_2r / (||a||^2 L/4) = {np.mean(ratio):.2f}; 16/SNR = {16*10**(-snr_db/10):.0f} samples")

# ---- (c) end-to-end accuracy: one source, power-law gamma=1, 3x3 grid of sensors, LS loading at the known frequency
print("(c) end-to-end accuracy, Cov(q) ~ (2 sigma^2/(N alpha^2)) (J' Pi J)^-1:")
P = np.array([[x, y] for x in (-1, 0, 1) for y in (-1, 0, 1)], float)
q0 = np.array([0.35, 0.2]); gamma = 1.0; fs = 100.0
def g_of(q): rho = np.linalg.norm(P - q, axis=1); return rho**(-gamma)
def J_of(q):
    rho = np.linalg.norm(P - q, axis=1); gp = -gamma * rho**(-gamma-1)
    return (gp / rho)[:, None] * (q - P)
g0 = g_of(q0); alpha = 1.0
for N in (500, 2000, 8000):
    for snr_db in (0, 10):
        sigma = np.sqrt(10**(-snr_db/10)) * np.linalg.norm(g0) / np.sqrt(len(P))   # SNR defined on the mean channel power (amplitude^2/2 vs sigma^2): use amplitude scale
        errs = []
        t = np.arange(N) / fs; f = 10.0
        C = np.vstack([np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)]).T                   # N x 2 regressors
        for trial in range(200):
            phi = rng.uniform(0, 2*np.pi)
            X = alpha * np.outer(g0, np.sin(2*np.pi*f*t + phi)) + sigma * rng.standard_normal((len(P), N))
            B = np.linalg.lstsq(C, X.T, rcond=None)[0]                                  # 2 x m amplitudes
            a_hat = np.sqrt(B[0]**2 + B[1]**2)                                          # amplitude per channel (positive loadings)
            # matching: maximise cos^2 between a_hat and g_q by Gauss-Newton from the truth (local accuracy)
            q = q0.copy()
            for it in range(30):
                g = g_of(q); J = J_of(q); Pi = np.eye(len(P)) - np.outer(g, g)/(g@g)
                resid = a_hat - (a_hat@g)/(g@g) * g
                step = np.linalg.lstsq((a_hat@g)/(g@g) * (Pi @ J), resid, rcond=None)[0]
                q = q + step
                if np.linalg.norm(step) < 1e-9: break
            errs.append(np.sum((q - q0)**2))
        rms = np.sqrt(np.mean(errs))
        Pi0 = np.eye(len(P)) - np.outer(g0, g0)/(g0@g0); J0 = J_of(q0)
        sigma_a2 = 2*sigma**2/N
        pred = np.sqrt(sigma_a2/alpha**2 * np.trace(np.linalg.inv(J0.T @ Pi0 @ J0)))
        print(f"   N={N:5d} sigma={sigma:.3f}: RMS error {rms:.4f}, predicted {pred:.4f}, ratio {rms/pred:.2f}")
