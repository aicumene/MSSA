"""Table `exact count against practice' with 200 trials per row and 95 % Wilson intervals.
Two unit-loading sources at 10 and 13 Hz, m = 8 channels, K = 2000, sampling 100 Hz.  NumPy only.
Signal eigenvalues are computed from the rank-2r factorization H(AS) = (A (x) I_L) [H(s_1); ...; H(s_r)],
noise and total eigenvalues from the Gram matrix of the block-Hankel matrix."""
import numpy as np, time, sys
rng = np.random.default_rng(11)
T = int(sys.argv[1]) if len(sys.argv) > 1 else 200
m, r, K, fs = 8, 2, 2000, 100.0
freqs = [10.0, 13.0]

def hankel(x, L):
    return np.lib.stride_tricks.sliding_window_view(x, L).T                      # L x K

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n; den = 1 + z*z/n; c = (p + z*z/(2*n)) / den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return (max(0.0, c-h), min(1.0, c+h))

def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k:3d}/{n:<3d} [{100*lo:5.1f},{100*hi:5.1f}]" if n else "   --              "

t0 = time.time()
print(f"L SNR_dB beta rho_MP rho_meas(mean) lam_2r(mean) | Weyl holds [95% CI] | exact | practical(kappa=0.15) [95% CI]   (T={T})")
for L in (20, 40, 80, 160):
    N = K + L - 1; t = np.arange(N) / fs; beta = m*L/K
    for snr_db in (-5, 0, 5, 10):
        sigma2 = 10**(-snr_db/10); edge = sigma2*(1+np.sqrt(beta))**2
        weyl = exact = prac = 0; rhos = []; lams = []
        for trial in range(T):
            A = rng.standard_normal((m, r)); A /= np.linalg.norm(A, axis=0)
            S = np.vstack([np.sin(2*np.pi*f*t + rng.uniform(0, 2*np.pi)) for f in freqs])
            # signal eigenvalues through the rank-2r factorization
            HS_small = np.vstack([hankel(S[k], L) for k in range(r)])              # rL x K
            B = np.kron(A, np.eye(L))                                                # mL x rL
            Rb = np.linalg.cholesky(B.T @ B + 1e-12*np.eye(r*L))                     # (B'B) = R R'
            M = Rb.T @ HS_small                                                      # rL x K, same nonzero spectrum as H(AS)
            lam = np.sort(np.linalg.eigvalsh(M @ M.T / K))[::-1]; lam2r = lam[2*r-1]
            E = np.sqrt(sigma2) * rng.standard_normal((m, N))
            HE = np.vstack([hankel(E[j], L) for j in range(m)])
            rho = np.linalg.eigvalsh(HE @ HE.T / K)[-1]
            HX = np.vstack([hankel(A[j] @ S + E[j], L) for j in range(m)])
            ell = np.sort(np.linalg.eigvalsh(HX @ HX.T / K))[::-1]
            rhos.append(rho); lams.append(lam2r)
            if lam2r > 4*rho:
                weyl += 1
                tau2 = 0.5*(rho + (np.sqrt(lam2r) - np.sqrt(rho))**2)
                exact += int(np.sum(ell > tau2) == 2*r)
            prac += int(np.sum(ell > edge*1.15) == 2*r)
        print(f"{L:3d} {snr_db:4d} {beta:5.2f} {edge:6.2f} {np.mean(rhos):8.2f} {np.mean(lams):10.2f} | {fmt(weyl, T)} | {fmt(exact, weyl)} | {fmt(prac, T)}", flush=True)
print(f"elapsed {time.time()-t0:.0f} s")
