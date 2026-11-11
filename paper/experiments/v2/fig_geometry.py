"""Fig.: geometry of the trajectory space behind the count of modes (after the topology-driven classifier).
One tone -> a closed curve in one plane; two tones -> two planes; a damped tone -> a spiral; noise -> an isotropic cloud
of radius sqrt(rho); the eigenvalue spectrum shows 2r eigenvalues above the floor.  NumPy + Matplotlib."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(3)
def hankel(x, L):
    return np.lib.stride_tricks.sliding_window_view(x, L).T          # L x K
def pcs(x, L, k=4):
    H = hankel(x, L); Hc = H - H.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(Hc, full_matrices=False)
    return (U[:, :k].T @ Hc), s**2 / H.shape[1]                       # k x K scores, eigenvalues of H H^T / K
N, L = 256, 16; t = np.arange(N)
one = np.sin(2*np.pi*t/16)
two = np.sin(2*np.pi*t/8) + 0.5*np.sin(2*np.pi*t/16)
damped = np.exp(-0.015*t) * np.sin(2*np.pi*t/16)
noisy = two + 0.4*rng.standard_normal(N)
fig, ax = plt.subplots(2, 3, figsize=(10.5, 6.6), dpi=160)
def plane(a, x, i, j, title, color="C0"):
    P, _ = pcs(x, L); a.plot(P[i], P[j], "-", color=color, lw=0.8, alpha=0.8); a.scatter(P[i], P[j], s=8, color=color)
    a.set_title(title, fontsize=9); a.set_xlabel(f"PC {i+1}", fontsize=8); a.set_ylabel(f"PC {j+1}", fontsize=8); a.set_aspect("equal", adjustable="datalim"); a.tick_params(labelsize=7); a.grid(alpha=.25)
plane(ax[0,0], one, 0, 1, "(a) one tone, period 16: a 16-gon in one plane")
plane(ax[0,1], two, 0, 1, "(b) two tones: first plane, period 8")
plane(ax[0,2], two, 2, 3, "(c) two tones: second plane, period 16", color="C1")
plane(ax[1,0], damped, 0, 1, "(d) damped tone: a spiral", color="C2")
plane(ax[1,1], noisy, 0, 1, "(e) two tones in noise, SNR 8 dB", color="C3")
# spectrum
_, lam_noisy = pcs(noisy, L, k=L); _, lam_clean = pcs(two, L, k=L)
a = ax[1,2]; idx = np.arange(1, L+1)
a.semilogy(idx, lam_noisy, "o-", color="C3", ms=4, lw=0.8, label="two tones + noise")
a.semilogy(idx, np.maximum(lam_clean, 1e-6), "s--", color="C0", ms=3, lw=0.8, label="two tones, no noise")
sigma2 = 0.4**2; beta = L/(N-L+1); rho = sigma2*(1+np.sqrt(beta))**2
a.axhline(rho, color="gray", ls=":", lw=1); a.text(L-0.5, rho*1.25, r"floor $\rho=\sigma_\varepsilon^2(1+\sqrt{\beta})^2$", ha="right", fontsize=7, color="gray")
a.set_title("(f) eigenvalues $\\ell_i$ of $\\mathcal{H}_L \\mathcal{H}_L^{\\top}/K$: four above the floor", fontsize=9)
a.set_xlabel("index $i$", fontsize=8); a.set_ylabel("$\\ell_i$", fontsize=8); a.set_ylim(1e-4, 20); a.tick_params(labelsize=7); a.grid(alpha=.25, which="both"); a.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig("fig6_geometry.png"); print("saved fig6_geometry.png; eigenvalues noisy:", np.round(lam_noisy[:6], 3), "floor", round(rho, 3))
