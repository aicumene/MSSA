"""Fig.: the geometry of Fig. 1 on real EEG, the data and the parameters of the topology-driven classifier.
UCI EEG Database (Begleiter; doi:10.24432/C5TS3D), small set `smni_eeg_data`: control subject co2c0000337 and alcoholic
subject co2a0000364, trial 0 of the single-stimulus condition, channel Pz, 256 samples at 256 Hz (1 s), embedding window
n = L = 16, mean removed.  Planes of principal components (1,2) and (3,4), and the eigenvalue spectrum against the surrogate
(permutation) null.  Usage: python fig_uci_geometry.py [path to smni_eeg_data]  (default ../data_uci_eeg/small/smni_eeg_data;
download smni_eeg_data.tar.gz from https://archive.ics.uci.edu/dataset/121 and extract the a_1_* and c_1_* archives)."""
import numpy as np, gzip, os, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(1)
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_uci_eeg", "small", "smni_eeg_data")
def read_trial(path):
    vals = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split()
            if len(p) == 4: vals.setdefault(p[1], []).append(float(p[3]))
    return vals
def hankel(x, L): return np.lib.stride_tricks.sliding_window_view(x, L).T
def pcs(x, L):
    H = hankel(x, L); Hc = H - H.mean(axis=1, keepdims=True)
    U, s, _ = np.linalg.svd(Hc, full_matrices=False); return U.T @ Hc, s**2/H.shape[1]
L, CH = 16, "PZ"
subs = [("c_1_co2c0000337/co2c0000337.rd.000.gz", "control subject, co2c0000337"), ("a_1_co2a0000364/co2a0000364.rd.000.gz", "alcoholic subject, co2a0000364")]
fig, ax = plt.subplots(2, 4, figsize=(13.2, 6.4), dpi=160)
for row, (rel, label) in enumerate(subs):
    w = np.array(read_trial(os.path.join(ROOT, rel))[CH]); N = len(w); w = w - w.mean()
    P, lam = pcs(w, L)
    sur = np.array([pcs(rng.permutation(w), L)[1] for _ in range(300)]); q95 = np.quantile(sur, 0.95, axis=0); d_osc = int(np.sum(lam > q95))
    col = "C0" if row == 0 else "C3"; tt = np.arange(N)/256.0; lab = "abcdefgh"
    a = ax[row, 0]; a.plot(tt, w, color="k", lw=0.8); a.set_title(f"({lab[4*row]}) {label}: Pz, trial 0, 1 s", fontsize=9)
    a.set_xlabel("time, s", fontsize=8); a.set_ylabel("$\\mu$V", fontsize=8); a.tick_params(labelsize=7); a.grid(alpha=.25)
    for k, (i, j) in enumerate(((0, 1), (2, 3))):
        a = ax[row, 1+k]; a.plot(P[i], P[j], "-", color=col, lw=0.7, alpha=0.8); a.scatter(P[i], P[j], s=6, color=col)
        a.set_title(f"({lab[4*row+1+k]}) plane of PC {i+1}, {j+1}: $\\ell_{{{i+1}}}={lam[i]:.0f}$, $\\ell_{{{j+1}}}={lam[j]:.0f}$", fontsize=9)
        a.set_xlabel(f"PC {i+1}", fontsize=8); a.set_ylabel(f"PC {j+1}", fontsize=8); a.set_aspect("equal", adjustable="datalim"); a.tick_params(labelsize=7); a.grid(alpha=.25)
    a = ax[row, 3]; idx = np.arange(1, L+1)
    a.semilogy(idx, lam, "o-", color=col, ms=4, lw=0.8, label="eigenvalues $\\ell_i$ of the trial")
    a.semilogy(idx, q95, "s--", color="gray", ms=3, lw=0.8, label="95 % quantile of 300 surrogates")
    a.set_title(f"({lab[4*row+3]}) spectrum: $\\widehat{{d}}_{{\\mathrm{{osc}}}} = {d_osc}$", fontsize=9)
    a.set_xlabel("index $i$", fontsize=8); a.set_ylabel("$\\ell_i$, $\\mu$V$^2$", fontsize=8); a.tick_params(labelsize=7); a.grid(alpha=.25, which="both"); a.legend(fontsize=7, loc="upper right")
    print(f"{label}: N={N}, sd {w.std():.1f} uV, d_osc={d_osc}, eigenvalues {np.round(lam[:6],1)}, surrogate q95 {np.round(q95[:4],1)}")
fig.tight_layout(); out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig7_uci_geometry.png"); fig.savefig(out); print("saved", out)
