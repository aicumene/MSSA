"""Window-length check for Fig. 2: for the two UCI trials (channel Pz, N = 256 at 256 Hz, mean removed), the surrogate count
and the ESPRIT poles of the leading conjugate pair for L = 8, 16, 32, 64, and the frequency of the dominant mode by a
least-squares fit of the poles.  Also the periodogram peak below 8 Hz for comparison.  NumPy only."""
import numpy as np, gzip, os, sys
rng = np.random.default_rng(1)
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_uci_eeg", "small", "smni_eeg_data")
FS = 256.0
def read_trial(path):
    vals = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split()
            if len(p) == 4: vals.setdefault(p[1].upper(), []).append(float(p[3]))
    return vals
def hankel(x, L): return np.lib.stride_tricks.sliding_window_view(x, L).T
def eigs(x, L):
    H = hankel(x, L); Hc = H - H.mean(axis=1, keepdims=True); return np.sort(np.linalg.eigvalsh(Hc @ Hc.T / H.shape[1]))[::-1], Hc
def esprit(Hc, r):
    U = np.linalg.svd(Hc, full_matrices=False)[0][:, :r]; return np.linalg.eigvals(np.linalg.lstsq(U[:-1], U[1:], rcond=None)[0])
subs = [("c_1_co2c0000337/co2c0000337.rd.000.gz", "control"), ("a_1_co2a0000364/co2a0000364.rd.000.gz", "alcoholic")]
print("subject  L  d_osc  poles of the signal subspace (Hz, |z|) sorted by fitted energy | periodogram peak < 8 Hz")
for rel, lab in subs:
    w = np.array(read_trial(os.path.join(ROOT, rel))["PZ"]); w = w - w.mean(); N = len(w)
    P = np.abs(np.fft.rfft(w*np.hanning(N)))**2; fr = np.fft.rfftfreq(N, 1/FS); k = np.argmax(P[(fr > 0.5) & (fr < 8)]); fpk = fr[(fr > 0.5) & (fr < 8)][k]
    for L in (8, 16, 32, 64):
        lam, Hc = eigs(w, L)
        sur = np.array([eigs(rng.permutation(w), L)[0] for _ in range(200)]); q95 = np.quantile(sur, 0.95, axis=0); d = int(np.sum(lam > q95))
        r = max(2, min(d, 8)); r += r % 2; z = esprit(Hc, r)
        t = np.arange(1, N + 1); V = z[None, :]**t[:, None]; c = np.linalg.lstsq(V, w.astype(complex), rcond=None)[0]
        pos = [i for i in range(len(z)) if z[i].imag > 1e-6]; en = {i: abs(c[i])**2*np.sum(np.abs(z[i])**(2*t)) for i in pos}
        order = sorted(pos, key=lambda i: -en[i])
        desc = "; ".join(f"{np.angle(z[i])*FS/(2*np.pi):5.1f} Hz |z|={abs(z[i]):.3f}" for i in order[:4])
        print(f"{lab:9s} {L:3d}  {d:3d}   {desc} | {fpk:.1f} Hz")
