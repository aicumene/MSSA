"""E12b: the same comparison on long records (60 s), where independence can be estimated."""
import json, time, numpy as np
from loc_common import *
from blind import fastica, sobi
rng = np.random.default_rng(717); t0 = time.time()
P = GRID3; m = len(P); g = Power(1.0); M = Matcher(P, g, grid2d(0.01, 0.99, 99), [0, 0], [1, 1])
def bandpass(X, lo, hi): return sosfiltfilt(butter(4, [lo, hi], btype='band', fs=FS, output='sos'), X, axis=1)
def unit(s): return s / (np.sqrt(2.0) * s.std())
def burst(N):
    t = np.arange(1, N + 1) / FS
    env = np.abs(sosfiltfilt(butter(2, 1.0, fs=FS, output='sos'), rng.standard_normal(N + 400))[200:200 + N]) + 0.02
    return unit(env * np.sin(2 * np.pi * 10.0 * t + np.cumsum(0.06 * rng.standard_normal(N)) + rng.uniform(0, 2 * np.pi)))
def narrow(N): return unit(bandpass(rng.standard_normal((1, N + 400)), 9.0, 11.0)[0, 200:200 + N])
def pair_error(Qh, Q):
    if len(Qh) < 2: return 1.0
    return min(max(np.linalg.norm(Qh[0] - Q[0]), np.linalg.norm(Qh[1] - Q[1])), max(np.linalg.norm(Qh[0] - Q[1]), np.linalg.norm(Qh[1] - Q[0])))
def stats(e): e = np.asarray(e); return dict(median=float(np.median(e)), gross=float(np.mean(e > 0.1)))
def best_corr(Sh, S): c = np.abs(np.corrcoef(np.vstack([Sh, S]))[:2, 2:]); return float(min(max(c[0, 0], c[0, 1]), max(c[1, 0], c[1, 1])))
res = []; sigma = noise_sigma(g, 10.0)
for N in (500, 2000, 6000, 20000):
    for kind, gen in (('independent bursts', burst), ('independent Gaussian narrow-band', narrow)):
        E = {k: [] for k in ('music', 'ica', 'sobi')}; C = {k: [] for k in ('ica', 'sobi')}; corr = []
        for _ in range(100):
            while True:
                Q = np.array([random_source(rng, P), random_source(rng, P)])
                if np.linalg.norm(Q[0] - Q[1]) >= 0.25: break
            A = profiles(Q, P, g).T; A[:, 1] *= np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1]); S = np.array([gen(N), gen(N)]); corr.append(abs(np.corrcoef(S)[0, 1]))
            Xb = bandpass(A @ S + sigma * rng.standard_normal((m, N)), 8.0, 12.0)
            W = np.linalg.svd(Xb, full_matrices=False)[0][:, :2]; E['music'].append(pair_error(list(M.scan2(W[:, 0], W[:, 1])), Q))
            for name, fn in (('ica', lambda Z, k: fastica(Z, k, random_state=int(rng.integers(1 << 30)))), ('sobi', sobi)):
                src, mix, _ = fn(Xb.T, 2); E[name].append(pair_error([M.fit(p if p.sum() > 0 else -p)[0] for p in mix], Q)); C[name].append(best_corr(src.T, S))
        row = dict(N=N, seconds=N / FS, kind=kind, source_correlation=float(np.median(corr)), **{k: stats(v) for k, v in E.items()}, waveform_corr={k: float(np.median(v)) for k, v in C.items()}); res.append(row)
        print(N, kind, 'corr %.2f |' % row['source_correlation'], {k: (round(row[k]['median'], 4), round(row[k]['gross'], 2)) for k in E}, 'waveform corr', {k: round(v, 2) for k, v in row['waveform_corr'].items()}, flush=True)
json.dump(res, open(os.path.join(HERE, 'results_ica_long.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
