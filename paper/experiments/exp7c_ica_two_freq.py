"""E12c: the case in which blind separation should work - two sources of different frequency (10 and 13 Hz)."""
import json, time, numpy as np
from loc_common import *
from blind import fastica, sobi
rng = np.random.default_rng(727); t0 = time.time()
N, L = 500, 40; P = GRID3; m = len(P); g = Power(1.0); M = Matcher(P, g, grid2d(0.01, 0.99, 99), [0, 0], [1, 1])
def bandpass(X, lo, hi): return sosfiltfilt(butter(4, [lo, hi], btype='band', fs=FS, output='sos'), X, axis=1)
def pair_error(Qh, Q):
    if len(Qh) < 2: return 1.0
    return min(max(np.linalg.norm(Qh[0] - Q[0]), np.linalg.norm(Qh[1] - Q[1])), max(np.linalg.norm(Qh[0] - Q[1]), np.linalg.norm(Qh[1] - Q[0])))
def stats(e): e = np.asarray(e); return dict(median=float(np.median(e)), gross=float(np.mean(e > 0.1)))
res = []
for snr in (10.0, 0.0):
    sigma = noise_sigma(g, snr); E = {k: [] for k in ('modes', 'ica', 'sobi')}
    for _ in range(200):
        while True:
            Q = np.array([random_source(rng, P), random_source(rng, P)])
            if np.linalg.norm(Q[0] - Q[1]) >= 0.25: break
        A = profiles(Q, P, g).T; A[:, 1] *= np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1])
        X = A @ sources([10.0, 13.0], rng.uniform(0, 2 * np.pi, 2), N) + sigma * rng.standard_normal((m, N))
        E['modes'].append(pair_error([M.fit(md['a'])[0] for md in mssa_modes(X, L, 4)[0][:2]], Q)); Xb = bandpass(X, 8.0, 15.0)
        for name, fn in (('ica', lambda Z, k: fastica(Z, k, random_state=int(rng.integers(1 << 30)))), ('sobi', sobi)):
            _, mix, _ = fn(Xb.T, 2); E[name].append(pair_error([M.fit(p if p.sum() > 0 else -p)[0] for p in mix], Q))
    row = dict(snr0=snr, **{k: stats(v) for k, v in E.items()}); res.append(row); print(snr, {k: (round(row[k]['median'], 4), round(row[k]['gross'], 2)) for k in E}, flush=True)
json.dump(res, open(os.path.join(HERE, 'results_ica_twofreq.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
