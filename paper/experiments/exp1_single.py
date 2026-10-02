"""E1: one source, accuracy of four fingerprints against noise, with the first-order prediction.
E5: accuracy against the number of sensors."""
import json, time, numpy as np
from loc_common import *
rng = np.random.default_rng(101); OUT = {}; t0 = time.time()
N, L = 500, 40; P = GRID3; m = len(P); GRID = grid2d(0.01, 0.99, 99)
def stats(e): e = np.asarray(e); return dict(rmse=float(np.sqrt(np.mean(e ** 2))), median=float(np.median(e)), gross=float(np.mean(e > 0.1)))
SNRS = [-30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 10.0, 20.0]
for name, g in (('power', Power(1.0)), ('exponential', Exponential(3.0))):
    M = Matcher(P, g, GRID, [0, 0], [1, 1]); res = []
    for snr in SNRS:
        sigma = noise_sigma(g, snr); E = {k: [] for k in ('signed', 'modulus', 'svd', 'band')}; pred = []
        for _ in range(300):
            q = random_source(rng, P); a = profiles(q, P, g)[0]; f = rng.uniform(8, 12)
            X = np.outer(a, sources([f], [rng.uniform(0, 2 * np.pi)], N)[0]) + sigma * rng.standard_normal((m, N))
            modes, U, s, s2 = mssa_modes(X, L, 2)
            if modes:
                E['signed'].append(np.linalg.norm(M.fit(modes[0]['a'])[0] - q)); E['modulus'].append(np.linalg.norm(M.fit(np.abs(modes[0]['c']))[0] - q))
            else:
                E['signed'].append(1.0); E['modulus'].append(1.0)
            E['svd'].append(np.linalg.norm(M.fit(svd_pair_loading(U, s, m, L, (0, 1)))[0] - q))
            E['band'].append(np.linalg.norm(M.fit(bandpower_loading(X, f))[0] - q))
            pred.append(np.trace(predicted_cov(q, 1.0, P, g, 2 * sigma ** 2 / N)))
        row = dict(snr0=snr, predicted_rmse=float(np.sqrt(np.mean(pred))), predicted_median=float(np.median(np.sqrt(pred))), **{k: stats(v) for k, v in E.items()})
        res.append(row); print('E1', name, snr, 'pred %.4f' % row['predicted_rmse'], {k: (round(row[k]['rmse'], 4), round(row[k]['median'], 4)) for k in E}, flush=True)
    OUT['E1_' + name] = res
# ---- E5: number of sensors (random arrays in the unit square), power law, SNR0 = 10 dB
g = Power(1.0); sigma = noise_sigma(g, 10.0); res = []
for mm in (3, 4, 5, 6, 9, 16, 25):
    E, pred = [], []
    for _ in range(20):
        pts = []                                  # sensors at least 0.12 apart, drawn one at a time
        while len(pts) < mm:
            p = rng.uniform(0, 1, 2)
            if all(np.linalg.norm(p - o) > 0.12 for o in pts): pts.append(p)
        Pm = np.array(pts)
        M = Matcher(Pm, g, GRID, [0, 0], [1, 1])
        for _ in range(30):
            q = random_source(rng, Pm); a = profiles(q, Pm, g)[0]
            X = np.outer(a, sources([rng.uniform(8, 12)], [rng.uniform(0, 2 * np.pi)], N)[0]) + sigma * rng.standard_normal((mm, N))
            modes = mssa_modes(X, L, 2)[0]
            E.append(np.linalg.norm(M.fit(modes[0]['a'])[0] - q) if modes else 1.0)
            pred.append(np.trace(predicted_cov(q, 1.0, Pm, g, 2 * sigma ** 2 / N)))
    row = dict(m=mm, predicted_median=float(np.median(np.sqrt(pred))), **stats(E)); res.append(row); print('E5', row, flush=True)
OUT['E5_sensors'] = res
json.dump(OUT, open(os.path.join(HERE, 'results_single.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
