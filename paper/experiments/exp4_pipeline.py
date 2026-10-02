"""E8: full pipeline for three sources - calibrated rank estimate, then localisation of every mode.
E9: the same recordings under three rules for the number of components."""
import json, time, numpy as np
from loc_common import *
from aicumene_mssa import estimate_rank_bai_yin
rng = np.random.default_rng(404); OUT = {}; t0 = time.time()
N, L = 500, 40; P = GRID3; m = len(P); g = Power(1.0); M = Matcher(P, g, grid2d(0.01, 0.99, 99), [0, 0], [1, 1])
def scene():
    while True:
        Q = np.array([random_source(rng, P) for _ in range(3)])
        if min(np.linalg.norm(Q[i] - Q[k]) for i in range(3) for k in range(i)) >= 0.2: break
    while True:
        f = np.sort(rng.uniform(5, 30, 3))
        if np.min(np.diff(f)) >= 2.0: break
    al = rng.uniform(0.7, 1.3, 3)
    return Q, f, al, (profiles(Q, P, g).T * al[None, :]) @ sources(f, rng.uniform(0, 2 * np.pi, 3), N)
ex = []
for _ in range(400):
    e = estimate_rank_bai_yin(block_hankel(rng.standard_normal((m, N)), L), 0.0); ex.append(e.eigenvalues[0] / e.threshold - 1)
delta = float(np.quantile(ex, 0.95)); print('delta95', round(delta, 3), flush=True)
def rank_mp(H, d):
    e = estimate_rank_bai_yin(H, 0.0); return int(np.sum(e.eigenvalues > e.threshold * (1 + d)))
# ---- E8
res = []
for snr in (-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 10.0, 20.0):
    sigma = noise_sigma(g, snr); ok_rank, n_found, err, pred = [], [], [], []
    for _ in range(300):
        Q, f, al, Xs = scene(); X = Xs + sigma * rng.standard_normal((m, N))
        rank = rank_mp(block_hankel(X, L), delta); ok_rank.append(rank == 6)
        if rank == 0: n_found.append(0); continue
        modes = mssa_modes(X, L, rank)[0]; n_found.append(len(modes))
        if rank == 6 and len(modes) == 3:
            for md in modes:
                k = int(np.argmin(np.abs(f - md['f'])))
                err.append(np.linalg.norm(M.fit(md['a'])[0] - Q[k])); pred.append(np.trace(predicted_cov(Q[k], al[k], P, g, 2 * sigma ** 2 / N)))
    e = np.asarray(err); row = dict(snr0=snr, p_rank=float(np.mean(ok_rank)), p_three_modes=float(np.mean(np.array(n_found) == 3)))
    if len(e):
        row.update(median=float(np.median(e)), gross=float(np.mean(e > 0.1)), rmse=float(np.sqrt(np.mean(e ** 2))),
                   rmse_no_gross=float(np.sqrt(np.mean(e[e <= 0.1] ** 2))) if np.any(e <= 0.1) else None, predicted_rmse=float(np.sqrt(np.mean(pred))))
    res.append(row); print('E8', row, flush=True)
OUT['E8_pipeline'] = dict(delta95=delta, rows=res)
# ---- E9: rules for the number of components on the same kind of recording
B = 30; res = []
for snr in (-15.0, -5.0, 5.0, 20.0):
    sigma = noise_sigma(g, snr); R = {k: [] for k in ('mp_calibrated', 'mp_uncorrected', 'shuffle_any', 'shuffle_leading')}
    for _ in range(100):
        Q, f, al, Xs = scene(); X = Xs + sigma * rng.standard_normal((m, N)); H = block_hankel(X, L)
        R['mp_calibrated'].append(rank_mp(H, delta)); R['mp_uncorrected'].append(rank_mp(H, 0.0))
        s = np.linalg.svd(H, compute_uv=False); S = np.empty((B, len(s)))
        for b in range(B):
            S[b] = np.linalg.svd(block_hankel(np.array([rng.permutation(x) for x in X]), L), compute_uv=False)
        above = s > np.quantile(S, 0.95, axis=0)
        R['shuffle_any'].append(int(above.sum())); R['shuffle_leading'].append(int(np.argmin(above)) if not above.all() else len(s))
    row = dict(snr0=snr, **{k: dict(p_correct=float(np.mean(np.array(v) == 6)), mean=float(np.mean(v)), median=float(np.median(v))) for k, v in R.items()})
    res.append(row); print('E9', row, flush=True)
OUT['E9_rank_rules'] = res
json.dump(OUT, open(os.path.join(HERE, 'results_pipeline.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
