"""E10: a weak source beside a strong one - rank rules against the amplitude ratio of the two sources."""
import json, time, numpy as np
from loc_common import *
from aicumene_mssa import estimate_rank_bai_yin
rng = np.random.default_rng(505); t0 = time.time()
N, L = 500, 40; P = GRID3; m = len(P); g = Power(1.0); B = 30
ex = []
for _ in range(400):
    e = estimate_rank_bai_yin(block_hankel(rng.standard_normal((m, N)), L), 0.0); ex.append(e.eigenvalues[0] / e.threshold - 1)
delta = float(np.quantile(ex, 0.95)); res = []
sigma = noise_sigma(g, 0.0)                       # noise fixed: SNR 0 dB for a unit source at the reference distance
for ratio in (1.0, 3.0, 10.0, 30.0, 100.0):       # the first source is `ratio` times stronger than the second
    R = {'mp_calibrated': [], 'shuffle': []}
    for _ in range(100):
        while True:
            Q = np.array([random_source(rng, P), random_source(rng, P)])
            if np.linalg.norm(Q[0] - Q[1]) >= 0.25: break
        A = profiles(Q, P, g).T; A[:, 1] *= np.sqrt(m) / np.linalg.norm(A[:, 1])      # weak source: unit RMS loading
        A[:, 0] *= ratio * np.sqrt(m) / np.linalg.norm(A[:, 0])                        # strong source: ratio times that
        X = A @ sources([10.0, 17.0], rng.uniform(0, 2 * np.pi, 2), N) + sigma * rng.standard_normal((m, N)); H = block_hankel(X, L)
        e = estimate_rank_bai_yin(H, 0.0); R['mp_calibrated'].append(int(np.sum(e.eigenvalues > e.threshold * (1 + delta))))
        s = np.linalg.svd(H, compute_uv=False); S = np.empty((B, len(s)))
        for b in range(B): S[b] = np.linalg.svd(block_hankel(np.array([rng.permutation(x) for x in X]), L), compute_uv=False)
        R['shuffle'].append(int(np.sum(s > np.quantile(S, 0.95, axis=0))))
    row = dict(ratio=ratio, **{k: dict(p_correct=float(np.mean(np.array(v) == 4)), mean=float(np.mean(v))) for k, v in R.items()}); res.append(row); print('E10', row, flush=True)
json.dump(dict(delta95=delta, weak_source_rms_snr_db=float(10 * np.log10(0.5 / sigma ** 2)), rows=res), open(os.path.join(HERE, 'results_dynamic_range.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
