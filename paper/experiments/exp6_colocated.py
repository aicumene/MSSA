"""E11: is spatial distinguishability needed? Two sources of different frequency at one position or at two."""
import json, time, numpy as np
from loc_common import *
rng = np.random.default_rng(606); t0 = time.time()
N, L = 500, 40; P = GRID3; m = len(P); g = Power(1.0); M = Matcher(P, g, grid2d(0.01, 0.99, 99), [0, 0], [1, 1]); sigma = noise_sigma(g, 10.0); res = []
for sep in (0.25, 0.5, 1.0, 15.0):                       # frequency separation in units of FS/N (15 units = 3 Hz)
    for placing in ('same position', 'different positions'):
        E = []
        for _ in range(300):
            q1 = random_source(rng, P)
            if placing == 'same position': q2 = q1
            else:
                while True:
                    q2 = random_source(rng, P)
                    if np.linalg.norm(q1 - q2) >= 0.25: break
            Q = np.array([q1, q2]); A = profiles(Q, P, g).T; A[:, 1] *= np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1]); f0 = rng.uniform(8, 12)
            X = A @ sources([f0, f0 + sep * FS / N], rng.uniform(0, 2 * np.pi, 2), N) + sigma * rng.standard_normal((m, N))
            modes = mssa_modes(X, L, 4)[0]
            if len(modes) < 2: E.append(1.0); continue
            Qh = [M.fit(md['a'])[0] for md in modes[:2]]
            E.append(min(max(np.linalg.norm(Qh[0] - Q[0]), np.linalg.norm(Qh[1] - Q[1])), max(np.linalg.norm(Qh[0] - Q[1]), np.linalg.norm(Qh[1] - Q[0]))))
        row = dict(separation=sep, placing=placing, median=float(np.median(E)), q90=float(np.quantile(E, 0.9)), gross=float(np.mean(np.array(E) > 0.1))); res.append(row); print('E11', row, flush=True)
json.dump(dict(snr0=10.0, rows=res), open(os.path.join(HERE, 'results_colocated.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
