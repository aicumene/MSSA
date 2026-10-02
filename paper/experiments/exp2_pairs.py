"""E2: two sources of different frequency (strength ratio; frequency separation).
E3: two sources of one frequency (phase test, subspace scan)."""
import json, time, itertools, numpy as np
from loc_common import *
rng = np.random.default_rng(202); OUT = {}; t0 = time.time()
N, L = 500, 40; P = GRID3; m = len(P); g = Power(1.0); GRID = grid2d(0.01, 0.99, 99); M = Matcher(P, g, GRID, [0, 0], [1, 1])
def two_sources(min_sep=0.25):
    while True:
        Q = np.array([random_source(rng, P), random_source(rng, P)])
        if np.linalg.norm(Q[0] - Q[1]) >= min_sep: return Q
def pair_error(Qh, Q):
    """Largest error over the two sources under the better of the two assignments."""
    if len(Qh) < 2: return 1.0
    return min(max(np.linalg.norm(Qh[0] - Q[0]), np.linalg.norm(Qh[1] - Q[1])), max(np.linalg.norm(Qh[0] - Q[1]), np.linalg.norm(Qh[1] - Q[0])))
def stats(e): e = np.asarray(e); return dict(median=float(np.median(e)), q90=float(np.quantile(e, 0.9)), gross=float(np.mean(e > 0.1)))
# ---- E2a: strength ratio, 10 and 13 Hz, SNR0 = 10 dB
sigma = noise_sigma(g, 10.0); res = []
for ratio in (1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0):
    E = {'signed': [], 'svd': []}
    for _ in range(300):
        Q = two_sources(); A = profiles(Q, P, g).T; A[:, 1] *= np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1]) / ratio
        X = A @ sources([10, 13], rng.uniform(0, 2 * np.pi, 2), N) + sigma * rng.standard_normal((m, N))
        modes, U, s, _ = mssa_modes(X, L, 4)
        E['signed'].append(pair_error([M.fit(md['a'])[0] for md in modes[:2]], Q))
        E['svd'].append(pair_error([M.fit(svd_pair_loading(U, s, m, L, pr))[0] for pr in ((0, 1), (2, 3))], Q))
    row = dict(ratio=ratio, **{k: stats(v) for k, v in E.items()}); res.append(row); print('E2a', row, flush=True)
OUT['E2a_strength_ratio'] = res
# ---- E2b: frequency separation in units of the Fourier limit FS/N, equal strength
res = []
for snr in (10.0, 20.0):
    sigma = noise_sigma(g, snr)
    for sep in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
        E = []
        for _ in range(300):
            Q = two_sources(); A = profiles(Q, P, g).T; A[:, 1] *= np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1])
            f0 = rng.uniform(8, 12)
            X = A @ sources([f0, f0 + sep * FS / N], rng.uniform(0, 2 * np.pi, 2), N) + sigma * rng.standard_normal((m, N))
            modes = mssa_modes(X, L, 4)[0]
            E.append(pair_error([M.fit(md['a'])[0] for md in modes[:2]], Q))
        row = dict(snr0=snr, separation=sep, **stats(E)); res.append(row); print('E2b', row, flush=True)
OUT['E2b_frequency_separation'] = res
# ---- E3: one frequency. False alarms of the phase test on one source; detection and localisation of two.
res_fa, res = [], []
for snr in (0.0, 10.0, 20.0):
    sigma = noise_sigma(g, snr); fa = []
    for _ in range(1000):
        q = random_source(rng, P); a = profiles(q, P, g)[0]
        X = np.outer(a, sources([10.0], [rng.uniform(0, 2 * np.pi)], N)[0]) + sigma * rng.standard_normal((m, N))
        modes = mssa_modes(X, L, 2)[0]
        if modes: fa.append(phase_test(modes[0]['T'], m))
    res_fa.append(dict(snr0=snr, false_alarm=float(np.mean(fa)))); print('E3 false alarm', res_fa[-1], flush=True)
    for dphi in (10.0, 20.0, 45.0, 90.0):
        det, e_scan, e_single = [], [], []
        for _ in range(300):
            Q = two_sources(); A = profiles(Q, P, g).T; ph = rng.uniform(0, 2 * np.pi)
            X = A @ sources([10.0, 10.0], [ph, ph + np.deg2rad(dphi)], N) + sigma * rng.standard_normal((m, N))
            modes = mssa_modes(X, L, 2)[0]
            if not modes: det.append(False); e_scan.append(1.0); e_single.append(1.0); continue
            md = modes[0]; det.append(bool(phase_test(md['T'], m)))
            e_scan.append(pair_error(list(M.scan2(md['a'], md['b'])), Q))
            qs = M.fit(md['a'])[0]; e_single.append(min(np.linalg.norm(qs - Q[0]), np.linalg.norm(qs - Q[1])))
        row = dict(snr0=snr, dphi=dphi, detected=float(np.mean(det)), scan=stats(e_scan), single=stats(e_single)); res.append(row); print('E3', row, flush=True)
OUT['E3_false_alarm'] = res_fa; OUT['E3_same_frequency'] = res
json.dump(OUT, open(os.path.join(HERE, 'results_pairs.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
