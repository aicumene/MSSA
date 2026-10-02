"""E12: spatial separation of independent components (FastICA, SOBI) against the plane scan of Section 6.2.

Two sources in one band at different positions; four kinds of source signals. Routes:
  mssa  - loadings of the MSSA modes in the band, pooled; their two leading directions span a plane, scanned for two profiles
  music - the same scan in the plane of the two leading spatial singular vectors of the band-passed recording
  ica   - FastICA on the band-passed recording, each spatial pattern matched to one attenuation profile
  sobi  - SOBI likewise
A last case has more sources than sensors."""
import json, time, numpy as np
from loc_common import *
from blind import fastica, sobi
from aicumene_mssa import estimate_rank_bai_yin
rng = np.random.default_rng(707); t0 = time.time()
N, L = 500, 40; P = GRID3; m = len(P); g = Power(1.0); M = Matcher(P, g, grid2d(0.01, 0.99, 99), [0, 0], [1, 1]); t = np.arange(1, N + 1) / FS
ex = []
for _ in range(300):
    e = estimate_rank_bai_yin(block_hankel(rng.standard_normal((m, N)), L), 0.0); ex.append(e.eigenvalues[0] / e.threshold - 1)
KAPPA = float(np.quantile(ex, 0.95))
def bandpass(X, lo, hi): return sosfiltfilt(butter(4, [lo, hi], btype='band', fs=FS, output='sos'), X, axis=1)
def unit(s): return s / (np.sqrt(2.0) * s.std())             # the RMS of a unit sinusoid
def burst():                                                  # 10 Hz rhythm with slowly wandering amplitude and phase
    env = np.abs(sosfiltfilt(butter(2, 1.0, fs=FS, output='sos'), rng.standard_normal(N + 400))[200:200 + N]) + 0.02
    ph = np.cumsum(0.06 * rng.standard_normal(N))
    return unit(env * np.sin(2 * np.pi * 10.0 * t + ph + rng.uniform(0, 2 * np.pi)))
def narrow(): return unit(bandpass(rng.standard_normal((1, N + 400)), 9.0, 11.0)[0, 200:200 + N])
def signals(kind):
    ph = rng.uniform(0, 2 * np.pi)
    if kind == 'sinusoids in quadrature': return sources([10.0, 10.0], [ph, ph + np.pi / 2], N)
    if kind == 'sinusoids 20 degrees apart': return sources([10.0, 10.0], [ph, ph + np.deg2rad(20)], N)
    if kind == 'independent bursts': return np.array([burst(), burst()])
    if kind == 'independent Gaussian narrow-band': return np.array([narrow(), narrow()])
def pair_error(Qh, Q):
    if len(Qh) < 2: return 1.0
    return min(max(np.linalg.norm(Qh[0] - Q[0]), np.linalg.norm(Qh[1] - Q[1])), max(np.linalg.norm(Qh[0] - Q[1]), np.linalg.norm(Qh[1] - Q[0])))
def stats(e): e = np.asarray(e); return dict(median=float(np.median(e)), gross=float(np.mean(e > 0.1)))
def plane_scan(Y):
    W = np.linalg.svd(Y, full_matrices=False)[0][:, :2]; return list(M.scan2(W[:, 0], W[:, 1]))
def route_mssa(X):
    H = block_hankel(X, L); e = estimate_rank_bai_yin(H, 0.0); rank = max(2, int(np.sum(e.eigenvalues > e.threshold * (1 + KAPPA))))
    modes = [md for md in mssa_modes(X, L, rank)[0] if 8.0 <= md['f'] <= 12.0]
    if not modes: return [], rank
    return plane_scan(np.column_stack([v for md in modes for v in (md['a'], md['b'])])), rank
def route_blind(X, fn):
    try: _, mix, _ = fn(bandpass(X, 8.0, 12.0).T, 2)
    except Exception: return []
    return [M.fit(p if p.sum() > 0 else -p)[0] for p in mix]
OUT = {'kappa95': KAPPA}; res = []
for snr in (10.0, 0.0):
    sigma = noise_sigma(g, snr)
    for kind in ('sinusoids in quadrature', 'sinusoids 20 degrees apart', 'independent bursts', 'independent Gaussian narrow-band'):
        E = {k: [] for k in ('mssa', 'music', 'ica', 'sobi')}; ranks = []; corr = []
        for _ in range(200):
            while True:
                Q = np.array([random_source(rng, P), random_source(rng, P)])
                if np.linalg.norm(Q[0] - Q[1]) >= 0.25: break
            A = profiles(Q, P, g).T; A[:, 1] *= np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1]); S = signals(kind); corr.append(abs(np.corrcoef(S)[0, 1]))
            X = A @ S + sigma * rng.standard_normal((m, N))
            Qh, rank = route_mssa(X); ranks.append(rank); E['mssa'].append(pair_error(Qh, Q))
            E['music'].append(pair_error(plane_scan(bandpass(X, 8.0, 12.0)), Q))
            E['ica'].append(pair_error(route_blind(X, lambda Z, k: fastica(Z, k, random_state=int(rng.integers(1 << 30)))), Q))
            E['sobi'].append(pair_error(route_blind(X, sobi), Q))
        row = dict(snr0=snr, kind=kind, source_correlation=float(np.median(corr)), rank_median=float(np.median(ranks)), **{k: stats(v) for k, v in E.items()}); res.append(row)
        print(snr, kind, 'corr %.2f rank %d |' % (row['source_correlation'], row['rank_median']), {k: (round(row[k]['median'], 4), round(row[k]['gross'], 2)) for k in E}, flush=True)
OUT['two_sources_one_band'] = res
# ---- more sources than sensors: twelve sinusoids, nine sensors, 10 dB
sigma = noise_sigma(g, 10.0); hit = {'mssa': [], 'ica': []}
for _ in range(100):
    while True:
        f = np.sort(rng.uniform(5, 40, 12))
        if np.min(np.diff(f)) >= 1.5: break
    Q = np.array([random_source(rng, P) for _ in range(12)]); A = profiles(Q, P, g).T; X = A @ sources(f, rng.uniform(0, 2 * np.pi, 12), N) + sigma * rng.standard_normal((m, N))
    modes = mssa_modes(X, L, 24)[0]; est = [(md['f'], M.fit(md['a'])[0]) for md in modes]
    hit['mssa'].append(np.mean([any(abs(fe - f[k]) < 0.3 and np.linalg.norm(qe - Q[k]) < 0.05 for fe, qe in est) for k in range(12)]))
    _, mix, _ = fastica(X.T, m, random_state=int(rng.integers(1 << 30))); qs = [M.fit(p if p.sum() > 0 else -p)[0] for p in mix]
    hit['ica'].append(np.mean([any(np.linalg.norm(qe - Q[k]) < 0.05 for qe in qs) for k in range(12)]))
OUT['twelve_sources_nine_sensors'] = {k: float(np.mean(v)) for k, v in hit.items()}; print('12 sources, 9 sensors: share located within 0.05:', OUT['twelve_sources_nine_sensors'])
json.dump(OUT, open(os.path.join(HERE, 'results_ica.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
