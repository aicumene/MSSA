"""E6: misspecified attenuation exponent and its joint estimation.
E7: current dipole (signed, orientation-dependent gain) against the scalar attenuation model."""
import json, time, numpy as np
from loc_common import *
from aicumene_mssa import estimate_rank_bai_yin
rng = np.random.default_rng(303); OUT = {}; t0 = time.time()
N, L = 500, 40; P = GRID3; m = len(P); GRID = grid2d(0.01, 0.99, 99)
def stats(e): e = np.asarray(e); return dict(median=float(np.median(e)), q90=float(np.quantile(e, 0.9)), gross=float(np.mean(e > 0.1)))

# ---- E6: true exponent 1, assumed exponent varied; then the exponent as a free parameter
g = Power(1.0); sigma = noise_sigma(g, 0.0); GAMS = [0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
Ms = {gm: Matcher(P, Power(gm), GRID, [0, 0], [1, 1]) for gm in GAMS}; E = {gm: [] for gm in GAMS}; Ej, Gj = [], []
for _ in range(300):
    q = random_source(rng, P); a = profiles(q, P, g)[0]
    X = np.outer(a, sources([rng.uniform(8, 12)], [rng.uniform(0, 2 * np.pi)], N)[0]) + sigma * rng.standard_normal((m, N))
    md = mssa_modes(X, L, 2)[0][0]
    for gm in GAMS: E[gm].append(np.linalg.norm(Ms[gm].fit(md['a'])[0] - q))
    best = min(GAMS, key=lambda gm: Ms[gm].fit(md['a'])[2]); q0, al0, _ = Ms[best].fit(md['a'])
    r = least_squares(lambda p: md['a'] - p[2] * profiles(p[:2], P, Power(p[3]))[0], np.r_[q0, al0, best], bounds=([0, 0, 0, 0.2], [1, 1, np.inf, 4.0]))
    Ej.append(np.linalg.norm(r.x[:2] - q)); Gj.append(r.x[3])
OUT['E6_exponent'] = dict(snr0=0.0, assumed={str(gm): stats(E[gm]) for gm in GAMS}, joint=dict(position=stats(Ej), gamma_median=float(np.median(Gj)), gamma_iqr=[float(np.quantile(Gj, 0.25)), float(np.quantile(Gj, 0.75))]))
print('E6', {gm: round(np.median(E[gm]), 4) for gm in GAMS}, 'joint', round(float(np.median(Ej)), 4), 'gamma', np.round(np.quantile(Gj, [0.25, 0.5, 0.75]), 3), flush=True)

# ---- E7: dipole in a homogeneous conductor; 32 sensors on the upper unit hemisphere
ms = 32; i = np.arange(ms); zc = (i + 0.5) / ms; ph = i * np.pi * (3 - np.sqrt(5)); Ps = np.c_[np.sqrt(1 - zc ** 2) * np.cos(ph), np.sqrt(1 - zc ** 2) * np.sin(ph), zc]
def gain(q):                                    # m x 3: potential of a unit dipole along x, y, z
    d = Ps - q[None, :]; return d / np.linalg.norm(d, axis=1)[:, None] ** 3
u = np.linspace(-0.85, 0.85, 18); G3 = np.array([(x, y, z) for x in u for y in u for z in np.linspace(0.05, 0.9, 10) if x * x + y * y + z * z <= 0.85 ** 2])
Qb = [np.linalg.qr(gain(q))[0] for q in G3]                                           # orthonormal bases of the gain subspaces
Msc = Matcher(Ps, Power(2.0), G3, [-1, -1, 0], [1, 1, 1])
def fit_dipole(a):
    k = int(np.argmax([np.sum((B.T @ a) ** 2) for B in Qb]))
    def resid(q): G = gain(q); return a - G @ np.linalg.lstsq(G, a, rcond=None)[0]
    return least_squares(resid, G3[k], bounds=([-1, -1, 0], [1, 1, 1])).x
res = []
for snr in (20.0, 0.0):
    for psi in (0.0, 30.0, 60.0, 90.0):
        Eg, Es = [], []
        for _ in range(200):
            while True:
                q = rng.uniform(-0.7, 0.7, 3); q[2] = abs(q[2])
                if 0.3 <= np.linalg.norm(q) <= 0.7 and q[2] >= 0.15: break
            er = q / np.linalg.norm(q); w = rng.standard_normal(3); et = w - (w @ er) * er; et /= np.linalg.norm(et)
            th = np.cos(np.deg2rad(psi)) * er + np.sin(np.deg2rad(psi)) * et; v = gain(q) @ th
            X, _ = add_noise(np.outer(v, sources([10.0], [rng.uniform(0, 2 * np.pi)], N)[0]), snr, rng)
            md = mssa_modes(X, L, 2)[0][0]
            Eg.append(np.linalg.norm(fit_dipole(md['a']) - q)); Es.append(np.linalg.norm(Msc.fit(np.abs(md['a']))[0] - q))
        row = dict(snr=snr, psi=psi, gain_model=stats(Eg), scalar_model=stats(Es)); res.append(row); print('E7', row, flush=True)
OUT['E7_dipole'] = res

json.dump(OUT, open(os.path.join(HERE, 'results_models.json'), 'w'), indent=1); print('DONE', round(time.time() - t0), 's')
