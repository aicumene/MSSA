"""Noise-free checks of the statements used in the paper."""
import json, numpy as np
from loc_common import *
rng = np.random.default_rng(1); OUT = {}
N, L = 400, 40; g = Power(1.0); P = GRID3; m = len(P)
def nrank(H): s = np.linalg.svd(H, compute_uv=False); return int(np.sum(s > 1e-9 * s[0]))

# (1) rank of the block-Hankel matrix counts distinct poles, not sources
Q = np.array([[0.3, 0.7], [0.75, 0.3], [0.6, 0.8]]); A = profiles(Q, P, g).T                     # m x 3
r_distinct = nrank(block_hankel(A @ sources([8, 11, 15], [0.3, 1.1, 2.0], N), L))
r_shared = nrank(block_hankel(A @ sources([8, 8, 15], [0.3, 1.4, 2.0], N), L))                    # two sources at 8 Hz
r_damped = nrank(block_hankel(A @ sources([8, 8, 15], [0.3, 1.4, 2.0], N, damping=[0.0, 1.5, 0.5]), L))  # same frequency, different damping
S3 = sources([8, 8, 15], [0.3, 1.4, 2.0], N)
OUT['rank'] = dict(three_distinct=r_distinct, two_share_pole=r_shared, same_freq_diff_damping=r_damped, rank_S_shared=int(np.linalg.matrix_rank(S3)))
print('(1) rank:', OUT['rank'])

# (2) loadings: ESPRIT + least squares is exact; singular pairs mix sources of equal strength
res = []
for ratio in (1.0, 1.05, 1.5, 3.0):
    e_ls, e_svd = [], []
    for _ in range(50):
        Q = np.array([random_source(rng, P), random_source(rng, P)]); A = profiles(Q, P, g).T
        A[:, 1] *= np.linalg.norm(A[:, 0]) / np.linalg.norm(A[:, 1]) / ratio
        X = A @ sources([10, 13], rng.uniform(0, 2 * np.pi, 2), N)
        modes, U, s, _ = mssa_modes(X, L, 4)
        e_ls.append(max(1 - abs(cosine(modes[k]['a'], A[:, k])) for k in range(2)))
        e_svd.append(max(min(1 - cosine(svd_pair_loading(U, s, m, L, pr), np.abs(A[:, k])) for pr in ((0, 1), (2, 3))) for k in range(2)))
    res.append(dict(ratio=ratio, ls_max=float(np.max(e_ls)), svd_median=float(np.median(e_svd)), svd_max=float(np.max(e_svd))))
    print('(2) strength ratio', ratio, '1-cos: ESPRIT+LS max %.1e | SVD pairs median %.1e max %.1e' % (np.max(e_ls), np.median(e_svd), np.max(e_svd)))
OUT['loadings'] = res

# (3) ambiguities of the normalised profile
def inv_point(q, o, R): return o + R ** 2 * (q - o) / np.sum((q - o) ** 2)
th = np.linspace(0, 2 * np.pi, 8, endpoint=False); Pc = np.c_[np.cos(th), np.sin(th)]          # 8 sensors on the unit circle
amb = {}
for gam in (0.5, 1.0, 2.0):
    q = np.array([0.3, 0.2]); qs = inv_point(q, np.zeros(2), 1.0)
    amb[f'circle_power{gam}'] = 1 - cosine(profiles(q, Pc, Power(gam))[0], profiles(qs, Pc, Power(gam))[0])
amb['circle_exponential'] = 1 - cosine(profiles(q, Pc, Exponential(3.0))[0], profiles(qs, Pc, Exponential(3.0))[0])
amb['grid_power1_inverse_point'] = 1 - cosine(profiles(np.array([0.3, 0.7]), P, g)[0], profiles(inv_point(np.array([0.3, 0.7]), np.array([0.5, 0.5]), 0.5), P, g)[0])
P3 = np.array([[0.0, 0.0], [1.0, 0.1], [0.3, 0.9]])                                            # three sensors: circumcircle
a_, b_, c_ = P3; dd = 2 * (a_[0] * (b_[1] - c_[1]) + b_[0] * (c_[1] - a_[1]) + c_[0] * (a_[1] - b_[1]))
o = np.array([((a_ @ a_) * (b_[1] - c_[1]) + (b_ @ b_) * (c_[1] - a_[1]) + (c_ @ c_) * (a_[1] - b_[1])) / dd,
              ((a_ @ a_) * (c_[0] - b_[0]) + (b_ @ b_) * (a_[0] - c_[0]) + (c_ @ c_) * (b_[0] - a_[0])) / dd]); R = np.linalg.norm(a_ - o)
q = np.array([0.45, 0.4]); amb['three_sensors_power1'] = 1 - cosine(profiles(q, P3, g)[0], profiles(inv_point(q, o, R), P3, g)[0])
Pl = np.c_[np.linspace(0, 1, 6), np.zeros(6)]                                                   # collinear sensors: mirror image
q = np.array([0.4, 0.3]); amb['line_mirror_power1'] = 1 - cosine(profiles(q, Pl, g)[0], profiles(q * [1, -1], Pl, g)[0])
amb['line_mirror_exponential'] = 1 - cosine(profiles(q, Pl, Exponential(3.0))[0], profiles(q * [1, -1], Pl, Exponential(3.0))[0])
OUT['ambiguity_1_minus_cos'] = {k: float(v) for k, v in amb.items()}
print('(3) 1 - cos between profiles of the two candidate points:'); [print('    %-28s %.2e' % kv) for kv in amb.items()]

# (3b) exponential law, d + 1 = 3 sensors: is there a second exact solution?
# proportional profiles <=> the distances to the sensors differ by one constant, so compare centred distances
def cdist(p, PP): rho = np.linalg.norm(PP - p[None, :], axis=1); return rho - rho.mean()
found = 0; tried = 0; starts = grid2d(-2.0, 3.0, 21)
for _ in range(200):
    q = rng.uniform(0.1, 0.9, 2); tried += 1; hit = False
    for q0 in starts:
        r = least_squares(lambda p: cdist(p, P3) - cdist(q, P3), q0)
        if r.cost < 1e-24 and np.linalg.norm(r.x - q) > 1e-3: hit = True; break
    found += hit
OUT['exponential_three_sensors_second_solution'] = dict(tried=tried, found=found)
print('(3b) exponential law, 3 sensors in the plane: a second exact solution exists for', found, 'of', tried, 'source positions')
P4 = np.array([[0.0, 0.0], [1.0, 0.1], [0.3, 0.9], [0.9, 0.8]]); found4 = 0
for _ in range(200):
    q = rng.uniform(0.1, 0.9, 2); hit = False
    for q0 in starts:
        r = least_squares(lambda p: cdist(p, P4) - cdist(q, P4), q0)
        if r.cost < 1e-24 and np.linalg.norm(r.x - q) > 1e-3: hit = True; break
    found4 += hit
OUT['exponential_four_sensors_second_solution'] = dict(tried=200, found=found4)
print('(3b) exponential law, 4 sensors in the plane: a second exact solution exists for', found4, 'of 200 source positions')

# (4) two sources under one pole: the loading spans the plane of the two fingerprints; the scan finds both
M = Matcher(P, g, grid2d(0.02, 0.98, 97), [0, 0], [1, 1]); e_plane, e_scan, e_single = [], [], []
for _ in range(100):
    while True:
        Q = np.array([random_source(rng, P), random_source(rng, P)])
        if np.linalg.norm(Q[0] - Q[1]) > 0.25: break
    A = profiles(Q, P, g).T; X = A @ sources([10, 10], [0.4, 0.4 + np.pi / 2], N)
    md = mssa_modes(X, L, 2)[0][0]; W = np.linalg.qr(np.c_[md['a'], md['b']])[0]
    e_plane.append(max(np.linalg.norm(A[:, k] - W @ (W.T @ A[:, k])) / np.linalg.norm(A[:, k]) for k in range(2)))
    Qh = M.scan2(md['a'], md['b'])
    e_scan.append(min(max(np.linalg.norm(Qh[0] - Q[0]), np.linalg.norm(Qh[1] - Q[1])), max(np.linalg.norm(Qh[0] - Q[1]), np.linalg.norm(Qh[1] - Q[0]))))
    e_single.append(min(np.linalg.norm(M.fit(np.abs(md['c']))[0] - Q[k]) for k in range(2)))
OUT['shared_pole'] = dict(plane_residual_max=float(np.max(e_plane)), scan_error_median=float(np.median(e_scan)), scan_error_max=float(np.max(e_scan)),
                          scan_fail_frac=float(np.mean(np.array(e_scan) > 1e-3)), single_fingerprint_error_median=float(np.median(e_single)))
print('(4) shared pole:', OUT['shared_pole'])
# (5) closed-form linear systems of Section 9 (five sensors in the plane)
ep, ee = [], []
for _ in range(200):
    Pm = rng.uniform(0, 1, (5, 2)); q = rng.uniform(0.1, 0.9, 2); gam, mu = 1.5, 3.0; Pj = Pm[1:] - Pm[0]
    a = profiles(q, Pm, Power(gam))[0]; rho = (a[0] / a[1:]) ** (1 / gam)                       # |p_j - q| = rho_j |p_1 - q|
    sol = np.linalg.lstsq(np.c_[-2 * Pj, (1 - rho ** 2)[:, None]], -np.sum(Pj ** 2, axis=1), rcond=None)[0]; ep.append(np.linalg.norm(sol[:2] + Pm[0] - q))
    a = profiles(q, Pm, Exponential(mu))[0]; dl = -np.log(a[1:] / a[0]) / mu                    # |p_j - q| - |p_1 - q| = Delta_j
    sol = np.linalg.lstsq(np.c_[-2 * Pj, (-2 * dl)[:, None]], dl ** 2 - np.sum(Pj ** 2, axis=1), rcond=None)[0]; ee.append(np.linalg.norm(sol[:2] + Pm[0] - q))
OUT['closed_form_max_error'] = dict(power=float(np.max(ep)), exponential=float(np.max(ee)))
print('(5) closed forms, max error:', OUT['closed_form_max_error'])
# (6) Theorem 3: the differential of the normalised map is singular iff the sensors lie on a sphere or hyperplane through q
def smin(q, PP, gg):
    gq = profiles(q, PP, gg)[0]; J = jacobian(q, PP, gg); Pp = np.eye(len(gq)) - np.outer(gq, gq) / (gq @ gq)
    return float(np.linalg.svd(Pp @ J, compute_uv=False)[-1] / np.linalg.norm(gq))
oc = np.array([0.3, -0.2]); Rc = 0.9; thc = rng.uniform(0, 2 * np.pi, 7); Pq = oc + Rc * np.c_[np.cos(thc), np.sin(thc)]; qc = oc + Rc * np.array([np.cos(1.234), np.sin(1.234)])
Pl2 = np.c_[np.array([-1.0, -0.4, 0.1, 0.5, 1.2]), np.zeros(5)]
u3 = rng.standard_normal((12, 3)); P3d = u3 / np.linalg.norm(u3, axis=1, keepdims=True); q3 = np.array([0.2, -0.3, 0.4])
OUT['differential_smallest_singular_value'] = dict(circle_through_q=smin(qc, Pq, g), line_through_q=smin(np.array([0.8, 0.0]), Pl2, g),
    circle_not_through_q=smin(oc + 0.5 * Rc * np.array([1.0, 0.0]), Pq, g), grid=smin(np.array([0.4, 0.6]), P, g), circle_through_q_exponential=smin(qc, Pq, Exponential(3.0)))
OUT['sphere_3d_inverse_1_minus_cos'] = float(1 - cosine(profiles(q3, P3d, g)[0], profiles(q3 / (q3 @ q3), P3d, g)[0]))
print('(6) differential:', OUT['differential_smallest_singular_value'], '| 3-D sphere:', OUT['sphere_3d_inverse_1_minus_cos'])
json.dump(OUT, open(os.path.join(HERE, 'results_exact.json'), 'w'), indent=1)
