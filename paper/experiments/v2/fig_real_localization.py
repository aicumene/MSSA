"""Fig.: the algorithm of Section 10 on a real recording (UCI EEG database, doi:10.24432/C5TS3D).
Sensors: the 61 EEG electrodes of the 64-channel montage at the ideal 10-10 positions on a unit sphere (X, Y and nd excluded).
Forward model: current dipole in a homogeneous conducting sphere of unit radius, potential on the surface by the Legendre series
V(r) = (1/4 pi) sum_n ((2n+1)/n) rho^(n-1) [ n q_r P_n(cos th) + |q_t| P_n^1(cos th) cos phi ]  (sigma = 1, R = 1), average reference.
Pipeline (Section 10): block-Hankel embedding, surrogate count, ESPRIT poles, least-squares complex loadings, in-phase test,
gain-matrix fit of the loading of the dominant mode on a grid in the ball with local refinement.  Recording: the average of the
10 single-stimulus trials of each of the two subjects of Fig. 2 (control co2c0000337, alcoholic co2a0000364), and every single trial.
NumPy + Matplotlib.  Usage: python fig_real_localization.py [path to smni_eeg_data]"""
import numpy as np, gzip, os, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(3)
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_uci_eeg", "small", "smni_eeg_data")
FS = 256.0

# ------------------------------------------------------------------ electrodes: ideal 10-10 positions on the unit sphere
def sph(theta_deg, phi_deg):
    """polar angle from Cz (deg), azimuth from the front measured towards the left (deg); x right, y anterior, z up"""
    th, ph = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    return np.array([-np.sin(th)*np.sin(ph), np.sin(th)*np.cos(ph), np.cos(th)])
def arc_point(A, M, B, t):
    """point at fraction t (0 at A, 1 at M, 2 at B) of the circle through A, M, B on the unit sphere"""
    n = np.cross(B - A, M - A); n /= np.linalg.norm(n); c = n * (A @ n)        # centre of the circle: projection of the origin on the plane
    u1 = A - c; r = np.linalg.norm(u1); u1 /= r
    v = M - c; u2 = v - (v @ u1)*u1; u2 /= np.linalg.norm(u2); aM = np.arctan2(v @ u2, v @ u1)
    w = B - c; aB = np.arctan2(w @ u2, w @ u1); aB = aB if aB > aM else aB + 2*np.pi
    ang = t*aM if t <= 1 else aM + (t - 1)*(aB - aM)
    p = c + r*(np.cos(ang)*u1 + np.sin(ang)*u2); return p/np.linalg.norm(p)
ring = {"FPZ": 0, "FP1": 18, "AF7": 36, "F7": 54, "FT7": 72, "T7": 90, "TP7": 108, "P7": 126, "PO7": 144, "O1": 162, "OZ": 180}
mid = {"FPZ": 72, "AFZ": 54, "FZ": 36, "FCZ": 18, "CZ": 0, "CPZ": -18, "PZ": -36, "POZ": -54, "OZ": -72}
POS = {}
for k, ph in ring.items():
    POS[k] = sph(72, ph)
    if 0 < ph < 180: POS[k.replace("7", "8").replace("1", "2")] = sph(72, -ph)
for k, th in mid.items(): POS[k] = sph(abs(th), 0 if th >= 0 else 180)
arcs = {"AF": ("AF7", "AFZ", "AF8"), "F": ("F7", "FZ", "F8"), "FC": ("FT7", "FCZ", "FT8"), "C": ("T7", "CZ", "T8"),
        "CP": ("TP7", "CPZ", "TP8"), "P": ("P7", "PZ", "P8"), "PO": ("PO7", "POZ", "PO8")}
for pre, (A, M, B) in arcs.items():
    for frac, odd, even in ((0.25, "5", "6"), (0.5, "3", "4"), (0.75, "1", "2")):
        POS[pre + odd] = arc_point(POS[A], POS[M], POS[B], frac); POS[pre + even] = arc_point(POS[A], POS[M], POS[B], 2 - frac)
EXCL = {"X", "Y", "ND"}

# ------------------------------------------------------------------ forward model: dipole in a homogeneous sphere
NMAX = 60
def legendre_table(x):
    """P_n(x) and P_n'(x) for n = 0..NMAX, x array"""
    P = np.zeros((NMAX + 1,) + x.shape); dP = np.zeros_like(P); P[0] = 1; P[1] = x; dP[1] = 1
    for n in range(1, NMAX):
        P[n+1] = ((2*n + 1)*x*P[n] - n*P[n-1])/(n + 1)
        dP[n+1] = dP[n-1] + (2*n + 1)*P[n]
    return P, dP
def gain(q, E, boundary=True):
    """m x 3 potentials of unit dipoles along x, y, z at position q (|q| < 1), electrodes E (m x 3, unit sphere)"""
    rho = np.linalg.norm(q)
    if rho < 1e-9:
        G = 3/(4*np.pi) * E
    else:
        rq = q/rho; cth = np.clip(E @ rq, -1, 1); sth = np.sqrt(1 - cth**2)
        P, dP = legendre_table(cth); n = np.arange(NMAX + 1)
        fac = ((2*n + 1)/np.maximum(n, 1)) if boundary else np.ones(NMAX + 1); fac[0] = 0
        rad = np.sum((fac*n*rho**np.maximum(n - 1, 0))[:, None]*P, axis=0)/(4*np.pi)                     # coefficient of q_r
        tan = np.sum((fac*rho**np.maximum(n - 1, 0))[:, None]*sth[None]*dP, axis=0)/(4*np.pi)              # coefficient of |q_t| cos(phi)
        Eperp = E - np.outer(cth, rq)                                                                       # r_perp (unnormalized): |r_perp| = sin th
        G = np.outer(rad, rq) + tan[:, None]*Eperp/np.maximum(sth, 1e-12)[:, None]                         # for unit moment e: q_r = e.rq, |q_t| cos phi = e.r_perp/|r_perp|
    return G - G.mean(axis=0, keepdims=True)                                                                # average reference
def gain_check(E):
    """series with boundary factor 1 against the infinite-medium dipole potential, average referenced"""
    err = 0
    for _ in range(5):
        q = rng.uniform(-0.5, 0.5, 3); d = E - q; Ginf = d/np.linalg.norm(d, axis=1)[:, None]**3/(4*np.pi); Ginf -= Ginf.mean(axis=0)
        err = max(err, np.max(np.abs(gain(q, E, boundary=False) - Ginf))/np.max(np.abs(Ginf)))
    return err

# ------------------------------------------------------------------ pipeline of Section 10 (NumPy only)
def read_trial(path):
    vals = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split()
            if len(p) == 4: vals.setdefault(p[1].upper(), []).append(float(p[3]))
    return vals
def block_hankel(X, L):
    return np.vstack([np.lib.stride_tricks.sliding_window_view(X[j], L).T for j in range(X.shape[0])])
def gram_eigs(H): return np.sort(np.linalg.eigvalsh(H.T @ H))[::-1]/H.shape[1]                              # nonzero eigenvalues of H H^T / K
def surrogate_count(X, L, B=100, eta=0.05):
    H = block_hankel(X, L); ell = gram_eigs(H)
    sur = np.array([gram_eigs(block_hankel(np.array([rng.permutation(x) for x in X]), L)) for _ in range(B)])
    q = np.quantile(sur, 1 - eta, axis=0); return int(np.sum(ell > q)), ell, q
def esprit(H, m, L, r):
    U = np.linalg.svd(H, full_matrices=False)[0][:, :r]; Bk = U.reshape(m, L, r)
    up = Bk[:, :-1, :].reshape(m*(L - 1), r); dn = Bk[:, 1:, :].reshape(m*(L - 1), r)
    return np.linalg.eigvals(np.linalg.lstsq(up, dn, rcond=None)[0])
def mode_amplitudes(X, z):
    m, N = X.shape; V = z[None, :]**np.arange(1, N + 1)[:, None]
    C = np.linalg.lstsq(V, X.T.astype(complex), rcond=None)[0]; res = X.T - (V @ C).real
    s2 = float(np.sum(res**2)/(m*(N - len(z)))); v = np.real(np.diag(np.linalg.inv(V.conj().T @ V)))
    return C.T, s2, v
def signed_loading(c):
    th = 0.5*np.angle(np.sum(c**2)); w = c*np.exp(-1j*th); a, b = 2*w.real, 2*w.imag
    return (a, b) if a.sum() >= 0 else (-a, -b)
def chi2_95(k): return k*(1 - 2/(9*k) + 1.6449*np.sqrt(2/(9*k)))**3                                        # Wilson-Hilferty
def modes_of(X, L, rank):
    m, N = X.shape; z = esprit(block_hankel(X, L), m, L, rank); C, s2, v = mode_amplitudes(X, z); out = []
    for i in np.argsort(np.angle(z)):
        if z[i].imag <= 1e-6: continue
        a, b = signed_loading(C[:, i]); va = 2*v[i]*s2
        out.append(dict(f=np.angle(z[i])*FS/(2*np.pi), damp=abs(z[i]), a=a, b=b, energy=a @ a, T=float(b @ b/va), T95=chi2_95(m - 1)))
    return out
def fit_dipole(a, E, grid, bases):
    k = int(np.argmax([np.sum((Bq.T @ a)**2) for Bq in bases])); q = grid[k].copy(); step = 0.05
    def expl(q):
        if np.linalg.norm(q) > 0.92: return -1
        G = gain(q, E); th = np.linalg.lstsq(G, a, rcond=None)[0]; return 1 - np.sum((a - G @ th)**2)/(a @ a)
    best = expl(q)
    while step > 1e-3:
        moved = False
        for d in np.vstack([np.eye(3), -np.eye(3)]):
            e = expl(q + step*d)
            if e > best: best, q, moved = e, q + step*d, True
        if not moved: step /= 2
    G = gain(q, E); th = np.linalg.lstsq(G, a, rcond=None)[0]
    return q, best, th, G @ th
def fit_scalar(a_abs, E, grid, gamma=2.0):
    """distance-only law g = rho^-gamma fitted to the absolute loading: best cosine on the grid"""
    best, bq = -1, None
    for q in grid:
        g = np.linalg.norm(E - q, axis=1)**(-gamma); c = (a_abs @ g)**2/((g @ g)*(a_abs @ a_abs))
        if c > best: best, bq = c, q
    return bq, best

# ------------------------------------------------------------------ data
subs = [("c_1_co2c0000337", "co2c0000337", "control"), ("a_1_co2a0000364", "co2a0000364", "alcoholic")]
L = 32
names0 = None; results = {}
for folder, sid, lab in subs:
    files = sorted(f for f in os.listdir(os.path.join(ROOT, folder)) if f.endswith(".gz"))
    trials = []
    for f in files:
        v = read_trial(os.path.join(ROOT, folder, f)); names = [n for n in v if n not in EXCL]
        if names0 is None: names0 = names
        trials.append(np.array([v[n] for n in names0]))
    trials = np.array(trials)                                                   # 10 x 61 x 256
    results[lab] = dict(trials=trials, sid=sid)
E = np.array([POS[n] for n in names0]); m = len(names0)
print(f"{m} electrodes; forward-model check (series with factor 1 vs infinite medium): max relative error {gain_check(E):.2e}")
u = np.linspace(-0.85, 0.85, 18); grid = np.array([(x, y, z) for x in u for y in u for z in np.linspace(-0.3, 0.85, 12) if x*x + y*y + z*z <= 0.85**2])
bases = [np.linalg.qr(gain(q, E))[0] for q in grid]
print(f"grid of {len(grid)} candidate positions")

def fit_dipole_joint(a, b, E, grid, bases):
    """one dipole position for the complex loading: the moment has an in-phase and a quadrature component"""
    k = int(np.argmax([np.sum((Bq.T @ a)**2) + np.sum((Bq.T @ b)**2) for Bq in bases])); q = grid[k].copy(); step = 0.05
    def expl(q):
        if np.linalg.norm(q) > 0.92: return -1
        Bq = np.linalg.qr(gain(q, E))[0]; return (np.sum((Bq.T @ a)**2) + np.sum((Bq.T @ b)**2))/(a @ a + b @ b)
    best = expl(q)
    while step > 1e-3:
        moved = False
        for dd in np.vstack([np.eye(3), -np.eye(3)]):
            e = expl(q + step*dd)
            if e > best: best, q, moved = e, q + step*dd, True
        if not moved: step /= 2
    return q, best
def analyse(X):
    """count, poles and loadings of a recording (average reference, channel means removed)"""
    X = X - X.mean(axis=0, keepdims=True); X = X - X.mean(axis=1, keepdims=True)
    d, ell, q95 = surrogate_count(X, L)
    rank = int(min(max(d, 2), 16)); rank += rank % 2
    md = [x for x in modes_of(X, L, rank) if 0.5 < x["f"] < 40]
    return dict(d=d, ell=ell, q95=q95, modes=md, X=X)
def fit_band(res, band):
    """the strongest mode of the recording in the band, with its dipole fits; None if the band holds no mode"""
    cand = [x for x in res["modes"] if band[0] <= x["f"] <= band[1]]
    if not cand: return None
    dom = max(cand, key=lambda x: x["energy"])
    qd, expl, th, pred = fit_dipole(dom["a"], E, grid, bases)
    qj, explj = fit_dipole_joint(dom["a"], dom["b"], E, grid, bases)
    quad = dom["b"] @ dom["b"]/(dom["a"] @ dom["a"] + dom["b"] @ dom["b"])
    return dict(dom=dom, q=qd, expl=expl, moment=th, pred=pred, q_joint=qj, expl_joint=explj, quad=quad)
BANDS = {"slow": (0.5, 5.0), "alpha": (8.0, 13.0)}
FEATURED = {"control": "slow", "alcoholic": "slow"}
for lab in ("control", "alcoholic"):
    R = results[lab]; erp = R["trials"].mean(axis=0)
    R["erp"] = analyse(erp); R["erp_fit"] = {b: fit_band(R["erp"], BANDS[b]) for b in BANDS}
    R["single"] = [analyse(T) for T in R["trials"]]
    R["single_fit"] = {b: [f for f in (fit_band(sres, BANDS[b]) for sres in R["single"]) if f is not None] for b in BANDS}
    a = R["erp"]
    print(f"\n{lab} ({R['sid']}), average of {len(R['trials'])} trials: d_osc = {a['d']}; modes (Hz, |z|, energy, T/T95): " +
          "; ".join(f"{x['f']:.1f} Hz |z|={x['damp']:.3f} E={x['energy']:.0f} T={x['T']:.0f}/{x['T95']:.0f}" for x in a["modes"]))
    for b in BANDS:
        F = R["erp_fit"][b]
        if F is None: print(f"   band {b}: no mode in the average trial")
        else: print(f"   band {b}: mode {F['dom']['f']:.1f} Hz, quadrature share {F['quad']:.2f}; dipole on the in-phase part at ({F['q'][0]:+.2f}, {F['q'][1]:+.2f}, {F['q'][2]:+.2f}), "
              f"|q| = {np.linalg.norm(F['q']):.2f}, explained {F['expl']:.3f}, moment direction {np.round(F['moment']/np.linalg.norm(F['moment']), 2)}; "
              f"complex-moment dipole at ({F['q_joint'][0]:+.2f}, {F['q_joint'][1]:+.2f}, {F['q_joint'][2]:+.2f}), explained {F['expl_joint']:.3f}")
        S = R["single_fit"][b]
        if S:
            Q = np.array([x["q"] for x in S]); Fq = np.array([x["dom"]["f"] for x in S]); Ex = np.array([x["expl"] for x in S])
            print(f"      single trials with a mode in the band: {len(S)} of {len(R['trials'])}; frequencies {np.round(Fq, 1)}; explained {np.round(Ex, 2)}; "
                  f"mean position ({Q.mean(0)[0]:+.2f}, {Q.mean(0)[1]:+.2f}, {Q.mean(0)[2]:+.2f}), |mean| = {np.linalg.norm(Q.mean(0)):.2f}, rms spread {np.sqrt(np.mean(np.sum((Q - Q.mean(0))**2, axis=1))):.2f}"
                  + (f", mean to average-trial fit {np.linalg.norm(Q.mean(0) - F['q']):.2f}" if F is not None else ""))
        else: print(f"      single trials: no mode in the band")

# ------------------------------------------------------------------ fits to JSON (for fig_real_3d.py)
import json
def jl(x): return [float(v) for v in np.asarray(x).ravel()]
J = dict(electrodes=dict(names=names0, xyz=[jl(e) for e in E]), subjects={})
for lab in ("control", "alcoholic"):
    R = results[lab]; D = dict(id=R["sid"], d_osc=int(R["erp"]["d"]), modes_hz=[float(x["f"]) for x in R["erp"]["modes"]], bands={},
                               eigenvalues=jl(R["erp"]["ell"][:60]), surrogate_q95=jl(R["erp"]["q95"][:60]),
                               erp=[jl(row) for row in R["erp"]["X"]])
    for b in BANDS:
        F = R["erp_fit"][b]; S = R["single_fit"][b]
        D["bands"][b] = dict(average=None if F is None else dict(f=float(F["dom"]["f"]), q=jl(F["q"]), moment=jl(F["moment"]), explained=float(F["expl"]),
                                                               loading=jl(F["dom"]["a"]), predicted=jl(F["pred"]), quadrature_share=float(F["quad"])),
                             single=[dict(f=float(x["dom"]["f"]), q=jl(x["q"]), moment=jl(x["moment"]), explained=float(x["expl"])) for x in S])
    J["subjects"][lab] = D
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_localization_fits.json"), "w") as fh: json.dump(J, fh, indent=1)
print("fits written to real_localization_fits.json")

# ------------------------------------------------------------------ figure
def proj(E):
    th = np.arccos(np.clip(E[:, 2], -1, 1)); r = th/(np.pi/2); xy = E[:, :2]/np.maximum(np.linalg.norm(E[:, :2], axis=1), 1e-9)[:, None]
    return r[:, None]*xy
P2 = proj(E)
def scalp(ax, vals, title):
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(P2[:, 0], P2[:, 1]); vm = np.max(np.abs(vals))
    cf = ax.tricontourf(tri, vals, levels=np.linspace(-vm, vm, 21), cmap="RdBu_r"); ax.scatter(P2[:, 0], P2[:, 1], s=6, c="k")
    for r in (0.8, 1.0): ax.add_patch(plt.Circle((0, 0), r, fill=False, lw=0.8, color="k"))
    ax.plot([0, -0.06, 0.06, 0], [1.0, 1.0, 1.0, 1.08], color="k", lw=0.8); ax.set_aspect("equal"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.2)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); return cf
fig, ax = plt.subplots(2, 4, figsize=(15.5, 7.6), dpi=160)
lab8 = "abcdefgh"; idx = np.arange(1, 41)
for row, (lab, col) in enumerate((("control", "C0"), ("alcoholic", "C3"))):
    R = results[lab]; res = R["erp"]; b = FEATURED[lab]; F = R["erp_fit"][b]
    a = ax[row, 0]
    a.semilogy(idx, res["ell"][:40], "o-", color=col, ms=3, lw=0.8, label=f"{lab}, average of {len(R['trials'])} trials")
    a.semilogy(idx, res["q95"][:40], "--", color="gray", lw=0.8, label="surrogate 95 % quantile")
    a.set_title(f"({lab8[4*row]}) {lab}: eigenvalues, $L = {L}$, $\\widehat{{d}}_{{\\mathrm{{osc}}}} = {res['d']}$" + "\n" + "modes at " + ", ".join(f"{x['f']:.1f}" for x in res["modes"]) + " Hz", fontsize=7.5)
    a.set_xlabel("index $i$", fontsize=8); a.set_ylabel("$\\ell_i$, $\\mu$V$^2$", fontsize=8); a.tick_params(labelsize=7); a.grid(alpha=.25, which="both"); a.legend(fontsize=7)
    scalp(ax[row, 1], F["dom"]["a"], f"({lab8[4*row+1]}) {lab}: in-phase loading of the {F['dom']['f']:.1f} Hz mode")
    scalp(ax[row, 2], F["pred"], f"({lab8[4*row+2]}) {lab}: one dipole, {100*F['expl']:.0f} % of the loading, $|q| = {np.linalg.norm(F['q']):.2f}$")
    a = ax[row, 3]; a.add_patch(plt.Circle((0, 0), 1, fill=False, lw=0.8, color="k")); a.scatter(E[:, 0], E[:, 1], s=5, c="gray")
    for bname, mk, cc in (("slow", "o", "C2"), ("alpha", "s", "C1")):
        Fb = R["erp_fit"][bname]; S = R["single_fit"][bname]
        if S:
            Q = np.array([x["q"] for x in S]); a.scatter(Q[:, 0], Q[:, 1], s=14, color=cc, marker=mk, alpha=0.6, label=f"{bname} band, {len(S)} single trials")
        if bname == "slow" and Fb is not None:
            a.scatter([Fb["q"][0]], [Fb["q"][1]], s=110, color=cc, marker="*", edgecolor="k", label=f"slow band, average trial, $z = {Fb['q'][2]:+.2f}$")
        if bname == "alpha" and S:
            qm = Q.mean(0); a.scatter([qm[0]], [qm[1]], s=110, color=cc, marker="P", edgecolor="k", label=f"alpha band, mean of the trials, $z = {qm[2]:+.2f}$")
    a.plot([0, -0.06, 0.06, 0], [1.0, 1.0, 1.0, 1.08], color="k", lw=0.8); a.set_aspect("equal"); a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.2)
    a.set_title(f"({lab8[4*row+3]}) {lab}: fitted dipole positions, top view, nose up", fontsize=9); a.set_xlabel("$x$ (right)", fontsize=8); a.set_ylabel("$y$ (anterior)", fontsize=8)
    a.tick_params(labelsize=7); a.grid(alpha=.25); a.legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig8_real_localization.png"); fig.savefig(out); print("saved", out)
