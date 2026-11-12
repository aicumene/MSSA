"""Checks for the identifiability statements of version 2 (exponential law, three levels; two sources in one mode;
checkable quadric condition; noise-floor convergence).  NumPy only."""
import numpy as np
rng = np.random.default_rng(2)
mu = 3.0
def prof(PP, q): return np.exp(-mu*np.linalg.norm(PP - q, axis=1))
def nrm(v): return v/np.linalg.norm(v)
def sheet_points(q, qp, Delta, n, d):
    """points x with |x-q'| - |x-q| = Delta on the hyperboloid sheet of revolution with foci q, q' (0<|Delta|<|q-q'|)"""
    c = np.linalg.norm(qp - q)/2; a = abs(Delta)/2; b = np.sqrt(c*c - a*a); e1 = (qp - q)/(2*c); mid = (q + qp)/2
    # orthonormal complement of e1
    Q = np.linalg.qr(np.column_stack([e1, rng.standard_normal((d, d-1))]))[0]; E = Q[:, 1:]
    pts = []
    for _ in range(n):
        s = rng.uniform(0.2, 1.5); u = nrm(rng.standard_normal(d-1))
        x_ax = -np.sign(Delta)*a*np.cosh(s)                          # the sheet nearer to q (|x-q'| > |x-q| when Delta>0)
        r = b*np.sinh(s)
        pts.append(mid + x_ax*e1 + r*(E @ u))
    return np.array(pts)

print("(1) the counterexample: d = 2, four sensors on one branch of a hyperbola with foci q, q' -> identical normalized profiles")
q, qp, Delta = np.array([0.0, 0.0]), np.array([1.0, 0.3]), 0.4
S4 = sheet_points(q, qp, Delta, 4, 2)
print("   differences |x-q'|-|x-q| :", np.round(np.linalg.norm(S4-qp, axis=1) - np.linalg.norm(S4-q, axis=1), 12))
print("   collinear triple? min |det| over triples =", min(abs(np.linalg.det(np.column_stack([S4[j]-S4[i], S4[k]-S4[i]]))) for i, j, k in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]).round(4))
print("   ||g_q/|g_q| - g_q'/|g_q'| || =", np.linalg.norm(nrm(prof(S4, q)) - nrm(prof(S4, qp))))

print("(2) sharpness of 2d+2 in the plane: five points lie on one branch of a hyperbola iff the configuration is ambiguous;")
print("    this is an open condition on the five points, so the ambiguous five-sensor configurations have positive measure")
def conic_through(PP):
    V = np.column_stack([PP[:,0]**2, PP[:,0]*PP[:,1], PP[:,1]**2, PP[:,0], PP[:,1], np.ones(len(PP))])
    return np.linalg.svd(V)[2][-1]                                   # a, b, c, d, e, f
def hyperbola_foci(co):
    a, b, c, d, e, f = co
    Q = np.array([[a, b/2], [b/2, c]])
    if np.linalg.det(Q) >= 0: return None                             # not a hyperbola
    x0 = np.linalg.solve(2*Q, -np.array([d, e])); f1 = f + 0.5*(d*x0[0] + e*x0[1])
    ev, R = np.linalg.eigh(Q)
    k = int(np.argmax(-f1*ev > 0))                                     # transverse axis: lambda_k X^2 = -f1 ... with -f1/lambda_k > 0
    A2 = -f1/ev[k]; B2 = f1/ev[1-k]
    if A2 <= 0 or B2 <= 0: return None
    cfoc = np.sqrt(A2 + B2); axis = R[:, k]
    return x0 - cfoc*axis, x0 + cfoc*axis, 2*np.sqrt(A2)
def ambiguous5(PP):
    fo = hyperbola_foci(conic_through(PP))
    if fo is None: return False
    q, qp, twoA = fo
    diff = np.linalg.norm(PP - qp, axis=1) - np.linalg.norm(PP - q, axis=1)
    return bool(np.all(np.abs(np.abs(diff) - twoA) < 1e-6) and (np.all(diff > 0) or np.all(diff < 0)))
q, qp, Delta = np.array([0.0, 0.0]), np.array([1.0, 0.3]), 0.4
S5 = sheet_points(q, qp, Delta, 5, 2)
print("   five points on a branch: ambiguous =", ambiguous5(S5), "; recovered foci", np.round(hyperbola_foci(conic_through(S5))[0], 6), np.round(hyperbola_foci(conic_through(S5))[1], 6))
n_cfg = 20000; amb = sum(ambiguous5(rng.uniform(0, 1, (5, 2))) for _ in range(n_cfg))
print(f"   uniformly random five-point configurations in the unit square: {amb}/{n_cfg} = {100*amb/n_cfg:.1f} % ambiguous for the exponential law")
amb6 = 0
for _ in range(2000):
    PP = rng.uniform(0, 1, (6, 2))
    V = np.column_stack([PP[:,0]**2, PP[:,0]*PP[:,1], PP[:,1]**2, PP[:,0], PP[:,1], np.ones(6)])
    amb6 += (np.linalg.svd(V, compute_uv=False)[-1] < 1e-9)
print(f"   six random points: {amb6}/2000 lie on a common conic (expected 0)")
print("(3) d = 3: seven sensors on a hyperboloid sheet; Jacobian of |p_j-q'|-|p_j-q| = Delta in (q, q', Delta) and a nearby ambiguous configuration")
def F(theta, PP, d):
    q, qp, De = theta[:d], theta[d:2*d], theta[2*d]
    return np.linalg.norm(PP - qp, axis=1) - np.linalg.norm(PP - q, axis=1) - De
def jacF(theta, PP, d):
    q, qp = theta[:d], theta[d:2*d]
    u = (PP - q)/np.linalg.norm(PP - q, axis=1)[:, None]; up = (PP - qp)/np.linalg.norm(PP - qp, axis=1)[:, None]
    return np.column_stack([u, -up, -np.ones(len(PP))])
def newton(theta, PP, d):
    for it in range(100):
        r = F(theta, PP, d); step = np.linalg.lstsq(jacF(theta, PP, d), -r, rcond=None)[0]; theta = theta + step
        if np.linalg.norm(step) < 1e-14: break
    return theta, np.linalg.norm(F(theta, PP, d))
d = 3; q = np.zeros(d); qp = np.array([0.6, 0.6, 0.6]); Delta = 0.35
for trial in range(5):
    PP = sheet_points(q, qp, Delta, 7, d); th0 = np.concatenate([q, qp, [Delta]])
    sv = np.linalg.svd(jacF(th0, PP, d), compute_uv=False)
    th, res = newton(th0, PP + 1e-5*rng.standard_normal(PP.shape), d)
    print(f"   singular values of the 7x7 Jacobian: min {sv.min():.4f}, max {sv.max():.3f}; sensors perturbed by 1e-5: Newton residual {res:.1e}, "
          f"|q-q'| = {np.linalg.norm(th[:d]-th[d:2*d]):.4f} (was {np.linalg.norm(qp-q):.4f}), Delta = {th[2*d]:.4f} (was {Delta})")
print("(4) the checkable condition: rank of the Veronese matrix of monomials of degree <= 2")
def veronese(PP):
    x, y = PP[:, 0], PP[:, 1]
    return np.column_stack([np.ones(len(PP)), x, y, x*x, x*y, y*y])
grid = np.array([[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])
circ = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2*np.pi, 9)[:-1]])
line = np.column_stack([np.linspace(0, 1, 6), 0.3*np.linspace(0, 1, 6)])
for name, PP in (("3x3 grid", grid), ("8 on a circle", circ), ("6 on a line", line), ("6 random", rng.uniform(0, 1, (6, 2)))):
    V = veronese(PP); rk = np.linalg.matrix_rank(V, tol=1e-9)
    ker = np.linalg.svd(V)[2][rk:]
    print(f"   {name:14s}: rank {rk} of 6, kernel dimension {6-rk}", end="")
    if 6-rk == 1:
        c = ker[0]; Qm = np.array([[c[3], c[4]/2], [c[4]/2, c[5]]]); ev = np.linalg.eigvalsh(Qm)
        kind = "ellipse/circle" if np.prod(ev) > 1e-12 else ("parabola/degenerate" if abs(np.prod(ev)) <= 1e-12 else "hyperbola")
        if np.prod(ev) < -1e-12: kind = "hyperbola"
        print(f"; the unique conic through the sensors is a {kind} (eigenvalues of the quadratic part {np.round(ev/np.max(np.abs(ev)), 3)})")
    else: print()

print("(5) two sources in one mode: zeros of J(q) = 1 - |W^T g_q|^2/|g_q|^2 on a fine grid, 9-sensor grid, power law gamma=1")
def gpow(PP, q): return np.linalg.norm(PP - q, axis=1)**(-1.0)
q1, q2 = np.array([0.3, 0.4]), np.array([0.7, 0.55])
W = np.linalg.qr(np.column_stack([gpow(grid, q1), gpow(grid, q2)]))[0]
xs = np.linspace(0.05, 0.95, 361); Jmin = []
for x in xs:
    for y in xs:
        g = gpow(grid, np.array([x, y])); Jv = 1 - np.sum((W.T @ g)**2)/(g @ g)
        if Jv < 1e-6: Jmin.append((x, y, Jv))
Jmin = np.array(Jmin)
print(f"   grid points with J < 1e-6: {len(Jmin)}; they lie within {max(min(np.linalg.norm(p[:2]-q1), np.linalg.norm(p[:2]-q2)) for p in Jmin):.4f} of the two sources")

print("(6) noise floor: |rho_{L,K} - sigma^2| against the Frobenius bound sigma^2 mL sqrt((mL+kappa-2)/K) (Gaussian, kappa = 3), 20 trials")
for mL, K in ((16, 2000), (16, 20000), (64, 20000), (64, 200000), (128, 200000)):
    m, L = (1, mL) if mL <= 16 else (4, mL//4)
    dev = []
    for _ in range(20):
        E = rng.standard_normal((m, K+L-1)); H = np.vstack([np.lib.stride_tricks.sliding_window_view(E[j], L).T for j in range(m)])
        dev.append(np.linalg.eigvalsh(H @ H.T / K)[-1] - 1.0)
    bound = mL*np.sqrt((mL+1)/K)
    print(f"   mL={mL:4d} K={K:7d} (mL)^2/K={mL*mL/K:7.3f}: rho - sigma^2 = {np.mean(dev):+.4f} (MP edge excess {(1+np.sqrt(mL/K))**2-1:.4f}); root-mean-square Frobenius bound {bound:.3f}")

print("(7) sharpness of 2d+2 in every dimension: the planar 5x5 Jacobian of the system |p_j-q'|-|p_j-q| = Delta at five points of the")
print("    hyperbola branch x = a cosh s, y = b sinh s (a = 1, c = 2, foci (-c,0), (c,0)), evaluated to 40 digits")
try:
    from mpmath import mp, mpf, cosh, sinh, sqrt, matrix, det
    mp.dps = 40; a, c = mpf(1), mpf(2); b = sqrt(c*c - a*a)
    def row(s):
        x, y = a*cosh(s), b*sinh(s); rho = sqrt((x + c)**2 + y**2); rhop = sqrt((x - c)**2 + y**2)
        return [(x + c)/rho, y/rho, -(x - c)/rhop, -y/rhop, mpf(-1)]
    for S in ([-1.5, -0.7, 0.2, 0.9, 1.6], [-1, -0.5, 0, 0.5, 1]):
        print(f"   s = {S}: det = {mp.nstr(det(matrix([row(mpf(s)) for s in S])), 15)}")
except ImportError:
    print("   mpmath not available; see the paper for the values")
