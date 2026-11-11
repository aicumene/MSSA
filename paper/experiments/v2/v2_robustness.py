"""Robustness of the complete procedure to three kinds of model mismatch not covered by version 1:
(a) errors in the sensor coordinates, (b) spatially correlated sensor noise, (c) propagation delays.
Setting of Section `Numerical experiments': 3x3 grid on the unit square, fs = 100 Hz, N = 500, L = 40,
one undamped source at a random position in [0.1,0.9]^2 at least 0.12 from every sensor, g(rho) = 1/rho,
SNR_0 defined at rho_0 = 0.5.  Pipeline: block-Hankel SVD -> calibrated count -> ESPRIT poles ->
least-squares complex loading at the estimated frequency -> Gauss-Newton matching from a grid start.  NumPy only."""
import numpy as np, time, sys
rng = np.random.default_rng(5)
T = int(sys.argv[1]) if len(sys.argv) > 1 else 300
P = np.array([[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])
m, fs, N, L = len(P), 100.0, 500, 40
K = N - L + 1; beta = m*L/K
t = np.arange(N) / fs
gamma = 1.0
def g_of(q, PP=P): return np.linalg.norm(PP - q, axis=1)**(-gamma)
def J_of(q, PP=P):
    rho = np.linalg.norm(PP - q, axis=1); gp = -gamma*rho**(-gamma-1)
    return (gp/rho)[:, None] * (q - PP)

def block_hankel(X):
    return np.vstack([np.lib.stride_tricks.sliding_window_view(X[j], L).T for j in range(X.shape[0])])

def random_source():
    while True:
        q = rng.uniform(0.1, 0.9, 2)
        if np.min(np.linalg.norm(P - q, axis=1)) >= 0.12: return q

def count_modes(HX, sigma2_hat, kappa=0.15):
    ell = np.linalg.svd(HX, compute_uv=False)**2 / K
    return int(np.sum(ell > sigma2_hat*(1+np.sqrt(beta))**2*(1+kappa))), ell

def mp_median(beta):
    # median of the Marchenko-Pastur law of ratio beta (variance 1), by numerical integration
    a, b = (1-np.sqrt(beta))**2, (1+np.sqrt(beta))**2
    x = np.linspace(a, b, 4001); dens = np.sqrt(np.maximum((b-x)*(x-a), 0))/(2*np.pi*beta*x)
    cdf = np.cumsum(dens)*(x[1]-x[0]); cdf /= cdf[-1]
    return x[np.searchsorted(cdf, 0.5)]

MP_MED = mp_median(beta)

def esprit_freqs(HX, d):
    U = np.linalg.svd(HX, full_matrices=False)[0][:, :d]                       # signal subspace, (mL) x d
    U1 = np.vstack([U[j*L:(j+1)*L-1] for j in range(m)]); U2 = np.vstack([U[j*L+1:(j+1)*L] for j in range(m)])
    Phi = np.linalg.lstsq(U1, U2, rcond=None)[0]
    z = np.linalg.eigvals(Phi)
    f = np.angle(z)*fs/(2*np.pi)
    return np.sort(f[f > 0.5])                                                  # positive frequencies

def complex_loading(X, f):
    C = np.vstack([np.cos(2*np.pi*f*t), np.sin(2*np.pi*f*t)]).T
    B = np.linalg.lstsq(C, X.T, rcond=None)[0]                                  # 2 x m
    return B[0] - 1j*B[1]                                                       # complex amplitude per channel

def match(a_hat, PP=P, W=None):
    """maximise the fit of a_hat (real, positive) to alpha g_q: grid start, then Gauss-Newton; W = whitening matrix or None"""
    Wm = np.eye(m) if W is None else W
    aw = Wm @ a_hat
    best, bq = -1, None
    for x in np.linspace(0.05, 0.95, 19):
        for y in np.linspace(0.05, 0.95, 19):
            gw = Wm @ g_of(np.array([x, y]), PP); c = (aw @ gw)**2/((gw @ gw)*(aw @ aw))
            if c > best: best, bq = c, np.array([x, y])
    q = bq.copy()
    for it in range(50):
        gw = Wm @ g_of(q, PP); Jw = Wm @ J_of(q, PP)
        alpha = (aw @ gw)/(gw @ gw); resid = aw - alpha*gw
        Pi = np.eye(m) - np.outer(gw, gw)/(gw @ gw)
        step = np.linalg.lstsq(alpha*(Pi @ Jw), resid, rcond=None)[0]
        q = q + step
        if np.linalg.norm(step) < 1e-10: break
    return q

def run(trials, snr_db, sensor_err=0.0, corr_len=0.0, speed=np.inf, whiten=False, use_modulus=True):
    sigma2 = (0.5**(-2*gamma)) / (2*10**(snr_db/10))                             # alpha = 1
    errs = []; gross = 0; counts = []; inphase_fail = 0; pred_terms = []
    Sigma = np.exp(-np.linalg.norm(P[:, None] - P[None], axis=2)/corr_len) if corr_len > 0 else np.eye(m)
    Ls = np.linalg.cholesky(Sigma)
    for _ in range(trials):
        q0 = random_source(); f0 = rng.uniform(8, 15); phi = rng.uniform(0, 2*np.pi)
        rho = np.linalg.norm(P - q0, axis=1); a0 = g_of(q0)
        delay = rho/speed if np.isfinite(speed) else np.zeros(m)
        X = a0[:, None]*np.sin(2*np.pi*f0*(t[None] - delay[:, None]) + phi) + np.sqrt(sigma2)*(Ls @ rng.standard_normal((m, N)))
        Pused = P + sensor_err*rng.standard_normal(P.shape)
        Xw = X
        if whiten:
            Xw = np.linalg.solve(Ls, X)                                            # known noise covariance
        HX = block_hankel(Xw)
        ell = np.linalg.svd(HX, compute_uv=False)**2 / K
        sigma2_hat = np.median(ell)/MP_MED
        d_hat, _ = count_modes(HX, sigma2_hat); counts.append(d_hat)
        if d_hat < 2: gross += 1; errs.append(np.nan); continue
        fr = esprit_freqs(HX, max(2, min(d_hat, 6)))
        f_hat = fr[np.argmin(np.abs(fr - f0))] if len(fr) else f0
        c = complex_loading(X, f_hat)
        # in-phase test: phases of the complex loading equal up to sign
        ph = np.angle(c*np.exp(-1j*np.angle(c[np.argmax(np.abs(c))])))
        if np.max(np.abs(np.sin(ph))) > 0.1: inphase_fail += 1
        a_hat = np.abs(c) if use_modulus else np.real(c*np.exp(-1j*np.angle(c[np.argmax(np.abs(c))])))
        W = np.linalg.inv(Ls) if whiten else None
        q = match(a_hat, Pused, W)
        e = np.linalg.norm(q - q0); errs.append(e); gross += (e > 0.1)
        # first-order prediction of the position error from sensor-coordinate errors alone
        if sensor_err > 0:
            J0 = J_of(q0); Pi0 = np.eye(m) - np.outer(a0, a0)/(a0 @ a0)
            G = np.linalg.inv(J0.T @ Pi0 @ J0) @ J0.T @ Pi0
            D = np.diag((gamma*rho**(-gamma-1))**2) * sensor_err**2                # Var of g_j from an isotropic error of p_j
            pred_terms.append(np.trace(G @ D @ G.T))
    errs = np.array(errs); ok = np.isfinite(errs) & (errs <= 0.1)
    return dict(rms_all=np.sqrt(np.nanmean(errs**2)), rms_fine=np.sqrt(np.mean(errs[ok]**2)) if ok.any() else np.nan,
                gross=gross/trials, count_ok=np.mean(np.array(counts) == 2), inphase_fail=inphase_fail/trials,
                pred_sensor=np.sqrt(np.mean(pred_terms)) if pred_terms else np.nan)

t0 = time.time()
print(f"T = {T} trials per row; SNR_0 = 10 dB unless stated; errors in units of the array side")
print("\n(a) sensor-coordinate errors sigma_p (isotropic Gaussian, used coordinates = true + error):")
print("sigma_p   RMS(fine)  gross   predicted RMS from sensor errors alone")
for se in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1):
    R = run(T, 10, sensor_err=se)
    print(f"{se:6.3f}   {R['rms_fine']:8.4f}  {100*R['gross']:5.1f}%   {R['pred_sensor']:.4f}", flush=True)
print("\n(b) spatially correlated noise, Sigma_ij = exp(-|p_i-p_j|/l_c), equal variance; count with calibrated threshold:")
print("l_c    whiten  count==2  RMS(fine)  gross")
for lc in (0.0, 0.25, 0.5, 1.0):
    for wh in (False, True):
        if lc == 0 and wh: continue
        R = run(T, 10, corr_len=lc, whiten=wh)
        print(f"{lc:4.2f}   {str(wh):5s}   {100*R['count_ok']:5.1f}%   {R['rms_fine']:8.4f}  {100*R['gross']:5.1f}%", flush=True)
print("\n(c) propagation delays, arrival at sensor j delayed by rho_j / v (v in array sides per second), f in [8,15] Hz:")
print("v       in-phase test fails  RMS(fine) modulus-matching  gross   | real-part matching: RMS(fine) gross")
for v in (np.inf, 50.0, 20.0, 10.0, 5.0):
    R = run(T, 10, speed=v); R2 = run(T, 10, speed=v, use_modulus=False)
    print(f"{v:6.1f}   {100*R['inphase_fail']:5.1f}%            {R['rms_fine']:8.4f}           {100*R['gross']:5.1f}%  | {R2['rms_fine']:8.4f}  {100*R2['gross']:5.1f}%", flush=True)
print("\n(d) reference: SNR_0 sweep without mismatch (count, localization):")
for snr in (0, 5, 10, 20):
    R = run(T, snr)
    print(f"SNR_0={snr:3d} dB: count==2 {100*R['count_ok']:5.1f}%, RMS(fine) {R['rms_fine']:.4f}, gross {100*R['gross']:5.1f}%", flush=True)
print(f"elapsed {time.time()-t0:.0f} s")
