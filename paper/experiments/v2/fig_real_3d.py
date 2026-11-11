"""Fig.: the sources of Fig. 3 inside a three-dimensional head model.  Reads real_localization_fits.json written by
fig_real_localization.py.  The head is the unit sphere of the forward model; the in-phase loading of the slow evoked wave is
painted on the surface by inverse-distance interpolation of the 61 electrode values; the fitted dipole of the average trial is
drawn with its moment direction, the single-trial alpha fits as a cloud with their mean, and a drop line from every source to the
equatorial plane gives the depth.  NumPy + Matplotlib (mplot3d)."""
import numpy as np, json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import cm
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "real_localization_fits.json")))
E = np.array(J["electrodes"]["xyz"]); names = J["electrodes"]["names"]

def sphere_mesh(nu=72, nv=36, zmin=-0.35):
    u = np.linspace(0, 2*np.pi, nu); v = np.linspace(np.arccos(1.0), np.arccos(zmin), nv)
    U, V = np.meshgrid(u, v); return np.sin(V)*np.cos(U), np.sin(V)*np.sin(U), np.cos(V)
def interpolate(vals, X, Y, Z, power=2.0):
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    d = np.linalg.norm(P[:, None, :] - E[None, :, :], axis=2)
    w = 1.0/np.maximum(d, 1e-6)**power; w[d < 1e-6] = 1e12
    return ((w*vals[None, :]).sum(axis=1)/w.sum(axis=1)).reshape(X.shape)
def draw_head(ax, loading=None, alpha_surf=0.32):
    X, Y, Z = sphere_mesh()
    if loading is not None:
        V = interpolate(np.asarray(loading), X, Y, Z); vm = np.max(np.abs(V))
        ax.plot_surface(X, Y, Z, facecolors=cm.RdBu_r((V/vm + 1)/2), rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=alpha_surf, shade=False)
    else:
        ax.plot_wireframe(X, Y, Z, rstride=3, cstride=6, color="gray", lw=0.4, alpha=0.5)
    t = np.linspace(0, 2*np.pi, 200); ax.plot(np.cos(t), np.sin(t), 0*t, color="k", lw=0.6, alpha=0.6)            # equator
    ax.scatter(E[:, 0], E[:, 1], E[:, 2], s=6, c="k", depthshade=False)
    ax.plot([0, 0], [1.0, 1.12], [0.25, 0.25], color="k", lw=1.2)                                                   # nose
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_zlim(-0.6, 1.05); ax.set_box_aspect((1, 1, 0.8))
    ax.set_xlabel("$x$ (right)", fontsize=7, labelpad=-4); ax.set_ylabel("$y$ (anterior)", fontsize=7, labelpad=-4); ax.set_zlabel("$z$ (up)", fontsize=7, labelpad=-4)
    ax.tick_params(labelsize=6, pad=-2); ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
def dropline(ax, q, color):
    ax.plot([q[0], q[0]], [q[1], q[1]], [q[2], 0], color=color, lw=0.8, ls=":"); ax.scatter([q[0]], [q[1]], [0], s=12, color=color, marker="x")
def draw_sources(ax, D, with_alpha=True):
    S = D["bands"]["slow"]["average"]
    if S is not None:
        q = np.array(S["q"]); mo = np.array(S["moment"]); mo = 0.3*mo/np.linalg.norm(mo)
        ax.scatter([q[0]], [q[1]], [q[2]], s=220, color="C2", marker="*", edgecolor="k", depthshade=False, label=f"slow wave, average trial, {S['f']:.1f} Hz, $|q| = {np.linalg.norm(q):.2f}$")
        ax.quiver(q[0], q[1], q[2], mo[0], mo[1], mo[2], color="C2", lw=2, arrow_length_ratio=0.3); dropline(ax, q, "C2")
    if with_alpha:
        A = D["bands"]["alpha"]["single"]
        if A:
            Q = np.array([x["q"] for x in A]); qm = Q.mean(0)
            ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], s=16, color="C1", marker="s", alpha=0.8, depthshade=False, label=f"alpha, {len(A)} single trials")
            ax.scatter([qm[0]], [qm[1]], [qm[2]], s=170, color="C1", marker="P", edgecolor="k", depthshade=False, label=f"alpha, mean of the trials, $|q| = {np.linalg.norm(qm):.2f}$")
            dropline(ax, qm, "C1")
fig = plt.figure(figsize=(14, 9.6), dpi=160)
views = (("oblique view from the back and left", 24, -128), ("side view from the left", 4, -180))
k = 0
for row, lab in enumerate(("control", "alcoholic")):
    D = J["subjects"][lab]; S = D["bands"]["slow"]["average"]
    for col, (vname, el, az) in enumerate(views):
        k += 1; ax = fig.add_subplot(2, 2, k, projection="3d")
        draw_head(ax, loading=S["loading"] if col == 0 else None)
        draw_sources(ax, D)
        ax.view_init(elev=el, azim=az)
        if col == 1: ax.set_xticks([]); ax.set_xlabel("")
        ax.set_title("", fontsize=1)
        ax.text2D(0.5, 0.97, f"({'abcd'[k-1]}) {lab}: {vname}" + (f"\nsurface: in-phase loading of the {S['f']:.1f} Hz mode" if col == 0 else ""), transform=ax.transAxes, ha="center", va="top", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower left")
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.0, hspace=0.02); out = os.path.join(HERE, "fig9_real_3d.png"); fig.savefig(out); print("saved", out)
