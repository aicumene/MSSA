"""Figures 1-4 of the source-localisation paper (reads the results_*.json files of this directory)."""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from loc_common import *

F = os.path.join(HERE, '..')
def load(name): return json.load(open(os.path.join(HERE, name)))

# palette: three categorical slots, one ordinal blue ramp, one sequential blue ramp; all text in ink
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
B_L, B_M, B_D = '#86b6ef', '#3987e5', '#184f95'
INK, INK2, GRID, AXIS = '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
SEQ = LinearSegmentedColormap.from_list('blue', ['#eef5fe', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': AXIS,
                     'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-', 'axes.axisbelow': True,
                     'lines.linewidth': 1.9, 'lines.markersize': 6.5, 'lines.markeredgecolor': 'white', 'lines.markeredgewidth': 1.0,
                     'legend.frameon': False, 'legend.fontsize': 8, 'savefig.dpi': 200, 'savefig.bbox': 'tight', 'figure.facecolor': 'white'})
HALO = [pe.withStroke(linewidth=2.2, foreground='white')]
def title(ax, letter, text): ax.set_title(f'{letter}   {text}', loc='left', color=INK, fontweight='bold' if False else None)
def panel(ax, letter, text):
    ax.text(0.0, 1.035, letter, transform=ax.transAxes, fontweight='bold', fontsize=10, va='bottom', ha='left')
    ax.text(0.075, 1.035, text, transform=ax.transAxes, fontsize=9, va='bottom', ha='left')


# ------------------------------------------------------------------ Figure 1: where the fingerprint is ambiguous
def inv_point(q, o, R): return o + R ** 2 * (q - o) / np.sum((q - o) ** 2)
def circumcircle(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    o = np.array([((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d,
                  ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d])
    return o, float(np.linalg.norm(a - o))
g = Power(1.0); u = np.linspace(-2.7, 2.7, 541); XX, YY = np.meshgrid(u, u)
def mismatch(Pm, q0):
    G = profiles(np.c_[XX.ravel(), YY.ravel()], Pm, g); Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
    a = profiles(q0, Pm, g)[0]; return (1 - (Gn @ (a / np.linalg.norm(a))) ** 2).reshape(XX.shape)
th = np.linspace(0, 2 * np.pi, 8, endpoint=False)
P_circ = np.c_[np.cos(th), np.sin(th)]; P_grid = np.array([(x, y) for x in (-1.0, 0.0, 1.0) for y in (-1.0, 0.0, 1.0)])
P_three = np.array([[-0.9, -0.6], [0.9, -0.5], [0.1, 0.95]]); P_line = np.c_[np.linspace(-1, 1, 6), np.zeros(6)]
o3, R3 = circumcircle(*P_three)
cases = [('a', 'eight sensors on a circle', P_circ, np.array([0.35, 0.25]), inv_point(np.array([0.35, 0.25]), np.zeros(2), 1.0), (np.zeros(2), 1.0)),
         ('b', 'nine sensors on a grid', P_grid, np.array([0.35, 0.25]), None, None),
         ('c', 'three sensors', P_three, np.array([0.5, 0.4]), inv_point(np.array([0.5, 0.4]), o3, R3), (o3, R3)),
         ('d', 'six sensors on a line', P_line, np.array([0.3, 0.7]), np.array([0.3, -0.7]), None)]
fig, axs = plt.subplots(2, 2, figsize=(7.2, 7.0), sharex=True, sharey=True)
for ax, (letter, text, Pm, q0, q1, circ) in zip(axs.ravel(), cases):
    im = ax.pcolormesh(XX, YY, np.maximum(mismatch(Pm, q0), 1e-4), norm=LogNorm(1e-4, 1.0), cmap=SEQ.reversed(), shading='auto', rasterized=True)
    if circ is not None: ax.add_patch(plt.Circle(circ[0], circ[1], fill=False, color=INK2, lw=0.8))
    h1, = ax.plot(Pm[:, 0], Pm[:, 1], '^', color=INK, ms=6.5, mec='white', mew=0.9, ls='none', label='sensor')
    h2, = ax.plot(*q0, 'o', color='white', ms=7.5, mec=INK, mew=1.5, ls='none', label='source')
    if q1 is not None: h3, = ax.plot(*q1, 'D', color='white', ms=6.5, mec=INK, mew=1.5, ls='none', label='second point with the same fingerprint')
    ax.set_aspect('equal'); ax.grid(False); ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.7, 2.7); panel(ax, letter, text)
fig.legend(handles=[h1, h2, h3], loc='upper center', ncol=3, bbox_to_anchor=(0.47, 1.0), handletextpad=0.3, columnspacing=1.6)
fig.subplots_adjust(left=0.07, right=0.88, top=0.92, bottom=0.07, wspace=0.08, hspace=0.14)
cb = fig.colorbar(im, cax=fig.add_axes([0.905, 0.2, 0.018, 0.58])); cb.set_label('mismatch of the fingerprint with that of the source, $1-\\cos^2$', color=INK2); cb.outline.set_edgecolor(AXIS)
fig.savefig(f'{F}/fig1_ambiguity.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 2: accuracy for one source
RS = load('results_single.json'); fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
for ax, key, letter, text in ((axs[0, 0], 'E1_power', 'a', 'power law, $g(\\rho)=\\rho^{-1}$'), (axs[0, 1], 'E1_exponential', 'b', 'exponential law, $g(\\rho)=e^{-3\\rho}$')):
    r = RS[key]; x = [d['snr0'] for d in r]
    ax.plot(x, [d['predicted_rmse'] for d in r], color=INK, lw=1.0, label='first-order prediction', zorder=2)
    ax.plot(x, [d['band']['rmse'] for d in r], 's-', color=ORANGE, label='band power, known frequency', zorder=3)
    ax.plot(x, [d['signed']['rmse'] for d in r], 'o-', color=BLUE, label='MSSA loading', zorder=4)
    ax.set_yscale('log'); ax.set_ylim(6e-4, 1.2); ax.set_xlabel('SNR at the reference distance (dB)'); ax.set_ylabel('RMS position error'); panel(ax, letter, text)
axs[0, 0].legend(loc='lower left', fontsize=7.5)
ax = axs[1, 0]; v = np.linspace(-0.25, 1.25, 151); VX, VY = np.meshgrid(v, v); g1 = Power(1.0); sig = noise_sigma(g1, 0.0)
D = np.array([[np.sqrt(np.trace(predicted_cov(np.array([x, y]), 1.0, GRID3, g1, 2 * sig ** 2 / 500))) for x in v] for y in v])
im = ax.pcolormesh(VX, VY, D, norm=LogNorm(3e-3, 0.3), cmap=SEQ, shading='auto', rasterized=True); ax.plot(GRID3[:, 0], GRID3[:, 1], '^', color=INK, ms=6.5, mec='white', mew=0.9, ls='none')
ax.set_aspect('equal'); ax.grid(False); ax.set_xlabel('$x$'); ax.set_ylabel('$y$'); panel(ax, 'c', 'predicted RMS error at 0 dB')
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_edgecolor(AXIS); cb.set_label('RMS position error', color=INK2)
ax = axs[1, 1]; r = RS['E5_sensors']; mm = [d['m'] for d in r]; med = [d['median'] for d in r]
ax.plot(mm, med, 'o-', color=BLUE); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(mm); ax.set_xticklabels([str(k) for k in mm]); ax.minorticks_off()
for d in r:
    if d['gross'] >= 0.005: ax.annotate(f"{100 * d['gross']:.0f} % gross errors", (d['m'], d['median']), textcoords='offset points', xytext=(8, 3), fontsize=7.5, color=INK2)
ax.set_xlabel('number of sensors'); ax.set_ylabel('median position error'); panel(ax, 'd', 'random arrays, 10 dB')
fig.tight_layout(h_pad=2.2, w_pad=1.6); fig.savefig(f'{F}/fig2_accuracy.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 3: two sources
RP = load('results_pairs.json'); fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0))
ax = axs[0]; r = RP['E2a_strength_ratio']; x = [d['ratio'] for d in r]
ax.plot(x, [d['svd']['median'] for d in r], 's-', color=ORANGE, label='singular pairs'); ax.plot(x, [d['signed']['median'] for d in r], 'o-', color=BLUE, label='poles + least squares')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([1, 1.25, 1.5, 2, 3]); ax.set_xticklabels(['1', '1.25', '1.5', '2', '3']); ax.minorticks_off()
ax.set_xlabel('ratio of source strengths'); ax.set_ylabel('median position error'); ax.legend(loc='upper right', bbox_to_anchor=(1.03, 1.0), fontsize=7.5); panel(ax, 'a', '10 and 13 Hz')
ax = axs[1]; r = RP['E2b_frequency_separation']
for snr, col, mk in ((10.0, B_L, 'o'), (20.0, B_D, 's')):
    rr = [d for d in r if d['snr0'] == snr]; ax.plot([d['separation'] for d in rr], [d['median'] for d in rr], mk + '-', color=col, label=f'{snr:.0f} dB')
ax.axvline(1.0, color=INK2, lw=0.8); ax.text(1.07, 0.97, 'Fourier limit', transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK2, va='top', rotation=90)
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(['0.25', '0.5', '1', '2', '4']); ax.minorticks_off()
ax.set_yticks([0.002, 0.005, 0.01, 0.02]); ax.set_yticklabels(['0.002', '0.005', '0.01', '0.02'])
ax.set_xlabel('frequency separation (units of $1/T$)'); ax.legend(loc='upper right', bbox_to_anchor=(1.02, 0.86)); panel(ax, 'b', 'close frequencies')
ax = axs[2]; r = RP['E3_same_frequency']
for snr, col, mk in ((0.0, B_L, 'o'), (10.0, B_M, 's'), (20.0, B_D, '^')):
    rr = [d for d in r if d['snr0'] == snr]; ax.plot([d['dphi'] for d in rr], [d['scan']['median'] for d in rr], mk + '-', color=col, label=f'subspace scan, {snr:.0f} dB')
rr = [d for d in r if d['snr0'] == 10.0]; ax.plot([d['dphi'] for d in rr], [d['single']['median'] for d in rr], 'D-', color=ORANGE, label='one fingerprint, 10 dB')
ax.set_yscale('log'); ax.set_xticks([10, 20, 45, 90]); ax.set_xlabel('phase difference (degrees)'); ax.set_ylim(8e-4, 20); ax.legend(loc='upper center', fontsize=7.2); panel(ax, 'c', 'one frequency')
fig.tight_layout(w_pad=1.2); fig.savefig(f'{F}/fig3_two_sources.png'); plt.close(fig)

# ------------------------------------------------------------------ Figure 4: models and the full pipeline
RM = load('results_models.json'); RL = load('results_pipeline.json'); fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.2))
ax = axs[0, 0]; e6 = RM['E6_exponent']; gs = sorted(float(k) for k in e6['assumed']); ax.plot(gs, [e6['assumed'][str(k)]['median'] for k in gs], 'o-', color=BLUE)
ax.axhline(e6['joint']['position']['median'], color=INK, lw=1.0); ax.axvline(1.0, color=INK2, lw=0.8)
ax.text(0.98, e6['joint']['position']['median'] * 1.07, 'estimated jointly', transform=ax.get_yaxis_transform(), fontsize=7.5, color=INK2, ha='right', va='bottom')
ax.text(1.03, 0.97, 'true exponent', transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK2, va='top')
ax.set_yscale('log'); ax.set_yticks([0.01, 0.02, 0.05, 0.08]); ax.set_yticklabels(['0.01', '0.02', '0.05', '0.08']); ax.minorticks_off()
ax.set_xlabel('exponent assumed in the fit'); ax.set_ylabel('median position error'); panel(ax, 'a', 'attenuation exponent fixed in advance')
ax = axs[0, 1]; e7 = RM['E7_dipole']
for snr, key, col, mk, lab in ((20.0, 'scalar_model', ORANGE, 'D', 'distance-only model, 20 dB'), (0.0, 'gain_model', B_L, 'o', 'gain-matrix model, 0 dB'), (20.0, 'gain_model', B_D, 's', 'gain-matrix model, 20 dB')):
    rr = [d for d in e7 if d['snr'] == snr]; ax.plot([d['psi'] for d in rr], [d[key]['median'] for d in rr], mk + '-', color=col, label=lab)
ax.set_yscale('log'); ax.set_ylim(2e-4, 3.0); ax.set_xticks([0, 30, 60, 90]); ax.set_xticklabels(['0\nradial', '30', '60', '90\ntangential'])
ax.set_xlabel('dipole orientation (degrees)'); ax.set_ylabel('median position error'); ax.legend(loc='center', bbox_to_anchor=(0.5, 0.555)); panel(ax, 'b', 'current dipole, 32 sensors')
ax = axs[1, 0]; r = RL['E8_pipeline']['rows']; x = [d['snr0'] for d in r]
ax.plot(x, [d['p_rank'] for d in r], 'o-', color=BLUE); ax.set_ylim(-0.03, 1.03); ax.set_xlabel('SNR at the reference distance (dB)'); ax.set_ylabel('P(estimated rank = 6)'); panel(ax, 'c', 'three sources: count')
ax = axs[1, 1]; rr = [d for d in r if d.get('predicted_rmse') is not None and d['p_rank'] > 0.5]
ax.plot([d['snr0'] for d in rr], [d['predicted_rmse'] for d in rr], color=INK, lw=1.0, label='first-order prediction')
ax.plot([d['snr0'] for d in rr], [d['rmse_no_gross'] for d in rr], 'o-', color=BLUE, label='pipeline with estimated rank')
ax.set_yscale('log'); ax.set_yticks([0.001, 0.003, 0.01, 0.03]); ax.set_yticklabels(['0.001', '0.003', '0.01', '0.03']); ax.minorticks_off()
ax.set_xlabel('SNR at the reference distance (dB)'); ax.set_ylabel('RMS position error'); ax.legend(loc='upper right'); panel(ax, 'd', 'three sources: position')
fig.tight_layout(h_pad=2.2, w_pad=1.6); fig.savefig(f'{F}/fig4_models_pipeline.png'); plt.close(fig)
# ------------------------------------------------------------------ Figure 5: blind separation against record length
RI = load('results_ica_long.json'); fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, kind, letter, text in ((axs[0], 'independent bursts', 'a', 'independent bursts (non-Gaussian)'), (axs[1], 'independent Gaussian narrow-band', 'b', 'independent Gaussian narrow-band signals')):
    rr = [d for d in RI if d['kind'] == kind]; x = [d['seconds'] for d in rr]
    ax.plot(x, [d['sobi']['median'] for d in rr], '^-', color=AQUA, label='SOBI, then matching')
    ax.plot(x, [d['ica']['median'] for d in rr], 's-', color=ORANGE, label='FastICA, then matching')
    ax.plot(x, [d['music']['median'] for d in rr], 'o-', color=BLUE, label='subspace scan')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in x]); ax.minorticks_off()
    ax.set_yticks([0.001, 0.01, 0.1]); ax.set_yticklabels(['0.001', '0.01', '0.1']); ax.set_ylim(6e-4, 0.4); ax.set_xlabel('record length (s)'); panel(ax, letter, text)
axs[0].set_ylabel('median position error'); axs[1].legend(loc='center right', bbox_to_anchor=(1.0, 0.45))
fig.tight_layout(w_pad=1.4); fig.savefig(f'{F}/fig5_blind_separation.png'); plt.close(fig)
print('figures written')
