# Preprint

**Alexandra Bernadotte, Ivan Menshikov.** *Multichannel Singular Spectrum Analysis: Counting Oscillatory Modes and Recovering
Their Spatial Loadings.* Aicumene Preprint 2026.044v2, November 11, 2026; SSRN working paper 7693202, https://ssrn.com/abstract=7693202, doi:10.2139/ssrn.7693202. Companion paper:
*From Oscillatory Modes to Source Geometry: Identifiability and Localization in Multichannel Time Series*, Aicumene Preprint 2026.058 (forthcoming).
https://research.aicumene.com/papers/MSSA/
Version 1 (August 12, 2026) appeared as *Multichannel Singular Spectrum Analysis for Estimating Oscillatory Sources and
Localizing Attenuating Oscillations*, DOI: [10.13140/RG.2.2.26719.01442](https://doi.org/10.13140/RG.2.2.26719.01442).

Cite as: Bernadotte, A. & Menshikov, I. *Multichannel Singular Spectrum Analysis: Counting Oscillatory Modes and Recovering Their Spatial Loadings.* Aicumene Preprint 2026.044v2, 2026. https://research.aicumene.com/papers/MSSA/

| file | contents |
|---|---|
| `paper.pdf` | the preprint, version 2 |
| `paper.tex`, `arxiv.sty`, `orcid.pdf`, `fig*.png` | LaTeX source of version 2; build with `pdflatex paper.tex` (three passes) |
| `experiments/v2/v2_checks.py` | version 2: noise floor against the record length, exact count by the Weyl interval, end-to-end accuracy (Lemma 1, Theorem 2, Proposition 7) |
| `experiments/v2/v2_table_ci.py` | version 2, Table 3: the exact count against practice, 200 trials per row with 95% Wilson intervals; output `rank_table_ci.txt` (`rank_table.txt` is the 25-trial version) |
| `experiments/v2/v2_identifiability_checks.py` | version 2: four sensors on a hyperbola branch, share of ambiguous five-sensor arrays, seven-sensor Jacobian in three dimensions, Veronese rank test, zeros of the plane scan, noise-floor bound (Propositions 4 and 5, Corollary 4, Theorem 7); output `identifiability_checks.txt` |
| `experiments/v2/v2_robustness.py` | version 2, Section 11.10: sensor-coordinate errors, spatially correlated noise with and without whitening, propagation delays; output `robustness_table.txt` |
| `experiments/v2/fig_geometry.py` | version 2, Fig. 1: geometry of the trajectory space (`fig6_geometry.png`) |
| `experiments/v2/fig_real_localization.py` | version 2, Fig. 3: the Section 10 algorithm on the two UCI recordings (ideal 10-10 electrodes on a unit sphere, dipole in a homogeneous sphere by the Legendre series, surrogate count, ESPRIT, loadings, dipole fits); output `real_localization.txt`, `fig8_real_localization.png`; same data argument as the next script |
| `experiments/v2/fig_real_3d.py` | version 2, Fig. 4: the sources of Fig. 3 inside the three-dimensional head model, from `real_localization_fits.json` written by the previous script (`fig9_real_3d.png`) |
| `experiments/v2/fig_uci_geometry.py` | version 2, Fig. 2: the same geometry on two trials of the UCI EEG database (doi:10.24432/C5TS3D); download `smni_eeg_data.tar.gz` from https://archive.ics.uci.edu/dataset/121, extract the `a_1_*` and `c_1_*` archives and pass the folder as the argument (`fig7_uci_geometry.png`) |
| `experiments/loc_common.py` | pole estimation (ESPRIT on the block-Hankel subspace), loadings, in-phase test, fingerprint matching, subspace scan, accuracy formula |
| `experiments/exp0_exact.py` … `exp7c_ica_two_freq.py` | the experiments of Section 11; each writes a `results_*.json` |
| `experiments/blind.py` | FastICA and SOBI baselines |
| `experiments/make_figures.py` | Figures 1–5 from the `results_*.json` files |

The scripts of version 1 need NumPy, SciPy and Matplotlib, and use the `aicumene_mssa` package from `../src` of this repository.
Run them from `paper/experiments/`, for example `python exp1_single.py`. The scripts of version 2 in `experiments/v2/` need only
NumPy (and Matplotlib for the two figures) and are self-contained; `v2_table_ci.py` takes about four minutes and `v2_robustness.py`
about eleven minutes with the BLAS thread count limited to two or three (`OPENBLAS_NUM_THREADS=3`).
