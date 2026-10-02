# Preprint

**Alexandra Bernadotte, Ivan Menshikov.** *Multichannel Singular Spectrum Analysis for Estimating Oscillatory Sources
and Localizing Attenuating Oscillations.* Aicumene. Preprint. 2026 (AiCumene:2026.044v1).
DOI: [10.13140/RG.2.2.26719.01442](https://doi.org/10.13140/RG.2.2.26719.01442) · https://research.aicumene.com/papers/MSSA/

Cite as: Bernadotte, A. & Menshikov, I. *Multichannel Singular Spectrum Analysis.* Aicumene. Preprint. 2026. DOI: https://doi.org/10.13140/RG.2.2.26719.01442

| file | contents |
|---|---|
| `paper.pdf` | the preprint |
| `paper.tex`, `arxiv.sty`, `orcid.pdf`, `fig*.png` | LaTeX source; build with `pdflatex paper.tex` (three passes) |
| `experiments/loc_common.py` | pole estimation (ESPRIT on the block-Hankel subspace), loadings, in-phase test, fingerprint matching, subspace scan, accuracy formula |
| `experiments/exp0_exact.py` … `exp7c_ica_two_freq.py` | the experiments of Section 11; each writes a `results_*.json` |
| `experiments/blind.py` | FastICA and SOBI baselines |
| `experiments/make_figures.py` | Figures 1–5 from the `results_*.json` files |

The scripts need NumPy, SciPy and Matplotlib, and use the `aicumene_mssa` package from `../src` of this repository.
Run them from `paper/experiments/`, for example `python exp1_single.py`.
