# MSSA — Multichannel Singular Spectrum Analysis

A lightweight, guarantee-backed toolbox for extracting **oscillatory modes** and
**signal-subspace geometry** from multichannel time series (EEG, BCI, and general
signals).

MSSA embeds a recording as a **block-Hankel (trajectory) matrix**, estimates its
signal-subspace rank with a self-calibrating **ε-rank rule at the Bai–Yin
random-matrix edge**, and recovers the underlying oscillatory poles by
**total-least-squares ESPRIT** — returning the number of active modes, their
frequencies, damping/growth rates and per-channel amplitudes, together with the
Grassmann geometry of the signal subspace.

It is the numerical core behind **BOID** (brain oscillator intrinsic dimension),
real-time EEG denoising, EEG source counting for brain–computer interfaces, and
subspace-based time-series classification.

> **Authors:** Alexandra Bernadotte and Ivan Menshikov (AiCumene).
> **Preprint (forthcoming):** https://research.aicumene.com/papers/MSSA/

## Why MSSA

Classical singular-spectrum / subspace methods pick the number of "signal"
components with an ad-hoc energy cutoff. MSSA replaces that heuristic with a rank
estimator that carries a **finite-sample recovery guarantee**: the noise floor is
the Bai–Yin edge of the lagged-scatter spectrum, self-calibrated from its robust
(Marchenko–Pastur) median, so no external noise level is required and the Hankel
dependence between trajectory columns does not inflate the threshold. Under a
spike condition the estimated rank equals the true rank with probability tending
to one.

## Features

- **Block-Hankel / trajectory embedding** of single- and multichannel signals.
- **ε-rank signal-subspace estimation** at the Bai–Yin edge (Marchenko–Pastur
  self-calibration); classical energy cutoff available as an option.
- **TLS-ESPRIT pole extraction:** frequencies, damping/growth rates, per-channel
  complex amplitudes.
- **Grassmann subspace geometry:** subspaces as points on Gr(n, r), chordal /
  sin Θ distance, extrinsic (projector) class means, discriminative subspaces.
- **Oscillatory-mode descriptors:** mode count (→ intrinsic dimension), band
  fractions, mode-frequency / mode-power entropies.
- Pure **NumPy** core — small, dependency-light, real-time friendly.

## Method (brief)

A window of a multichannel signal modeled as a finite sum of exposinusoidal modes
has a block-Hankel embedding of **exact rank r = k₁ + 2·k₂** (k₁ real poles, k₂
oscillatory conjugate pairs). Estimating the number of oscillators therefore
reduces to recovering the signal-subspace rank of a noisy trajectory matrix —
solved here by the ε-rank rule — after which ESPRIT reads the poles off the lag
shift-invariance of the Hankel row layout. The induced subspace score is
√r-Lipschitz and the discriminative subspace is Davis–Kahan stable, so
generalization is controlled by the intrinsic Grassmann dimension r(n−r) rather
than by the ambient embedding dimension.

## Install

```bash
pip install numpy        # scipy optional
pip install -e .         # from the repository root
```

## Quickstart

```python
import numpy as np
from mssa import MSSA

fs = 100.0
X = eeg_window                                   # (n_samples, n_channels)

m  = MSSA(window=12, rank_method="epsilon", center=True).fit(X)   # Hankel + ε-rank subspace
md = m.esprit(fs=fs, method="tls")               # TLS-ESPRIT poles

m.rank            # estimated signal-subspace rank r̂ (= 2 × #oscillator pairs)
md.frequency      # mode frequencies (Hz)
md.damping        # damping / growth rates
```

## Applications

- **BOID** — brain oscillator intrinsic dimension: oscillator count per window as
  a state marker (companion study: *Brain states are organized by regulated
  oscillator dimensionality*).
- **Real-time EEG filtering / denoising** via signal-subspace reconstruction.
- **EEG source counting** for brain–computer interfaces.
- **Subspace classification** of time series through class-projector Grassmann
  geometry.

## Related work & ecosystem

MSSA is the shared numerical core of a connected body of work on Hankel-embedded,
subspace-geometric brain-signal analysis. Works that build on it:

- **BOID — regulated oscillator dimensionality.** A. Bernadotte, I. Menshikov,
  *Brain states are organized by regulated oscillator dimensionality* (2026).
  A brain-state marker (oscillator count per window) built directly on MSSA.
- **BOID in coma.** A. Bernadotte, I. Menshikov, *Brain oscillator intrinsic
  dimension marks consciousness, recovery mode, and failed downshifting in coma.*
  Preprint (2026). ResearchGate 408624059.
- **Topology-driven classification.** A. Bernadotte, *Topology-driven
  classification of time series.* bioRxiv (2026). doi:10.64898/2026.04.25.720787.
  Grassmann/subspace classification of MSSA signal subspaces.
- **Source counting for BCI.** A. Bernadotte, V. Buchstaber, *Method for
  evaluating the number of signal sources and application to non-invasive
  brain–computer interface.* arXiv:2410.11844 (2024).
  doi:10.48550/arXiv.2410.11844.
- **Real-time pipeline.** I. Menshikov, N. Elfimov, A. Bernadotte, *A lightweight
  Hankel-embedded pipeline for real-time EEG filtering and classification.*
  Proc. ICARA, 589–593 (2026). doi:10.1109/ICARA69401.2026.11480292.
- **Source estimation in EEG.** A. Bernadotte, *Estimating the number of sources
  in EEG with Hankel embedding for brain–computer interface.* Proc. ICARA,
  621–626 (2026). doi:10.1109/ICARA69401.2026.11480398.

Foundational methods:

- N. Golyandina, V. Nekrutkin, A. Zhigljavsky, *Analysis of Time Series
  Structure: SSA and Related Techniques.* Chapman & Hall/CRC (2001).
- N. Golyandina, A. Zhigljavsky, *Singular Spectrum Analysis for Time Series.*
  Springer Briefs in Statistics (2013; 2nd ed. 2020).
- V. M. Buchstaber, *Time series analysis and Grassmannians.* Amer. Math. Soc.
  Transl. 162, 1–17 (1994).
- R. Roy, T. Kailath, *ESPRIT — estimation of signal parameters via rotational
  invariance techniques.* IEEE Trans. ASSP 37, 984–995 (1989).
  doi:10.1109/29.32276.

## Citation

If you use MSSA, please cite:

> Bernadotte, A. & Menshikov, I. *Multichannel Singular Spectrum Analysis.*
> Aicumene. Preprint. 2026. https://research.aicumene.com/papers/MSSA/

The preprint will be posted shortly, and a **peer-reviewed article is
forthcoming** — please cite the published journal version once it is available.
A DOI and a machine-readable `CITATION.cff` will be added then.

## License

**PolyForm Noncommercial License 1.0.0** — free for noncommercial use (research,
education, personal and nonprofit); commercial use requires a separate license
from the authors. Copyright © 2026 Alexandra Bernadotte and Ivan Menshikov
(AiCumene). See `LICENSE`.
