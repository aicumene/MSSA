# Repository boundaries

## MSSA
Owns generic multichannel singular-spectrum functionality:
- block-Hankel construction
- lagged-scatter decomposition
- singular values/eigenvectors
- generic rank estimation
- grouped reconstruction
- singular-spectrum diagnostics

## BOID
Owns BOID-specific functionality:
- TLS-ESPRIT
- pole extraction
- positive-frequency oscillator counting
- BOID descriptors
- paper-level analyses and classifiers

BOID should depend on `aicumene-mssa` rather than duplicate MSSA internals.
