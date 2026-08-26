# Retired Box-Cox gravity challenger

This directory preserves the experimental Box-Cox distance-decay challenger
retired from the active package in August 2026. It is historical evidence, not
an executable or supported package entry point.

The candidate model was

\[
P(j\mid i) \propto V_j^\alpha
\exp\{-\gamma g_\lambda(d_{ij}/1\ \mathrm{km})\},\qquad
g_\lambda(d)=\frac{d^\lambda-1}{\lambda},
\]

with the `lambda = 0` limit equal to `log(d)`. Three-fold merchant-grouped
cross-validation selected `lambda = 1.5`; the full-sample estimates were
`alpha = 0.4227767882` and `gamma = 1.0856072751`. Although the full-sample
distance Wasserstein error fell from 71.6 m to 41.2 m, the candidate failed its
pre-registered acceptance rule because the combined held-out absolute mean
distance bias increased from about 20.69 m to 21.72 m. It was never activated
in `experiment.toml`.

Historical implementation files:

- `challenger.py`: selection, fitting, evaluation, acceptance, and outputs.
- `gravity.py`: Box-Cox transformation and production-weight helper.
- `boxcox_tests.py`: synthetic and integration tests from the trial.

The implementation depended on Box-Cox helper functions that existed in the
active `gravity_model.py` and `gravity_plots.py` at commit `4442f31`. Consult
repository history to reconstruct a runnable snapshot. The real-data audit
outputs remain ignored runtime artifacts under
`outputs/work/calibration/gravity_joint/boxcox_challenger/a45720a0a6f16935/`.

Method references:

- Kau, J. B. and Sirmans, C. F. (1979), *The Functional Form of the Gravity
  Model*, https://doi.org/10.1177/016001767900400202.
- Fik, T. J. and Mulligan, G. F. (1998), *Functional Form and Spatial
  Interaction Models*, https://doi.org/10.1068/a301497.
