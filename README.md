# fsb_ssg

[![CI](https://github.com/slosar/fsb_ssg/actions/workflows/ci.yml/badge.svg)](https://github.com/slosar/fsb_ssg/actions/workflows/ci.yml)

**F**iltered-**S**quare **B**ispectrum estimator for the **s**hear–**s**hear–**g**alaxy field.

`fsb_ssg` measures the shear–shear–density bispectrum by **filtering and squaring**
the weak-lensing shear field and cross-correlating the squared field with the
lens galaxy density, using [NaMaster](https://github.com/LSSTDESC/NaMaster) for
the mode decoupling and the Gaussian covariance. It handles a survey mask,
spatially inhomogeneous shape noise, and optimal pre- and post-square weighting.

```
gamma --W1--> W1·gamma --F(band)--> xi --square--> M0, M4
                                                |
                                   (W2 weight, via NaMaster)
                                                v
                       cross-correlate with lens delta  ->  C_l^{delta M}
```

* `M0 = |xi|^2` is a spin-0 field; `M4 = xi^2` is spin-4 (E/B).
* `W1` (pre-filter) is an inverse-variance shear weight; `W2` (post-square) is the
  optimal inverse-variance weight on the squared field, supplied as the NaMaster
  field weight. **NaMaster mode-decouples `W2`, so the decoupled bandpowers are
  unbiased for any `W2`** — `W2` only changes the variance (the SNR).
* The additive `M0` noise bias is removed with an exact `|K|^2` (`bl2beam`)
  template; `M4` has zero noise mean.

The estimator is validated against 100 full-sky Takahashi-like simulations and a
realistic DES Y6 mask; see [`science/`](science/) for the full validation and the
accompanying report.

## Installation

`fsb_ssg` depends on `numpy`, [`healpy`](https://healpy.readthedocs.io) and
[`pymaster`](https://namaster.readthedocs.io) (NaMaster). The most reliable way
to get NaMaster is from conda-forge:

```bash
conda install -c conda-forge healpy namaster numpy
pip install fsb_ssg            # or:  pip install -e .  (from a clone)
```

Pure pip also works if the NaMaster system libraries (GSL, FFTW, cfitsio) are
present:

```bash
pip install fsb_ssg
```

## Weighting schemes

| scheme  | `W1`      | `W2`     | description                        |
|---------|-----------|----------|------------------------------------|
| `none`  | `1`       | `1`      | binary mask only                   |
| `W2`    | `1`       | optimal  | post-square inverse-variance       |
| `W1W2`  | `1/n^2`   | optimal  | full optimal weighting (fiducial)  |

## Quick start

```python
import numpy as np
from fsb_ssg import FSBConfig, FSBEstimator

cfg = FSBConfig(nside=1024, filter_bands=[(50, 500), (500, 1000), (1000, 1500)])
est = FSBEstimator(cfg)
l_lo, l_hi = 500, 1000

# inputs: shear (g1, g2), lens density `delta`, source density `delta_src`, mask
n2  = est.noise_variance(delta_src, mask)                 # inhomogeneous shape noise
W1  = est.make_W1(n2, mask, "W1W2")                       # pre-filter weight
c   = est.band_signal_c(cl_ee, cl_bb, l_lo, l_hi)         # band signal density
W2  = est.make_W2(n2, mask, "W1W2", l_lo, l_hi, c)        # post-square weight

ws00, ws04 = est.workspaces(mask, W2)                     # NaMaster coupling (reuse!)
f0d  = est.field0(mask, (delta - delta[mask > 0].mean()) * mask)

m0, m4e, m4b = est.filtered_square(g1, g2, W1, l_lo, l_hi)
bias = est.noise_bias_template(n2, mask, "W1W2", l_lo, l_hi)

cl_m0      = est.cross_m0(f0d, m0 - bias, W2, ws00)       # spin-0 x spin-0
cl_m4e, _  = est.cross_m4(f0d, m4e, m4b, W2, ws04)        # spin-0 x spin-4 (E, B)
ells       = est.ells
```

See [`docs/usage.md`](docs/usage.md) for a fuller walk-through, including the
transfer-function calibration and the covariance.

## Repository layout

```
src/fsb_ssg/        the installable library (config, transforms, noise, kernels, estimator)
tests/              unit tests (synthetic maps; no survey data needed) — run with pytest
docs/               usage documentation
science/            simulation-based validation + LaTeX report (data not included)
  pipeline/         staged measurement scripts + the simlib adapter
  report/           the validation report (LaTeX) and its figures
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

The unit tests run on small synthetic maps in a few seconds (no survey data),
and on GitHub Actions for Python 3.11 and 3.12.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
