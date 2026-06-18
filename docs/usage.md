# Using `fsb_ssg`

This walk-through covers a single-realization measurement and the building
blocks behind it. The scientific motivation and the full derivation of the
weights and the transfer function are in the report under
[`../science/report/`](../science/report/).

## 1. Configuration

Everything starts from an [`FSBConfig`](../src/fsb_ssg/config.py): it fixes the
resolution and the survey / noise parameters and derives the geometry.

```python
from fsb_ssg import FSBConfig
cfg = FSBConfig(
    nside=1024,                  # lmax is fixed to 2*nside
    sigma_e=0.26,                # per-component shape-noise rms
    n_gal_arcmin2=10.0,          # source number density
    dell=48,                     # NaMaster bandpower width for the cross multipole
    filter_bands=[(50, 500), (500, 1000), (1000, 1500)],
)
cfg.lmax, cfg.npix, cfg.omega_pix, cfg.n_bar_sr   # derived quantities
```

`bands_for_nside(nside)` returns the default bands whose upper edge fits below
`2*nside`, which is handy for low-resolution validation runs.

## 2. The estimator object

`FSBEstimator` binds a config to the building blocks and caches the NaMaster
bins:

```python
from fsb_ssg import FSBEstimator
est = FSBEstimator(cfg)
est.ells          # effective multipole of each bandpower
est.n_bands       # number of bandpowers
```

## 3. Noise model and weights

The shape-noise variance per component is

```
n^2(n) = sigma_e^2 / ( n_bar · Omega_pix · (1 + delta_src(n)) )
```

— largest in voids, and correlated with the lens density. From it we build the
pre-filter weight `W1` and the optimal post-square weight `W2`:

```python
n2 = est.noise_variance(delta_src, mask)            # 0 outside the mask
W1 = est.make_W1(n2, mask, "W1W2")                  # 1/n^2 (or the mask for none/W2)
c  = est.band_signal_c(cl_ee, cl_bb, l_lo, l_hi)    # band-averaged signal density
W2 = est.make_W2(n2, mask, "W1W2", l_lo, l_hi, c)   # optimal inverse-variance map
```

`W2` is deterministic in the (known) noise map and the filter. Its overall
normalization is irrelevant: NaMaster deconvolves it.

## 4. Filtered-square step

The shear is `W1`-weighted, forward-transformed once, then for each band
bandpass-filtered, inverse-transformed and squared:

```python
m0, m4e, m4b = est.filtered_square(g1, g2, W1, l_lo, l_hi)
# m0 = g1f^2 + g2f^2     (spin-0 intensity)
# m4e = g1f^2 - g2f^2,  m4b = 2 g1f g2f    (spin-4 E/B);  m4e^2 + m4b^2 == m0^2
```

If you reuse the same `W1` across several bands, split the transform to avoid
repeating it:

```python
aE, aB = est.shear_to_bandalm(g1, g2, W1)           # forward SHT once
for (l_lo, l_hi) in cfg.filter_bands:
    m0, m4e, m4b = est.band_fields(aE, aB, l_lo, l_hi)
```

## 5. Noise debiasing

`M0` has a non-zero noise mean that, because the noise tracks `delta_src`,
cross-correlates with the lens density and **must** be subtracted. It is given
exactly by the squared-kernel template:

```python
bias = est.noise_bias_template(n2, mask, "W1W2", l_lo, l_hi)
m0_debiased = m0 - bias
```

`M4` has zero noise mean (Wick) and needs no debiasing.

## 6. Cross-spectra through NaMaster

`delta` carries the binary mask; the squared field carries the `W2` weight. The
coupling matrices depend only on the `(mask, W2)` pair, so build them once and
reuse them across realizations:

```python
ws00, ws04 = est.workspaces(mask, W2)
f0d = est.field0(mask, (delta - delta[mask > 0].mean()) * mask)

cl_m0      = est.cross_m0(f0d, m0_debiased, W2, ws00)     # C_l^{delta M0}
cl_m4e, cl_m4b = est.cross_m4(f0d, m4e, m4b, W2, ws04)    # E (signal) and B (null)
```

## 7. Transfer function and covariance

Because `W1` is applied **before** the square (and the mask leaks power across
its edge), the masked, weighted estimator differs from the full-sky truth by a
deterministic transfer function `T_l`; results are reported in truth units as
`C_l / T_l`. `T_l` is calculable from the mask, the weights, the filter and the
shear power spectrum alone (no signal simulations) — see the report appendix and
`science/pipeline/predict_transfer.py`.

NaMaster also supplies a Gaussian covariance via `pymaster.gaussian_covariance`
(see `science/pipeline/nmt_covariance.py`), which the validation shows matches
the realization scatter.

## Functional API

Every method above has a plain-function counterpart that takes explicit
arguments instead of a config, exported at the package top level: `noise_variance`,
`make_W1`, `make_W2`, `noise_bias_template`, `band_signal_c`, `shear_to_bandalm`,
`band_fields`, `kernel2_conv`, `response_beam`, `make_bins`, `field0`,
`workspace00`, `workspace04`, `cross_M0`, `cross_M4`. Use whichever style fits.
