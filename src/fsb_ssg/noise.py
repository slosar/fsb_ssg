"""Inhomogeneous shape-noise model, optimal weights, and the M0 noise bias.

The shape-noise variance per component is

    n^2(n) = sigma_e^2 / ( n_bar Omega_pix (1 + delta_src(n)) ),

largest in voids (where the source density is low) and correlated with the lens
density.  Two weights are built from it:

  * ``W1`` (pre-filter): inverse-variance ``1/n^2`` for the fiducial scheme,
    binary mask otherwise;
  * ``W2`` (post-square): the deterministic optimal map ``E[M0]_sig / Var[M0]``.

Because NaMaster mode-decouples the ``W2`` field weight, the decoupled
bandpowers are unbiased for *any* ``W2`` shape and overall normalisation; ``W2``
only changes the variance (SNR).  The additive ``M0`` noise mean is removed with
an exact ``|K|^2`` template (:func:`noise_bias_template`); ``M4`` has zero noise
mean by Wick and needs no debiasing.
"""

from __future__ import annotations

import numpy as np

from .kernels import kernel2_conv


def noise_variance(delta_source, mask, config):
    """Per-component shape-noise variance ``n^2(n)`` inside the mask (0 outside)."""
    n2 = config.sigma_e ** 2 / np.clip(
        config.n_bar_sr * config.omega_pix * (1.0 + delta_source), 1e-3, None)
    return n2 * mask


def make_noise(n2, mask, rng):
    """Draw one inhomogeneous shape-noise realization ``(eps1, eps2)``."""
    sig = np.zeros_like(n2)
    inm = mask > 0
    sig[inm] = np.sqrt(n2[inm])
    e1 = rng.normal(0.0, 1.0, n2.size) * sig
    e2 = rng.normal(0.0, 1.0, n2.size) * sig
    return e1, e2


def make_W1(n2, mask, scheme):
    """Pre-filter weight map.

    ``'none'`` / ``'W2'`` -> binary mask; ``'W1W2'`` -> ``1/n^2`` normalised to
    unit in-mask mean (the normalisation is cosmetic; it cancels downstream).
    """
    if scheme in ("none", "W2"):
        return mask.copy()
    inm = mask > 0
    W1 = np.zeros_like(n2)
    W1[inm] = 1.0 / n2[inm]
    W1 /= W1[inm].mean()
    return W1


def make_W2(n2, mask, scheme, l_lo, l_hi, c_band):
    """Optimal post-square weight ``W2 ~ E[M0]_sig / Var[M0]`` (deterministic).

    Built from filter-smoothed powers of the noise map.  The overall
    normalisation is irrelevant (NaMaster deconvolves it).  For scheme ``'none'``
    returns the binary mask (uniform weight).

    ``c_band`` is the band-averaged per-component signal density (see
    :func:`band_signal_c`).
    """
    if scheme == "none":
        return mask.copy()
    inm = mask > 0
    W1 = make_W1(n2, mask, scheme)
    w1sq = W1 ** 2
    # signal mean  ~ |K|^2 * (W1^2 c);  total variance = (|K|^2 * (W1^2 (n^2+c)))^2
    Esig = kernel2_conv(w1sq * c_band * mask, l_lo, l_hi)
    Etot = kernel2_conv(w1sq * (n2 + c_band * mask), l_lo, l_hi)
    W2 = np.zeros_like(n2)
    W2[inm] = Esig[inm] / np.clip(Etot[inm] ** 2, 1e-300, None)
    W2 /= W2[inm].mean()
    return W2 * mask


def noise_bias_template(n2, mask, scheme, l_lo, l_hi):
    """Deterministic ``E[M0_noise]`` for the W1-weighted noise (bl2beam ``|K|^2``)."""
    W1 = make_W1(n2, mask, scheme)
    return kernel2_conv(W1 ** 2 * n2, l_lo, l_hi)


def band_signal_c(cl_ee, cl_bb, l_lo, l_hi, config):
    """Per-component white-band signal density ``c`` (slowly-varying approx).

    Used only to set the *shape* of the optimal ``W2``; an overall factor
    cancels in the decoupling.
    """
    ell = np.arange(cl_ee.size)
    inb = (ell >= l_lo) & (ell <= l_hi)
    w = (2 * ell + 1)[inb]
    cbar = np.sum(w * (cl_ee + cl_bb)[inb]) / np.sum(w)
    return cbar / (2.0 * config.omega_pix)
