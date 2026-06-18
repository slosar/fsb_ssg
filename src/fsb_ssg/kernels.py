"""Real-space filter kernels used by the noise model and the transfer function.

The bandpass ``f_l`` has a real-space kernel ``K(theta) = bl2beam(f_l)``.  Two
derived objects appear repeatedly:

  * ``|K|^2`` smoothing (:func:`kernel2_conv`) builds the exact mean of the
    filtered-squared field for a given input variance density -- used both for
    the optimal post-square weight and for the additive M0 noise bias template;
  * the squeezed-response kernel ``kappa`` (:func:`response_beam`) is the
    harmonic profile of the linear response of the filtered-squared field to a
    long-wavelength density mode, used by the analytic transfer function.
"""

from __future__ import annotations

import numpy as np
import healpy as hp


def kernel2_conv(input_map, l_lo, l_hi, nside=None, n_theta=20000):
    """Exact ``2 * Omega_pix * [ |K|^2 (*) input_map ]`` via the bl2beam route.

    With ``input_map`` a per-component variance density this returns the
    filtered-squared mean ``E[M0]`` summed over the two shear components.

    Parameters
    ----------
    input_map : ndarray
        HEALPix map of the per-component variance density to convolve.
    l_lo, l_hi : int
        Top-hat band edges defining the kernel ``K``.
    nside : int, optional
        Resolution; inferred from ``input_map`` when not given.
    n_theta : int
        Number of sample points for the bl2beam / beam2bl transforms.
    """
    if nside is None:
        nside = hp.get_nside(input_map)
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)
    omega = 4.0 * np.pi / (12 * nside ** 2)
    W = ((ell >= l_lo) & (ell <= l_hi)).astype(float)
    theta = np.linspace(0, np.pi, n_theta)
    K_theta = hp.bl2beam(W, theta)
    K2_ell = hp.beam2bl(K_theta ** 2, theta, lmax)
    return 2.0 * omega * hp.smoothing(input_map, beam_window=K2_ell)


def response_beam(l_lo, l_hi, cgamma, lmax, n_theta=20000):
    """Harmonic profile ``b^kappa_l`` of the squeezed-response kernel.

        b^kappa_l = beam2bl[ bl2beam(f_l) * bl2beam(f_l C^gg_l) ]

    normalised to 1 at ``l = 0``.  The delta-correlated part of the
    filtered-squared field is ``kappa * (W1^2 delta)``.  It reduces to the
    noise ``|K|^2`` kernel when ``C^gg`` is white.

    Parameters
    ----------
    l_lo, l_hi : int
        Top-hat band edges.
    cgamma : ndarray
        Per-component shear power spectrum ``C^gg_l = (C^EE + C^BB) / 2``.
    lmax : int
        Maximum multipole of the returned profile.
    """
    ell = np.arange(lmax + 1)
    f = ((ell >= l_lo) & (ell <= l_hi)).astype(float)
    theta = np.linspace(0, np.pi, n_theta)
    bk = hp.beam2bl(
        hp.bl2beam(f, theta) * hp.bl2beam(f * cgamma[: lmax + 1], theta),
        theta, lmax)
    return bk / bk[0]
