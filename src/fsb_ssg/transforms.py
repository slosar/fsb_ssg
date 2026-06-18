"""Spin-2 spherical-harmonic helpers and the filtered-square step.

Conventions:
  * spin-2 SHT via the healpy polarised transform with the intensity (I) part
    set to zero, so ``(g1, g2)`` are carried as ``(Q, U)``;
  * a sharp top-hat bandpass ``F_l = 1`` for ``l in [l_lo, l_hi]``;
  * the squared fields
      ``M0  = g1f^2 + g2f^2``      (spin-0 intensity),
      ``M4E = g1f^2 - g2f^2``,  ``M4B = 2 g1f g2f``   (spin-4 E/B).
"""

from __future__ import annotations

import numpy as np
import healpy as hp


def map2alm_spin(m_re, m_im, lmax):
    """``(g1, g2) = (Q, U)`` maps -> ``(alm_E, alm_B)``; the I-part is set to 0."""
    mI = np.zeros_like(m_re)
    _, aE, aB = hp.map2alm([mI, m_re, m_im], lmax=lmax, pol=True)
    return aE, aB


def alm2map_spin(aE, aB, nside):
    """``(alm_E, alm_B)`` -> ``(g1, g2) = (Q, U)`` maps."""
    aI = np.zeros_like(aE)
    _, mQ, mU = hp.alm2map([aI, aE, aB], nside=nside, pol=True)
    return mQ, mU


def bandpass(lmax, l_lo, l_hi):
    """Top-hat bandpass window ``f_l = 1`` for ``l_lo <= l <= l_hi`` else 0."""
    ell = np.arange(lmax + 1)
    w = np.zeros(lmax + 1)
    w[(ell >= l_lo) & (ell <= l_hi)] = 1.0
    return w


def shear_to_bandalm(g1, g2, W1, lmax):
    """Forward spin-2 SHT of the ``W1``-weighted shear.

    Done once per (realization, W1); the bandpass is applied later in harmonic
    space by :func:`band_fields`, so several bands reuse the same transform.
    """
    return map2alm_spin(W1 * g1, W1 * g2, lmax)


def band_fields(aE, aB, l_lo, l_hi, nside, lmax):
    """Apply the top-hat band, inverse SHT, and square.

    Returns ``(M0, M4E, M4B)``.  By construction ``M4E**2 + M4B**2 == M0**2``.
    """
    win = bandpass(lmax, l_lo, l_hi)
    g1f, g2f = alm2map_spin(hp.almxfl(aE, win), hp.almxfl(aB, win), nside)
    m0 = g1f ** 2 + g2f ** 2
    m4e = g1f ** 2 - g2f ** 2
    m4b = 2.0 * g1f * g2f
    return m0, m4e, m4b
