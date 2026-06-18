"""NaMaster mode-decoupled cross-spectra and a high-level estimator class.

``M0`` is crossed with the lens density ``delta`` as spin-0 x spin-0; the
spin-4 field ``(M4E, M4B)`` is crossed with ``delta`` as spin-0 x spin-4 (the
E component is the signal, B a null test).  The post-square weight ``W2`` is
supplied as the NaMaster field weight on the squared map, and ``delta`` carries
the binary mask; NaMaster's coupling matrix then deconvolves the joint effect of
the mask and ``W2``.

The :class:`FSBEstimator` class binds an :class:`~fsb_ssg.config.FSBConfig` to
these building blocks so a full measurement can be expressed without repeating
``nside`` / ``lmax`` / bin arguments.
"""

from __future__ import annotations

import numpy as np
import pymaster as nmt

from .config import FSBConfig
from . import noise as _noise
from . import transforms as _transforms


# --------------------------------------------------------------------------- #
# Low-level NaMaster wrappers
# --------------------------------------------------------------------------- #
def make_bins(lmax, dell):
    """Linear NaMaster bandpowers of width ``dell`` up to ``lmax``."""
    return nmt.NmtBin.from_lmax_linear(lmax, dell)


def field0(mask, m, lmax):
    """Spin-0 NaMaster field of map ``m`` with weight ``mask``."""
    return nmt.NmtField(mask, [m], lmax=lmax, spin=0)


def workspace00(mask_delta, weight_M, bins):
    """spin-0 x spin-0 coupling matrix for the ``(delta, M0)`` weight pair."""
    f0d = nmt.NmtField(mask_delta, [mask_delta], lmax=bins.lmax)
    f0M = nmt.NmtField(weight_M, [weight_M], lmax=bins.lmax)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f0d, f0M, bins)
    return w


def workspace04(mask_delta, weight_M, bins):
    """spin-0 x spin-4 coupling matrix for the ``(delta, M4)`` weight pair."""
    f0d = nmt.NmtField(mask_delta, [mask_delta], lmax=bins.lmax, spin=0)
    f4M = nmt.NmtField(weight_M, [weight_M, weight_M], lmax=bins.lmax, spin=4)
    w = nmt.NmtWorkspace()
    w.compute_coupling_matrix(f0d, f4M, bins)
    return w


def cross_M0(f0_delta, m0_map, weight_M, ws00, lmax):
    """Decoupled ``C_l^{delta M0}`` bandpowers."""
    fM = nmt.NmtField(weight_M, [m0_map], lmax=lmax, spin=0)
    return ws00.decouple_cell(nmt.compute_coupled_cell(f0_delta, fM))[0]


def cross_M4(f0_delta, m4e, m4b, weight_M, ws04, lmax):
    """Decoupled ``(C_l^{delta M4E}, C_l^{delta M4B})`` bandpowers."""
    f4 = nmt.NmtField(weight_M, [m4e, m4b], lmax=lmax, spin=4)
    cl = ws04.decouple_cell(nmt.compute_coupled_cell(f0_delta, f4))
    return cl[0], cl[1]


# --------------------------------------------------------------------------- #
# High-level estimator
# --------------------------------------------------------------------------- #
class FSBEstimator:
    """Filtered-square bispectrum estimator bound to a configuration.

    The estimator caches its NaMaster bins; everything else is built on demand.
    A typical single-realization measurement is::

        est = FSBEstimator(FSBConfig(nside=1024))
        n2  = est.noise_variance(delta_src, mask)
        W1  = est.make_W1(n2, mask, "W1W2")
        W2  = est.make_W2(n2, mask, "W1W2", l_lo, l_hi, c)
        ws00, ws04 = est.workspaces(mask, W2)               # reuse across reals
        f0d = est.field0(mask, delta_lens_masked)
        m0, m4e, m4b = est.filtered_square(g1, g2, W1, l_lo, l_hi)
        bias = est.noise_bias_template(n2, mask, "W1W2", l_lo, l_hi)
        cl_m0  = est.cross_m0(f0d, m0 - bias, W2, ws00)
        cl_m4e, _ = est.cross_m4(f0d, m4e, m4b, W2, ws04)
    """

    def __init__(self, config=None):
        self.config = config if config is not None else FSBConfig()
        self.bins = make_bins(self.config.lmax, self.config.dell)

    # -- geometry ---------------------------------------------------------- #
    @property
    def ells(self):
        """Effective multipole of each NaMaster bandpower."""
        return self.bins.get_effective_ells()

    @property
    def n_bands(self):
        return self.bins.get_n_bands()

    # -- noise & weights (bound to config) --------------------------------- #
    def noise_variance(self, delta_source, mask):
        return _noise.noise_variance(delta_source, mask, self.config)

    def make_noise(self, n2, mask, rng):
        return _noise.make_noise(n2, mask, rng)

    @staticmethod
    def make_W1(n2, mask, scheme):
        return _noise.make_W1(n2, mask, scheme)

    @staticmethod
    def make_W2(n2, mask, scheme, l_lo, l_hi, c_band):
        return _noise.make_W2(n2, mask, scheme, l_lo, l_hi, c_band)

    @staticmethod
    def noise_bias_template(n2, mask, scheme, l_lo, l_hi):
        return _noise.noise_bias_template(n2, mask, scheme, l_lo, l_hi)

    def band_signal_c(self, cl_ee, cl_bb, l_lo, l_hi):
        return _noise.band_signal_c(cl_ee, cl_bb, l_lo, l_hi, self.config)

    # -- filtered-square step ---------------------------------------------- #
    def shear_to_bandalm(self, g1, g2, W1):
        return _transforms.shear_to_bandalm(g1, g2, W1, self.config.lmax)

    def band_fields(self, aE, aB, l_lo, l_hi):
        return _transforms.band_fields(
            aE, aB, l_lo, l_hi, self.config.nside, self.config.lmax)

    def filtered_square(self, g1, g2, W1, l_lo, l_hi):
        """``(g1, g2)`` -> ``(M0, M4E, M4B)`` for one band (full SHT each call)."""
        aE, aB = self.shear_to_bandalm(g1, g2, W1)
        return self.band_fields(aE, aB, l_lo, l_hi)

    # -- NaMaster cross-spectra -------------------------------------------- #
    def field0(self, mask, m):
        return field0(mask, m, self.config.lmax)

    def workspaces(self, mask, weight_M):
        """Return ``(ws00, ws04)`` coupling matrices for one ``(mask, W2)`` pair."""
        return (workspace00(mask, weight_M, self.bins),
                workspace04(mask, weight_M, self.bins))

    def cross_m0(self, f0_delta, m0_map, weight_M, ws00):
        return cross_M0(f0_delta, m0_map, weight_M, ws00, self.config.lmax)

    def cross_m4(self, f0_delta, m4e, m4b, weight_M, ws04):
        return cross_M4(f0_delta, m4e, m4b, weight_M, ws04, self.config.lmax)
