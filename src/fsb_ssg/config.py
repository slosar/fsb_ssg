"""Configuration for the FSB-SSG estimator.

A single :class:`FSBConfig` holds the resolution and the survey / noise
parameters; everything else in the library derives geometry (``lmax``,
``npix``, ``omega_pix``) and the noise normalisation (``n_bar_sr``) from it,
so the rest of the code never has to hard-code these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Weighting schemes understood throughout the library.  'W1W2' is the fiducial
# (full optimal) scheme; see the package docstring for the definitions.
SCHEMES = ("none", "W2", "W1W2")
FIDUCIAL = "W1W2"

# Default bandpass bands (the "filter" scale of the FSB) for production runs at
# NSIDE >= 1024.  At lower NSIDE only bands with l_hi <= lmax = 2*nside are
# usable; use :func:`bands_for_nside` to pick a valid subset.
DEFAULT_FILTER_BANDS = ((50, 500), (500, 1000), (1000, 1500))


def bands_for_nside(nside, bands=DEFAULT_FILTER_BANDS):
    """Return the subset of ``bands`` whose upper edge fits below ``2*nside``."""
    lmax = 2 * nside
    return tuple(b for b in bands if b[1] <= lmax)


@dataclass(frozen=True)
class FSBConfig:
    """Resolution and survey/noise parameters.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the working maps.  ``lmax`` is fixed to
        ``2 * nside``.
    sigma_e : float
        Per-component shape-noise rms (ellipticity dispersion).
    n_gal_arcmin2 : float
        Mean source number density in galaxies / arcmin^2.
    dell : int
        Linear NaMaster bandpower width for the cross multipole ``l``.
    filter_bands : tuple of (int, int)
        Top-hat bandpass bands ``[l_lo, l_hi]`` applied to the shear before
        squaring.
    """

    nside: int = 1024
    sigma_e: float = 0.26
    n_gal_arcmin2: float = 10.0
    dell: int = 48
    filter_bands: tuple = DEFAULT_FILTER_BANDS

    @property
    def lmax(self) -> int:
        return 2 * self.nside

    @property
    def npix(self) -> int:
        return 12 * self.nside ** 2

    @property
    def omega_pix(self) -> float:
        """Solid angle per pixel (steradian)."""
        return 4.0 * np.pi / self.npix

    @property
    def n_bar_sr(self) -> float:
        """Mean source number density in galaxies / steradian."""
        return self.n_gal_arcmin2 * (180.0 * 60.0 / np.pi) ** 2
