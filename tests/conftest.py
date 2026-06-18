"""Shared fixtures for the fsb_ssg unit tests.

All tests run on small, synthetic maps (no survey data required): a smooth
source-density field, a polar-cap mask, and random Gaussian shear, all at a low
NSIDE so the whole suite runs in seconds.
"""

import numpy as np
import healpy as hp
import pytest

from fsb_ssg import FSBConfig

NSIDE = 64


@pytest.fixture(scope="session")
def config():
    """Small-NSIDE config with a single in-range band."""
    return FSBConfig(nside=NSIDE, dell=16, filter_bands=((10, 40),))


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def mask():
    """Binary polar-cap mask (~15% of the sky)."""
    npix = hp.nside2npix(NSIDE)
    disc = hp.query_disc(NSIDE, hp.ang2vec(0.0, 0.0), radius=1.0)
    m = np.zeros(npix)
    m[disc] = 1.0
    return m


@pytest.fixture(scope="session")
def delta_source():
    """Smooth source-density contrast in a safe range (1 + delta > 0)."""
    cl = 1.0 / (np.arange(1, 3 * NSIDE) ** 2)
    cl = np.concatenate([[0.0], cl])
    m = hp.synfast(0.02 * cl, NSIDE, lmax=2 * NSIDE)
    return np.clip(m, -0.8, 5.0)


@pytest.fixture(scope="session")
def shear(rng):
    """A random band-limited spin-2 shear field ``(g1, g2)`` at NSIDE."""
    lmax = 2 * NSIDE
    cl = np.concatenate([[0.0, 0.0], 1.0 / np.arange(2, lmax + 1) ** 2])
    aE = hp.synalm(cl, lmax=lmax)
    aB = hp.synalm(0.3 * cl, lmax=lmax)
    aI = np.zeros_like(aE)
    _, g1, g2 = hp.alm2map([aI, aE, aB], nside=NSIDE, pol=True)
    return g1, g2
