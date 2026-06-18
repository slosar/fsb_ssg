import numpy as np
import healpy as hp

from fsb_ssg import bandpass, map2alm_spin, alm2map_spin, band_fields, shear_to_bandalm


def test_bandpass_window():
    w = bandpass(100, 10, 40)
    assert w.shape == (101,)
    assert np.all(w[10:41] == 1.0)
    assert w[9] == 0.0 and w[41] == 0.0
    assert w.sum() == 31


def test_spin2_roundtrip_recovers_power(shear):
    """alm -> map -> alm should recover the low-l shear power (transfer ~ 1)."""
    g1, g2 = shear
    nside = hp.get_nside(g1)
    lmax = 2 * nside
    aE, aB = map2alm_spin(g1, g2, lmax)
    g1b, g2b = alm2map_spin(aE, aB, nside)
    aE2, aB2 = map2alm_spin(g1b, g2b, lmax)
    # transfer t_l = Re<a_in a_rec*> / <|a_in|^2> on the E mode, low-l
    cross = hp.alm2cl(aE, aE2)
    auto = hp.alm2cl(aE)
    t = cross[2:30] / auto[2:30]
    assert np.all(np.abs(t - 1.0) < 0.05)


def test_band_fields_algebraic_identity(shear):
    """M4E^2 + M4B^2 == M0^2 exactly, and M0 >= 0."""
    g1, g2 = shear
    nside = hp.get_nside(g1)
    lmax = 2 * nside
    aE, aB = shear_to_bandalm(g1, g2, np.ones_like(g1), lmax)
    m0, m4e, m4b = band_fields(aE, aB, 10, 40, nside, lmax)
    assert np.all(m0 >= -1e-12)
    np.testing.assert_allclose(m4e ** 2 + m4b ** 2, m0 ** 2, rtol=1e-10, atol=1e-12)


def test_band_fields_zero_outside_band():
    """A field with no power in the band yields a vanishing squared field."""
    nside = 32
    lmax = 2 * nside
    # power only at l < 10, band is [20, 40] -> filtered field is ~0
    cl = np.zeros(lmax + 1)
    cl[2:8] = 1.0
    aE = hp.synalm(cl, lmax=lmax)
    aB = np.zeros_like(aE)
    m0, m4e, m4b = band_fields(aE, aB, 20, 40, nside, lmax)
    assert np.max(np.abs(m0)) < 1e-10
