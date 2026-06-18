import numpy as np

from fsb_ssg import FSBConfig, bands_for_nside, SCHEMES, FIDUCIAL


def test_derived_geometry():
    c = FSBConfig(nside=512)
    assert c.lmax == 1024
    assert c.npix == 12 * 512 ** 2
    assert np.isclose(c.omega_pix, 4 * np.pi / c.npix)
    # n_bar in 1/sr from 10/arcmin^2
    assert np.isclose(c.n_bar_sr, 10.0 * (180.0 * 60.0 / np.pi) ** 2)


def test_bands_for_nside_filters_by_lmax():
    # at NSIDE=256, lmax=512: only the first default band fits
    assert bands_for_nside(256) == ((50, 500),)
    # at NSIDE=1024 all three fit
    assert len(bands_for_nside(1024)) == 3


def test_scheme_constants():
    assert FIDUCIAL == "W1W2"
    assert set(SCHEMES) == {"none", "W2", "W1W2"}
