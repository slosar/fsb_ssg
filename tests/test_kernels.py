import numpy as np
import healpy as hp

from fsb_ssg import response_beam, kernel2_conv


def test_response_beam_white_reduces_to_K2():
    """For a white shear spectrum, kappa reduces to |K|^2 (normalised to 1)."""
    nside = 64
    lmax = 2 * nside
    l_lo, l_hi = 10, 40
    cgamma = np.full(lmax + 1, 3.7)          # constant (white)
    bk = response_beam(l_lo, l_hi, cgamma, lmax)
    # reference: beam2bl(|K|^2), normalised at l=0
    ell = np.arange(lmax + 1)
    f = ((ell >= l_lo) & (ell <= l_hi)).astype(float)
    theta = np.linspace(0, np.pi, 20000)
    ref = hp.beam2bl(hp.bl2beam(f, theta) ** 2, theta, lmax)
    ref = ref / ref[0]
    np.testing.assert_allclose(bk, ref, rtol=1e-6, atol=1e-8)
    assert np.isclose(bk[0], 1.0)


def test_kernel2_conv_constant_input():
    """|K|^2-smoothing a constant map gives a constant = 2 Omega_pix c0 K2_0."""
    nside = 64
    l_lo, l_hi = 5, 30
    c0 = 1.7
    out = kernel2_conv(np.full(hp.nside2npix(nside), c0), l_lo, l_hi)
    # independent expectation from the same |K|^2 monopole
    lmax = 3 * nside - 1
    ell = np.arange(lmax + 1)
    omega = 4.0 * np.pi / (12 * nside ** 2)
    W = ((ell >= l_lo) & (ell <= l_hi)).astype(float)
    theta = np.linspace(0, np.pi, 20000)
    K2_0 = hp.beam2bl(hp.bl2beam(W, theta) ** 2, theta, lmax)[0]
    expect = 2.0 * omega * c0 * K2_0
    assert np.allclose(out, expect, rtol=1e-3)


def test_kernel2_conv_linear():
    nside = 32
    npix = hp.nside2npix(nside)
    a = np.abs(np.random.default_rng(0).normal(size=npix)) + 0.1
    b = np.abs(np.random.default_rng(1).normal(size=npix)) + 0.1
    ka = kernel2_conv(a, 8, 24)
    kb = kernel2_conv(b, 8, 24)
    kab = kernel2_conv(a + 3.0 * b, 8, 24)
    np.testing.assert_allclose(kab, ka + 3.0 * kb, rtol=1e-6, atol=1e-8)
