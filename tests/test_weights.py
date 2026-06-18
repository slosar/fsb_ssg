import numpy as np

from fsb_ssg import (
    noise_variance, make_W1, make_W2, noise_bias_template, band_signal_c,
)


def test_make_W1_schemes(config, delta_source, mask):
    n2 = noise_variance(delta_source, mask, config)
    # unweighted schemes -> binary mask
    np.testing.assert_array_equal(make_W1(n2, mask, "none"), mask)
    np.testing.assert_array_equal(make_W1(n2, mask, "W2"), mask)
    # inverse-variance scheme
    W1 = make_W1(n2, mask, "W1W2")
    inm = mask > 0
    assert np.all(W1[~inm] == 0.0)
    assert np.all(W1[inm] > 0.0)
    assert np.isclose(W1[inm].mean(), 1.0)          # unit in-mask mean
    # monotone in 1/n^2
    order = np.argsort(n2[inm])
    assert np.all(np.diff(W1[inm][order]) <= 1e-12)  # larger n2 -> smaller W1


def test_make_W2_properties(config, delta_source, mask):
    n2 = noise_variance(delta_source, mask, config)
    l_lo, l_hi = config.filter_bands[0]
    c = 1.0
    # scheme 'none' -> uniform (mask) weight
    np.testing.assert_array_equal(make_W2(n2, mask, "none", l_lo, l_hi, c), mask)
    for scheme in ("W2", "W1W2"):
        W2 = make_W2(n2, mask, scheme, l_lo, l_hi, c)
        inm = mask > 0
        assert np.all(W2[~inm] == 0.0)
        assert np.all(W2[inm] > 0.0)
        assert np.isclose(W2[inm].mean(), 1.0)
        assert np.all(np.isfinite(W2))


def test_noise_bias_template_positive(config, delta_source, mask):
    n2 = noise_variance(delta_source, mask, config)
    l_lo, l_hi = config.filter_bands[0]
    B = noise_bias_template(n2, mask, "W1W2", l_lo, l_hi)
    # E[M0_noise] is a (smoothed) positive density inside the footprint
    inm = mask > 0
    assert np.all(B[inm] > 0.0)


def test_band_signal_c_positive(config):
    lmax = config.lmax
    cl_ee = np.full(lmax + 1, 2.0)
    cl_bb = np.full(lmax + 1, 1.0)
    c = band_signal_c(cl_ee, cl_bb, 10, 40, config)
    # c = <C^EE+C^BB>/(2 Omega_pix) = 3/(2 Omega_pix)
    assert np.isclose(c, 3.0 / (2.0 * config.omega_pix))
