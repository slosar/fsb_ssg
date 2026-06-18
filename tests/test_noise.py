import numpy as np

from fsb_ssg import noise_variance, make_noise


def test_noise_variance_formula_and_support(config, delta_source, mask):
    n2 = noise_variance(delta_source, mask, config)
    # zero outside the mask
    assert np.all(n2[mask == 0] == 0.0)
    # correct formula inside
    inm = mask > 0
    expect = config.sigma_e ** 2 / (
        config.n_bar_sr * config.omega_pix * (1.0 + delta_source[inm]))
    np.testing.assert_allclose(n2[inm], expect, rtol=1e-12)


def test_noise_variance_scales_inversely_with_density(config, mask):
    """Higher source density -> lower shape-noise variance."""
    npix = mask.size
    low = np.zeros(npix)
    high = np.full(npix, 1.0)
    n2_low = noise_variance(low, mask, config)
    n2_high = noise_variance(high, mask, config)
    inm = mask > 0
    np.testing.assert_allclose(n2_high[inm], 0.5 * n2_low[inm], rtol=1e-12)


def test_make_noise_matches_variance(config, mask, rng):
    """Empirical per-component variance of the drawn noise matches n2."""
    npix = mask.size
    n2 = np.full(npix, 0.5) * mask
    e1, e2 = make_noise(n2, mask, rng)
    inm = mask > 0
    assert np.all(e1[~inm] == 0.0)
    assert np.all(e2[~inm] == 0.0)
    # one realization over ~7000 masked pixels -> ~2% accuracy on the variance
    assert abs(np.mean(e1[inm] ** 2) / 0.5 - 1.0) < 0.05
    assert abs(np.mean(e2[inm] ** 2) / 0.5 - 1.0) < 0.05
