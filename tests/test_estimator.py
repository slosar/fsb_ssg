import numpy as np
import healpy as hp
import pytest

pytest.importorskip("pymaster")

from fsb_ssg import (
    FSBConfig, FSBEstimator, noise_variance, make_noise, noise_bias_template,
)


def test_w2_normalisation_invariance(config, mask, shear):
    """Decoupled C_l^{delta M0} is invariant to the overall W2 normalisation.

    This is the central property that lets W2 be an arbitrary optimal weight:
    NaMaster's mode-decoupling removes it from the mean.
    """
    est = FSBEstimator(config)
    g1, g2 = shear
    l_lo, l_hi = config.filter_bands[0]
    m0, _, _ = est.filtered_square(g1, g2, mask, l_lo, l_hi)

    # a non-trivial positive weight inside the mask
    rng = np.random.default_rng(7)
    W2 = mask * (1.0 + 0.5 * np.abs(rng.normal(size=mask.size)))
    delta = est.field0(mask, (m0 - m0[mask > 0].mean()) * mask)  # any spin-0 probe

    ws00_a, _ = est.workspaces(mask, W2)
    ws00_b, _ = est.workspaces(mask, 13.0 * W2)
    cl_a = est.cross_m0(delta, m0, W2, ws00_a)
    cl_b = est.cross_m0(delta, m0, 13.0 * W2, ws00_b)
    np.testing.assert_allclose(cl_a, cl_b, rtol=1e-8, atol=1e-12)


def test_noise_bias_template_matches_montecarlo():
    """The |K|^2 template reproduces the Monte-Carlo mean of M0 from pure noise."""
    cfg = FSBConfig(nside=64, dell=16, filter_bands=((10, 40),))
    est = FSBEstimator(cfg)
    npix = cfg.npix
    mask = np.ones(npix)
    n2 = np.full(npix, 0.4)
    l_lo, l_hi = cfg.filter_bands[0]

    template = noise_bias_template(n2, mask, "none", l_lo, l_hi)

    rng = np.random.default_rng(99)
    nrec = 60
    acc = np.zeros(npix)
    for _ in range(nrec):
        e1, e2 = make_noise(n2, mask, rng)
        m0, _, _ = est.filtered_square(e1, e2, mask, l_lo, l_hi)
        acc += m0
    mc_mean = acc / nrec
    # both are (nearly) constant full-sky maps; compare their sky means
    assert np.isclose(mc_mean.mean(), template.mean(), rtol=0.03)


def test_full_measurement_smoke(config, mask, shear, delta_source):
    """End-to-end single-realization measurement returns finite, right-shaped C_l."""
    est = FSBEstimator(config)
    g1, g2 = shear
    l_lo, l_hi = config.filter_bands[0]

    n2 = noise_variance(delta_source, mask, config)
    W1 = est.make_W1(n2, mask, "W1W2")
    W2 = est.make_W2(n2, mask, "W1W2", l_lo, l_hi, 1.0)
    ws00, ws04 = est.workspaces(mask, W2)

    dl = delta_source.copy()
    dlm = (dl - np.sum(dl * mask) / np.sum(mask)) * mask
    f0d = est.field0(mask, dlm)

    m0, m4e, m4b = est.filtered_square(g1, g2, W1, l_lo, l_hi)
    bias = est.noise_bias_template(n2, mask, "W1W2", l_lo, l_hi)
    cl_m0 = est.cross_m0(f0d, m0 - bias, W2, ws00)
    cl_e, cl_b = est.cross_m4(f0d, m4e, m4b, W2, ws04)

    nb = est.n_bands
    for arr in (cl_m0, cl_e, cl_b):
        assert arr.shape == (nb,)
        assert np.all(np.isfinite(arr))
    assert est.ells.shape == (nb,)
