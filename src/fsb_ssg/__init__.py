r"""fsb_ssg -- Filtered-Square Bispectrum estimator for shear x shear x galaxy.

A NaMaster-based estimator for the shear--shear--density bispectrum, measured by
filtering and squaring the shear field and cross-correlating the squared field
with the lens density.  The package handles a survey mask, spatially
inhomogeneous shape noise, and optimal pre- and post-square weighting.

Pipeline (per realization)::

    gamma --W1--> W1 gamma --F(band)--> xi --square--> M0, M4
                                                    |
                                       (W2 weight, via NaMaster)
                                                    v
                            cross-correlate with lens delta -> C_l^{delta M}

Weighting schemes (``SCHEMES``):

    'none'  : W1 = 1,     W2 = 1        (binary mask only)
    'W2'    : W1 = 1,     W2 = optimal  (post-square inverse-variance)
    'W1W2'  : W1 = 1/n^2, W2 = optimal  (full optimal weighting)   <-- FIDUCIAL

The squared fields are ``M0 = |xi|^2`` (spin 0) and ``M4 = xi^2`` (spin 4,
E/B).  NaMaster mode-decouples the ``W2`` field weight, so the decoupled
bandpowers are unbiased for any ``W2``; ``W2`` only changes the variance.

See ``science/`` in the source tree for the full simulation-based validation
and the accompanying report.
"""

from .config import (
    FSBConfig,
    SCHEMES,
    FIDUCIAL,
    DEFAULT_FILTER_BANDS,
    bands_for_nside,
)
from .transforms import (
    map2alm_spin,
    alm2map_spin,
    bandpass,
    shear_to_bandalm,
    band_fields,
)
from .kernels import kernel2_conv, response_beam
from .noise import (
    noise_variance,
    make_noise,
    make_W1,
    make_W2,
    noise_bias_template,
    band_signal_c,
)
from .estimator import (
    FSBEstimator,
    make_bins,
    field0,
    workspace00,
    workspace04,
    cross_M0,
    cross_M4,
)

__version__ = "0.1.0"

__all__ = [
    "FSBConfig",
    "SCHEMES",
    "FIDUCIAL",
    "DEFAULT_FILTER_BANDS",
    "bands_for_nside",
    "map2alm_spin",
    "alm2map_spin",
    "bandpass",
    "shear_to_bandalm",
    "band_fields",
    "kernel2_conv",
    "response_beam",
    "noise_variance",
    "make_noise",
    "make_W1",
    "make_W2",
    "noise_bias_template",
    "band_signal_c",
    "FSBEstimator",
    "make_bins",
    "field0",
    "workspace00",
    "workspace04",
    "cross_M0",
    "cross_M4",
    "__version__",
]
