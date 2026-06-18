"""Simulation adapter binding :mod:`fsb_ssg` to the Takahashi-sim validation.

The library (:mod:`fsb_ssg`) is generic; this module pins it to the specific
configuration and data of the science validation:

  * an :class:`fsb_ssg.FSBConfig` driven by ``FSB_NSIDE`` (bands chosen so their
    upper edge fits below ``lmax = 2*nside``);
  * the survey-specific map IO (source shear, lens / source density, the four
    disjoint DES Y6 footprints), read from ``FSB_SIM_DIR``;
  * thin, config-bound wrappers (``noise_variance``, ``band_signal_c``,
    ``make_bins``, ``field0``, ``cross_M0``, ``cross_M4``) so the pipeline
    scripts can call them without repeating ``config`` / ``lmax`` / ``dell``.

The pipeline scripts ``import simlib as L`` and otherwise use the library
unchanged, so the physics lives in one place.

Environment:
  FSB_NSIDE     working resolution (default 512)
  FSB_SIM_DIR   directory holding the simulation maps (required for real runs)
"""

import os

import numpy as np
import healpy as hp

import fsb_ssg as F
from fsb_ssg import (
    FSBConfig, SCHEMES, FIDUCIAL, bands_for_nside,
    map2alm_spin, alm2map_spin, bandpass, shear_to_bandalm, band_fields,
    kernel2_conv, response_beam,
    make_noise, make_W1, make_W2, noise_bias_template,
    workspace00, workspace04,
)

# --------------------------------------------------------------------------- #
# Configuration (env-driven)
# --------------------------------------------------------------------------- #
NSIDE = int(os.environ.get("FSB_NSIDE", "512"))
FILTER_BANDS = list(bands_for_nside(NSIDE))
DELL = 48

CONFIG = FSBConfig(nside=NSIDE, dell=DELL, filter_bands=tuple(FILTER_BANDS))

LMAX = CONFIG.lmax
NPIX = CONFIG.npix
OMEGA_PIX = CONFIG.omega_pix
SIGMA_E = CONFIG.sigma_e
N_BAR_SR = CONFIG.n_bar_sr

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_OUT = os.path.join(_HERE, "data")
FIGDIR = os.path.abspath(os.path.join(_HERE, "..", "report", "figures"))
SIM_DIR = os.environ.get(
    "FSB_SIM_DIR", os.path.abspath(os.path.join(_HERE, "..", "..", "simulated_data")))

os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# Redshift tags of the simulation maps.
Z_SHEAR = "1.0334"
Z_LENS = "0.5406"
Z_SRC = "1.0"


# --------------------------------------------------------------------------- #
# Config-bound wrappers (so scripts need not pass config/lmax/dell)
# --------------------------------------------------------------------------- #
def noise_variance(delta_source, mask):
    return F.noise_variance(delta_source, mask, CONFIG)


def band_signal_c(cl_ee, cl_bb, l_lo, l_hi):
    return F.band_signal_c(cl_ee, cl_bb, l_lo, l_hi, CONFIG)


def make_bins(lmax=LMAX):
    return F.make_bins(lmax, DELL)


def field0(mask, m, lmax=LMAX):
    return F.field0(mask, m, lmax)


def cross_M0(f0_delta, m0_map, weight_M, ws00, lmax=LMAX):
    return F.cross_M0(f0_delta, m0_map, weight_M, ws00, lmax)


def cross_M4(f0_delta, m4e, m4b, weight_M, ws04, lmax=LMAX):
    return F.cross_M4(f0_delta, m4e, m4b, weight_M, ws04, lmax)


# --------------------------------------------------------------------------- #
# Survey-specific map IO
# --------------------------------------------------------------------------- #
def _read_degrade(path, nside, spin0=True):
    """Read a scalar map and harmonically degrade to ``nside``."""
    m = hp.read_map(path)
    if hp.get_nside(m) == nside:
        return m
    lmax = 3 * nside - 1
    if spin0:
        return hp.alm2map(hp.map2alm(m, lmax=lmax), nside=nside)
    raise RuntimeError("use load_shear for spin-2 degrade")


def load_shear(ridx, nside):
    """Source shear ``(g1, g2)``, spin-2 harmonically degraded to ``nside``."""
    g1 = hp.read_map(os.path.join(SIM_DIR, f"map_g1_z{Z_SHEAR}_r{ridx:03d}_nside1024.fits.gz"))
    g2 = hp.read_map(os.path.join(SIM_DIR, f"map_g2_z{Z_SHEAR}_r{ridx:03d}_nside1024.fits.gz"))
    if hp.get_nside(g1) != nside:
        lmax = 3 * nside - 1
        aE, aB = map2alm_spin(g1, g2, lmax)
        g1, g2 = alm2map_spin(aE, aB, nside)
    return g1, g2


def load_delta_lens(ridx, nside):
    return _read_degrade(
        os.path.join(SIM_DIR, f"map_delta_z{Z_LENS}_r{ridx:03d}_nside1024.fits.gz"), nside)


def load_delta_source(ridx, nside):
    return _read_degrade(
        os.path.join(SIM_DIR, f"map_delta_source_z{Z_SRC}_r{ridx:03d}_nside1024.fits.gz"), nside)


def load_cached(ridx, nside=NSIDE):
    """Return ``(g1, g2, delta_lens, delta_src)`` from the degraded cache if
    present (see ``build_cache.py``), else degrade on the fly."""
    path = os.path.join(DATA_OUT, f"cache_ns{nside}", f"r{ridx:03d}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return (d["g1"].astype(np.float64), d["g2"].astype(np.float64),
                d["delta_lens"].astype(np.float64), d["delta_src"].astype(np.float64))
    g1, g2 = load_shear(ridx, nside)
    return g1, g2, load_delta_lens(ridx, nside), load_delta_source(ridx, nside)


def load_masks(nside):
    """Four disjoint DES Y6 footprints, binarised at the working ``nside``."""
    masks = []
    for i in range(1, 5):
        m = hp.read_map(os.path.join(SIM_DIR, f"desy6_map{i}_ns1024.fits.gz"))
        if hp.get_nside(m) != nside:
            m = hp.alm2map(hp.map2alm(m, lmax=3 * nside - 1), nside=nside)
        masks.append((m > 0.5).astype(float))
    return masks
