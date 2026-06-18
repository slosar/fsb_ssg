"""Full-sky, no-noise, no-mask, unit-weight ground-truth bispectrum.

For each of NREAL realizations: cross the lens density delta_lens with the
filtered-squared source shear M0 = |F[gamma]|^2 and the spin-4 E-mode
M4E = Re(F[gamma]^2), full sky.  The estimator is run through NaMaster with a
unit (all-sky) mask/weight so it is apples-to-apples with the masked run; the
mean over realizations is the ground truth, the scatter its (full-sky) error.

Output: data/ground_truth_ns{NSIDE}.npz with
  ells, bands, cl_m0[band, real, bin], cl_m4e[band, real, bin],
  cl_ee, cl_bb  (mean shear spectra, for the W2 signal density).

Run:  FSB_NSIDE=512 python ground_truth.py
"""
import os
import time
import numpy as np
import healpy as hp
import simlib as L

NREAL = int(os.environ.get("FSB_NREAL", "100"))


def main():
    t0 = time.time()
    nside, lmax = L.NSIDE, L.LMAX
    bins = L.make_bins(lmax)
    ells = bins.get_effective_ells()
    nb = bins.get_n_bands()
    bands = L.FILTER_BANDS
    one = np.ones(L.NPIX)

    # Full-sky workspaces (unit weight) reused for all realizations/bands.
    ws00 = L.workspace00(one, one, bins)
    ws04 = L.workspace04(one, one, bins)

    cl_m0 = np.zeros((len(bands), NREAL, nb))
    cl_m4e = np.zeros((len(bands), NREAL, nb))
    clee_acc = np.zeros(lmax + 1)
    clbb_acc = np.zeros(lmax + 1)

    for k in range(NREAL):
        r = k + 1
        g1, g2, dl, _ = L.load_cached(r, nside)
        dl = dl - dl.mean()                      # full-sky monopole removal
        f0d = L.field0(one, dl)
        aE, aB = L.map2alm_spin(g1, g2, lmax)    # W1 = 1
        clee_acc += hp.alm2cl(aE)
        clbb_acc += hp.alm2cl(aB)
        for bi, (l0, l1) in enumerate(bands):
            m0, m4e, m4b = L.band_fields(aE, aB, l0, l1, nside, lmax)
            cl_m0[bi, k] = L.cross_M0(f0d, m0, one, ws00)
            e, _b = L.cross_M4(f0d, m4e, m4b, one, ws04)
            cl_m4e[bi, k] = e
        if r % 10 == 0 or r == 1:
            print(f"  truth r{r:03d}/{NREAL}  ({time.time()-t0:.0f}s)", flush=True)

    out = os.path.join(L.DATA_OUT, f"ground_truth_ns{nside}.npz")
    np.savez(out, ells=ells, bands=np.array(bands), nside=nside, lmax=lmax,
             cl_m0=cl_m0, cl_m4e=cl_m4e,
             cl_ee=clee_acc / NREAL, cl_bb=clbb_acc / NREAL)
    print(f"saved {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
