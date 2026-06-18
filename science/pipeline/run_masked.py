"""Masked + variable-noise FSB measurement over 100 sims x 4 disjoint masks = 400.

Three weighting schemes: 'none' (binary mask), 'W2' (post-square IVW),
'W1W2' (W1=1/n^2 pre-filter + W2 post-square).  For each sample we store the
cross-spectra of delta_lens with M0 (bl2beam-debiased) and with M4E, for both
the signal-only channel (-> transfer function) and the signal+noise channel
(-> data vector + sample covariance).

W2 (the NaMaster weight) is fixed per (mask, scheme, band), built from the
ensemble-mean noise variance, so coupling matrices are computed once and reused.
W1 and the bl2beam bias template use the actual per-realization noise.

Output: data/masked_ns{NSIDE}.npz
  schemes, bands, ells,
  sig_m0[scheme,band,400,nbin], tot_m0[...], sig_m4e[...], tot_m4e[...]

Run:  FSB_NSIDE=512 python run_masked.py
      FSB_NREAL=5 FSB_NMASK=1 FSB_BANDS=1 python run_masked.py   # quick test
"""
import os
import time
import numpy as np
import simlib as L

NREAL = int(os.environ.get("FSB_NREAL", "100"))
# Mask subset (for parallel runs): comma-separated indices into the 4 masks.
MASK_IDS = [int(x) for x in os.environ.get("FSB_MASKS", "0,1,2,3").split(",")]
NBANDS = int(os.environ.get("FSB_BANDS", str(len(L.FILTER_BANDS))))
SCHEMES = L.SCHEMES
W1GROUPS = {"unit": ["none", "W2"], "ivar": ["W1W2"]}   # group by shared W1


def ensemble_n2(nside):
    """Mask-independent mean noise variance density (mean_r of sigma^2/(1+d_src))."""
    acc = np.zeros(L.NPIX)
    for r in range(1, NREAL + 1):
        _, _, _, ds = L.load_cached(r, nside)
        acc += L.SIGMA_E**2 / np.clip(L.N_BAR_SR * L.OMEGA_PIX * (1.0 + ds), 1e-3, None)
    return acc / NREAL


def main():
    t0 = time.time()
    nside, lmax = L.NSIDE, L.LMAX
    bins = L.make_bins(lmax)
    ells = bins.get_effective_ells()
    nb = bins.get_n_bands()
    bands = L.FILTER_BANDS[:NBANDS]
    all_masks = L.load_masks(nside)
    masks = [all_masks[i] for i in MASK_IDS]
    nm = len(masks)
    print(f"masks={MASK_IDS} bands={bands} nreal={NREAL}", flush=True)

    # band signal density c from mean shear spectra (for W2 shape only).
    gtf = os.path.join(L.DATA_OUT, f"ground_truth_ns{nside}.npz")
    if os.path.exists(gtf):
        gt = np.load(gtf)
        cl_ee, cl_bb = gt["cl_ee"], gt["cl_bb"]
    else:
        g1, g2, _, _ = L.load_cached(1, nside)   # fallback: single realization
        aE, aB = L.map2alm_spin(g1, g2, lmax)
        cl_ee, cl_bb = L.hp.alm2cl(aE), L.hp.alm2cl(aB)

    n2bar_glob = ensemble_n2(nside)
    print(f"ensemble n2 built ({time.time()-t0:.0f}s)", flush=True)

    # Precompute fixed W2 maps and NaMaster workspaces per (mask, scheme, band).
    W2map, ws00, ws04 = {}, {}, {}
    for mi, mask in enumerate(masks):
        n2bar = n2bar_glob * mask
        for scheme in SCHEMES:
            for bi, (l0, l1) in enumerate(bands):
                c = L.band_signal_c(cl_ee, cl_bb, l0, l1)
                W2 = L.make_W2(n2bar, mask, scheme, l0, l1, c)
                key = (mi, scheme, bi)
                W2map[key] = W2
                ws00[key] = L.workspace00(mask, W2, bins)
                ws04[key] = L.workspace04(mask, W2, bins)
    print(f"workspaces built: {len(ws00)} ({time.time()-t0:.0f}s)", flush=True)

    nsamp = NREAL * nm
    shape = (len(SCHEMES), len(bands), nsamp, nb)
    sig_m0 = np.zeros(shape); tot_m0 = np.zeros(shape)
    sig_m4e = np.zeros(shape); tot_m4e = np.zeros(shape)
    sidx = {s: i for i, s in enumerate(SCHEMES)}

    for k in range(NREAL):
        r = k + 1
        g1, g2, dl, ds = L.load_cached(r, nside)
        for mi, mask in enumerate(masks):
            gmi = MASK_IDS[mi]                       # global mask index (seeding)
            samp = k * nm + mi
            n2 = L.noise_variance(ds, mask)
            dlm = (dl - np.sum(dl * mask) / np.sum(mask)) * mask
            f0d = L.field0(mask, dlm)
            rng = np.random.default_rng(10_000 * r + gmi)
            e1, e2 = L.make_noise(n2, mask, rng)

            for grp, gschemes in W1GROUPS.items():
                W1 = L.make_W1(n2, mask, gschemes[0])
                aEs, aBs = L.shear_to_bandalm(g1, g2, W1, lmax)
                aEt, aBt = L.shear_to_bandalm(g1 + e1, g2 + e2, W1, lmax)
                for bi, (l0, l1) in enumerate(bands):
                    m0s, m4es, m4bs = L.band_fields(aEs, aBs, l0, l1, nside, lmax)
                    m0t, m4et, m4bt = L.band_fields(aEt, aBt, l0, l1, nside, lmax)
                    B = L.noise_bias_template(n2, mask, gschemes[0], l0, l1)
                    for scheme in gschemes:
                        key = (mi, scheme, bi)
                        W2 = W2map[key]
                        si = sidx[scheme]
                        sig_m0[si, bi, samp] = L.cross_M0(f0d, m0s, W2, ws00[key])
                        tot_m0[si, bi, samp] = L.cross_M0(f0d, m0t - B, W2, ws00[key])
                        e, _ = L.cross_M4(f0d, m4es, m4bs, W2, ws04[key])
                        sig_m4e[si, bi, samp] = e
                        e, _ = L.cross_M4(f0d, m4et, m4bt, W2, ws04[key])
                        tot_m4e[si, bi, samp] = e
        if r % 5 == 0 or r == 1:
            print(f"  r{r:03d}/{NREAL}  ({time.time()-t0:.0f}s)", flush=True)

    tag = "".join(str(i) for i in MASK_IDS)
    suffix = "" if MASK_IDS == [0, 1, 2, 3] else f"_m{tag}"
    out = os.path.join(L.DATA_OUT, f"masked_ns{nside}{suffix}.npz")
    np.savez(out, schemes=np.array(SCHEMES), bands=np.array(bands), ells=ells,
             nside=nside, lmax=lmax, nreal=NREAL, mask_ids=np.array(MASK_IDS),
             sig_m0=sig_m0, tot_m0=tot_m0, sig_m4e=sig_m4e, tot_m4e=tot_m4e)
    print(f"saved {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
