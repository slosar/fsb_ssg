"""NaMaster Gaussian covariance vs sample scatter, for the M0 (spin0xspin0) cross.

The 4 masks are equal-area disjoint footprints, so the covariance of one
sample (one realization on one footprint) is what the 400-sample scatter
estimates.  We compute the NaMaster Gaussian (Knox-like) covariance of the
C_l^{delta M0} bandpowers on a single footprint and compare its diagonal to the
sample scatter from run_masked.

Gaussian covariance of a cross C^{ab} (a=delta spin0, b=M0 spin0) needs the
underlying C^{aa}, C^{bb}, C^{ab}.  We estimate them on the spot from the
ensemble of masked maps (fsky-deconvolved pseudo-spectra), averaged over a few
realizations, which is sufficient for a Gaussian-covariance cross-check.

Output: data/nmt_cov_ns{NSIDE}.npz  and prints a diag(Gaussian)/diag(sample) ratio.

Run:  FSB_NSIDE=512 python nmt_covariance.py
"""
import os
import numpy as np
import healpy as hp
import pymaster as nmt
import simlib as L

NAVG = int(os.environ.get("FSB_COVAVG", "10"))   # realizations for auto-spectra
MASK_I = 0
SCHEMES = L.SCHEMES


def main():
    nside, lmax = L.NSIDE, L.LMAX
    bins = L.make_bins(lmax)
    nb = bins.get_n_bands()
    bands = list(map(tuple, np.load(
        os.path.join(L.DATA_OUT, f"masked_ns{nside}.npz"), allow_pickle=True)["bands"]))
    mask = L.load_masks(nside)[MASK_I]
    inm = mask > 0
    gt = np.load(os.path.join(L.DATA_OUT, f"ground_truth_ns{nside}.npz"))
    cl_ee, cl_bb = gt["cl_ee"], gt["cl_bb"]

    # mean noise variance for the fixed W2 (same recipe as run_masked)
    n2bar = np.zeros(L.NPIX)
    for r in range(1, NAVG + 1):
        _, _, _, ds = L.load_cached(r, nside)
        n2bar += L.SIGMA_E**2 / np.clip(L.N_BAR_SR * L.OMEGA_PIX * (1 + ds), 1e-3, None)
    n2bar = (n2bar / NAVG) * mask

    out = {}; out4 = {}
    for scheme in SCHEMES:
        for bi, (l0, l1) in enumerate(bands):
            c = L.band_signal_c(cl_ee, cl_bb, l0, l1)
            W2 = L.make_W2(n2bar, mask, scheme, l0, l1, c)

            # accumulate fsky-deconvolved auto/cross pseudo-spectra over NAVG sims
            cl_dd = np.zeros(lmax + 1); cl_mm = np.zeros(lmax + 1); cl_dm = np.zeros(lmax + 1)
            cl_EE = np.zeros(lmax + 1); cl_BB = np.zeros(lmax + 1); cl_EB = np.zeros(lmax + 1)
            cl_dE = np.zeros(lmax + 1); cl_dB = np.zeros(lmax + 1)
            fdd = np.mean(mask**2); fmm = np.mean(W2**2); fdm = np.mean(mask * W2)
            dlm = None; m0 = m4e = m4b = None
            for r in range(1, NAVG + 1):
                g1, g2, dl, ds = L.load_cached(r, nside)
                n2 = L.noise_variance(ds, mask)
                rng = np.random.default_rng(10_000 * r + MASK_I)
                e1, e2 = L.make_noise(n2, mask, rng)
                W1 = L.make_W1(n2, mask, scheme)
                aE, aB = L.shear_to_bandalm(g1 + e1, g2 + e2, W1, lmax)
                m0, m4e, m4b = L.band_fields(aE, aB, l0, l1, nside, lmax)
                m0 = m0 - L.noise_bias_template(n2, mask, scheme, l0, l1)
                dlm = (dl - np.sum(dl * mask) / np.sum(mask)) * mask
                a_d = hp.map2alm(mask * dlm, lmax=lmax)
                a_m = hp.map2alm(W2 * m0, lmax=lmax)
                aE4, aB4 = hp.map2alm_spin([W2 * m4e, W2 * m4b], 4, lmax=lmax)
                cl_dd += hp.alm2cl(a_d) / fdd / NAVG
                cl_mm += hp.alm2cl(a_m) / fmm / NAVG
                cl_dm += hp.alm2cl(a_d, a_m) / fdm / NAVG
                cl_EE += hp.alm2cl(aE4) / fmm / NAVG
                cl_BB += hp.alm2cl(aB4) / fmm / NAVG
                cl_EB += hp.alm2cl(aE4, aB4) / fmm / NAVG
                cl_dE += hp.alm2cl(a_d, aE4) / fdm / NAVG
                cl_dB += hp.alm2cl(a_d, aB4) / fdm / NAVG

            # M0: spin0 x spin0 Gaussian covariance
            f_d = nmt.NmtField(mask, [dlm], lmax=lmax, spin=0)
            f_m = nmt.NmtField(W2, [m0], lmax=lmax, spin=0)
            w = nmt.NmtWorkspace(); w.compute_coupling_matrix(f_d, f_m, bins)
            cw = nmt.NmtCovarianceWorkspace()
            cw.compute_coupling_coefficients(f_d, f_m, f_d, f_m)
            out[(scheme, bi)] = nmt.gaussian_covariance(
                cw, 0, 0, 0, 0, [cl_dd], [cl_dm], [cl_dm], [cl_mm], w, wb=w)

            # M4E: spin0 x spin4 Gaussian covariance (extract the E-cross block)
            f4 = nmt.NmtField(W2, [m4e, m4b], lmax=lmax, spin=4)
            w4 = nmt.NmtWorkspace(); w4.compute_coupling_matrix(f_d, f4, bins)
            cw4 = nmt.NmtCovarianceWorkspace()
            cw4.compute_coupling_coefficients(f_d, f4, f_d, f4)
            nb = bins.get_n_bands()
            cov4 = nmt.gaussian_covariance(
                cw4, 0, 4, 0, 4,
                [cl_dd], [cl_dE, cl_dB], [cl_dE, cl_dB],
                [cl_EE, cl_EB, cl_EB, cl_BB], w4, wb=w4)
            out4[(scheme, bi)] = cov4[:nb, :nb]      # E-cross (TE-like) block

    # compare to sample scatter
    mk = np.load(os.path.join(L.DATA_OUT, f"masked_ns{nside}.npz"), allow_pickle=True)
    ells = mk["ells"]
    sel = (ells >= 0) & (ells <= 768)
    lines = []
    res = {}
    for fname, gdict, totkey in [("M0", out, "tot_m0"), ("M4E", out4, "tot_m4e")]:
        for si, scheme in enumerate(SCHEMES):
            for bi, band in enumerate(bands):
                tot = mk[totkey][si, bi]               # [nsamp, nbin]
                samp_cov = np.cov(tot, rowvar=False)
                gcov = gdict[(scheme, bi)]
                dr = np.sqrt(np.diag(gcov)[sel]) / np.sqrt(np.clip(np.diag(samp_cov)[sel], 1e-300, None))
                res[f"{fname}_{scheme}_b{bi}_gauss"] = gcov
                res[f"{fname}_{scheme}_b{bi}_sample"] = samp_cov
                lines.append(f"{fname:3s} band{bi}{band} {scheme:5s}: "
                             f"median diag(Gauss)/diag(sample) std-ratio = {np.median(dr):.3f}")
    txt = "\n".join(lines)
    print(txt)
    np.savez(os.path.join(L.DATA_OUT, f"nmt_cov_ns{nside}.npz"),
             ells=ells, bands=np.array(bands), **res)
    with open(os.path.join(L.DATA_OUT, f"nmt_cov_ns{nside}.txt"), "w") as f:
        f.write(txt + "\n")
    plot(nside)


def plot(nside):
    """sqrt(diag) of the NaMaster Gaussian covariance vs the 400-realization
    sample scatter, for the M0 cross, per scheme and band (loads the saved npz)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGDIR = L.FIGDIR
    d = np.load(os.path.join(L.DATA_OUT, f"nmt_cov_ns{nside}.npz"))
    mk = np.load(os.path.join(L.DATA_OUT, f"masked_ns{nside}.npz"), allow_pickle=True)
    ells = d["ells"]; bands = [tuple(int(x) for x in b) for b in mk["bands"]]
    colors = {"none": "tab:red", "W2": "tab:orange", "W1W2": "tab:blue"}
    sel = (ells > 0) & (ells <= 768)
    fields = ["M0", "M4E"]
    fig, axes = plt.subplots(len(fields), len(bands),
                             figsize=(5 * len(bands), 3.8 * len(fields)), squeeze=False)
    for fi, fname in enumerate(fields):
        for bi, band in enumerate(bands):
            ax = axes[fi, bi]
            ymax = 0
            for s in SCHEMES:
                kg, ks = f"{fname}_{s}_b{bi}_gauss", f"{fname}_{s}_b{bi}_sample"
                if ks not in d.files:
                    continue
                samp = np.sqrt(np.clip(np.diag(d[ks]), 0, None))[sel]
                gauss = np.sqrt(np.clip(np.diag(d[kg]), 0, None))[sel]
                ax.plot(ells[sel], samp, "o", ms=4, color=colors[s], label=f"{s} (sims)")
                ax.plot(ells[sel], gauss, "-", lw=1.8, color=colors[s], alpha=0.8,
                        label=f"{s} (NaMaster)")
                ymax = max(ymax, samp.max())
            ax.set_yscale("log")
            if ymax > 0:
                ax.set_ylim(top=5 * ymax)             # clip rare Gaussian-cov spikes
            ax.set_xlim(0, 768)
            if fi == len(fields) - 1:
                ax.set_xlabel(r"$\ell$")
            ax.set_title(rf"$\sigma(\hat C_\ell^{{\delta {fname}}})$  band {band}", fontsize=10)
            if bi == 0:
                ax.set_ylabel(r"per-realization $\sigma$")
                ax.legend(fontsize=6, ncol=1)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"cov_compare_ns{nside}.pdf"))
    plt.close(fig)
    print(f"saved {FIGDIR}/cov_compare_ns{nside}.pdf")


if __name__ == "__main__":
    if os.environ.get("FSB_PLOTONLY"):
        plot(L.NSIDE)
    else:
        main()
