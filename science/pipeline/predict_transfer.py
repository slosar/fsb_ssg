"""Analytic transfer function (three methods) + the fiducial (B)-corrected
cross-spectra.

Squeezed-limit response (appendix): the delta-correlated part of the
filtered-squared field is the linear response  kappa * (W1^2 delta_lens), with
the filtered-shear kernel b^kappa_l = fsb_lib.response_beam (shear contracted
analytically into kappa(C^gg)).  delta_src is an externally fixed survey window
(it sets the noise and the weights W1, W2); only delta_lens is random, drawn from
C^{d_lens}, and is taken uncorrelated with the window.

Transfer methods (all from spectra + geometry, never the lens map):
  (A) MEASURED   : T_A = <C^sig>/C^truth from the 400-realization run (loaded).
  (B) GAUSSIAN MC: fixed delta_src -> fixed W1,W2; draw Gaussian delta_lens;
                   push kappa*(W1^2 delta_lens) through NaMaster; average.   FIDUCIAL.
  (C) CLOSED FORM: T_C = g_edge * N_weight  (no Monte-Carlo), with
                   g_edge   = <kappa*mask>_mask           (mask-edge loss),
                   N_weight = <W2 mask W1^2>/<W2 mask>     (inverse-variance Jensen).

Outputs:
  data/transfer_pred_ns{NSIDE}.npz        (T_B, T_C)
  figures/transfer_pred_ns{NSIDE}.pdf     (A/B/C comparison, full ell range)
  figures/cross_{M0,M4E}_pred{B,C}_ns{NSIDE}.pdf  (cross-spectra corrected by T_B, T_C)

Run:  FSB_NSIDE=1024 python predict_transfer.py
      FSB_NSIDE=1024 FSB_PLOTONLY=1 python predict_transfer.py   # figures from saved npz
"""
import os
import warnings
import numpy as np
import healpy as hp
warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN high-l T_B bins
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simlib as L

NMC = int(os.environ.get("FSB_NMC", "1000"))       # Gaussian delta_lens draws
SRC_FIX = int(os.environ.get("FSB_SRCFIX", "1"))  # realization used as the fixed window
FIGDIR = L.FIGDIR
ELL_MAX = 768
COLORS = {"none": "tab:red", "W2": "tab:orange", "W1W2": "tab:blue"}


def measure_cl_lens(nside, lmax, nr=30):
    acc = np.zeros(lmax + 1)
    for r in range(1, nr + 1):
        _, _, dl, _ = L.load_cached(r, nside)
        acc += hp.anafast(dl - dl.mean(), lmax=lmax) / nr
    return acc


def compute():
    """Compute T_B (Gaussian-MC) and T_C (closed form); save transfer_pred npz."""
    nside, lmax = L.NSIDE, L.LMAX
    bins = L.make_bins(lmax)
    ells = bins.get_effective_ells()
    nb = bins.get_n_bands()
    bands = L.FILTER_BANDS
    masks = L.load_masks(nside)
    one = np.ones(L.NPIX)

    gt = np.load(os.path.join(L.DATA_OUT, f"ground_truth_ns{nside}.npz"))
    cgamma = 0.5 * (gt["cl_ee"] + gt["cl_bb"])
    cl_lens = measure_cl_lens(nside, lmax)
    bkappa = {bi: L.response_beam(l0, l1, cgamma, lmax) for bi, (l0, l1) in enumerate(bands)}

    # fixed delta_src window -> deterministic survey weights W1, W2
    _, _, _, ds_fix = L.load_cached(SRC_FIX, nside)
    W1map, W2map, ws00 = {}, {}, {}
    for mi, mask in enumerate(masks):
        n2 = L.noise_variance(ds_fix, mask)
        for s in L.SCHEMES:
            W1map[(mi, s)] = L.make_W1(n2, mask, s)
            for bi, (l0, l1) in enumerate(bands):
                c = L.band_signal_c(gt["cl_ee"], gt["cl_bb"], l0, l1)
                W2map[(mi, s, bi)] = L.make_W2(n2, mask, s, l0, l1, c)
                ws00[(mi, s, bi)] = L.workspace00(mask, W2map[(mi, s, bi)], bins)
    ws_unit = L.workspace00(one, one, bins)
    print("workspaces built", flush=True)

    # (C) closed form: T_C = g_edge * N_weight
    T_C = np.zeros((len(L.SCHEMES), len(bands), nb))
    for si, s in enumerate(L.SCHEMES):
        for bi in range(len(bands)):
            vals = []
            for mi, mask in enumerate(masks):
                inm = mask > 0
                ge = hp.smoothing(mask, beam_window=bkappa[bi])
                g_edge = ge[inm].mean() / mask[inm].mean()
                W2m = W2map[(mi, s, bi)] * mask
                Nw = np.sum(W2m * W1map[(mi, s)] ** 2) / np.sum(W2m)  # W1_none^2 = mask -> /1
                vals.append(g_edge * Nw)
            T_C[si, bi] = np.mean(vals)

    # (B) Gaussian-MC over delta_lens (fixed window)
    meas = np.zeros((len(L.SCHEMES), len(bands), nb)); truth = np.zeros((len(bands), nb))
    rng = np.random.default_rng(11)
    for k in range(NMC):
        np.random.seed(rng.integers(0, 2**31 - 1))
        dl = hp.synfast(cl_lens, nside)
        f0d_full = L.field0(one, dl)
        for bi in range(len(bands)):
            truth[bi] += L.cross_M0(f0d_full, hp.smoothing(dl, beam_window=bkappa[bi]), one, ws_unit)
        for mi, mask in enumerate(masks):
            dlm = (dl - np.sum(dl * mask) / np.sum(mask)) * mask
            f0d = L.field0(mask, dlm)
            a_u = {bi: hp.smoothing(mask * dl, beam_window=bkappa[bi]) for bi in range(len(bands))}
            a_i = {bi: hp.smoothing(W1map[(mi, "W1W2")] ** 2 * dl, beam_window=bkappa[bi])
                   for bi in range(len(bands))}
            for si, s in enumerate(L.SCHEMES):
                for bi in range(len(bands)):
                    a = a_i[bi] if s == "W1W2" else a_u[bi]
                    meas[si, bi] += L.cross_M0(f0d, a, W2map[(mi, s, bi)], ws00[(mi, s, bi)])
        if k % 10 == 0:
            print(f"  MC {k}/{NMC}", flush=True)
    meas /= (NMC * len(masks)); truth /= NMC
    T_B = np.where(np.abs(truth[None]) > 0, meas / truth[None], np.nan)

    np.savez(os.path.join(L.DATA_OUT, f"transfer_pred_ns{nside}.npz"),
             ells=ells, bands=np.array(bands), schemes=np.array(L.SCHEMES), T_B=T_B, T_C=T_C)
    print(f"saved transfer_pred_ns{nside}.npz", flush=True)


def plot():
    """Make the transfer-comparison figure and the (B)-corrected cross-spectra,
    all from saved data products (cheap; no NaMaster)."""
    nside = L.NSIDE
    pr = np.load(os.path.join(L.DATA_OUT, f"transfer_pred_ns{nside}.npz"))
    meas = np.load(os.path.join(L.DATA_OUT, f"transfer_measured_ns{nside}.npz"))
    gt = np.load(os.path.join(L.DATA_OUT, f"ground_truth_ns{nside}.npz"))
    mk = np.load(os.path.join(L.DATA_OUT, f"masked_ns{nside}.npz"), allow_pickle=True)
    ells = pr["ells"]; bands = [tuple(int(x) for x in b) for b in pr["bands"]]
    T_B, T_C = pr["T_B"], pr["T_C"]

    # --- transfer A/B/C comparison (full ell range) ---
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4.2),
                             squeeze=False, sharey=True)
    for bi, band in enumerate(bands):
        ax = axes[0, bi]
        tr = meas[f"truth_M0_b{bi}"]; good = np.abs(tr) > 0.15 * np.abs(tr).max()
        for si, s in enumerate(L.SCHEMES):
            c = COLORS[s]
            ax.plot(ells, T_B[si, bi], "-", lw=2.0, color=c, alpha=0.9, label=f"{s}: (B) Gauss-MC")
            ax.plot(ells, T_C[si, bi], "--", lw=1.6, color=c, alpha=0.9, label=f"{s}: (C) closed")
            ax.plot(ells[good], meas[f"T_M0_b{bi}_{s}"][good], "o", ms=5, color=c,
                    label=f"{s}: (A) measured")
            ax.plot(ells[good], meas[f"T_M4E_b{bi}_{s}"][good], "x", ms=5, color=c, mew=1.5)
        ax.axhline(1.0, color="grey", lw=0.8, ls=":")
        ax.set_xlim(0, ELL_MAX); ax.set_ylim(0.0, 1.6)
        ax.set_title(rf"transfer  band {band}", fontsize=10); ax.set_xlabel(r"$\ell$")
        if bi == 0:
            ax.set_ylabel(r"$T_\ell$"); ax.legend(fontsize=5.5, ncol=1, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"transfer_pred_ns{nside}.pdf")); plt.close(fig)

    # --- cross-spectra corrected by an analytic transfer (B Gauss-MC or C closed) ---
    truth = {"M0": gt["cl_m0"].mean(1), "M4E": gt["cl_m4e"].mean(1)}
    tot = {"M0": mk["tot_m0"], "M4E": mk["tot_m4e"]}

    def plot_cross(T, tag):
        """Cross-spectra corrected by transfer T (shape [scheme, band, nb]),
        saved as cross_{M0,M4E}_pred{tag}_ns{nside}.pdf."""
        for fname in ["M0", "M4E"]:
            fig, axes = plt.subplots(2, len(bands), figsize=(6 * len(bands), 5.6), squeeze=False,
                                     sharex="col", gridspec_kw={"height_ratios": [3, 1]})
            for bi, band in enumerate(bands):
                ax, axr = axes[0, bi], axes[1, bi]
                tr = truth[fname][bi]
                corr, errm = {}, {}
                for si, s in enumerate(L.SCHEMES):
                    Tb = T[si, bi]; good = np.isfinite(Tb) & (Tb > 0.3)
                    cv = tot[fname][si, bi] / np.where(good, Tb, np.nan)
                    corr[s] = np.nanmean(cv, 0)
                    errm[s] = np.sqrt(np.clip(np.diag(np.cov(np.nan_to_num(cv), rowvar=False)), 0, None)
                                      / cv.shape[0])
                sig0 = errm["none"]
                ax.plot(ells, tr, "k-", lw=2, label="truth (full sky)", zorder=5)
                for si, s in enumerate(L.SCHEMES):
                    off = (si - 1) * (ells[1] - ells[0]) * 0.18
                    ax.errorbar(ells + off, corr[s], yerr=errm[s], fmt="o", ms=3, color=COLORS[s],
                                alpha=0.8, capsize=2, label=s + (" (fiducial)" if s == L.FIDUCIAL else ""))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        axr.errorbar(ells + off, (corr[s] - tr) / sig0, yerr=errm[s] / sig0,
                                     fmt="o", ms=3, color=COLORS[s], alpha=0.8, capsize=2)
                ax.axhline(0, color="grey", lw=0.6); ax.set_xlim(0, ELL_MAX)
                ax.set_title(rf"$C_\ell^{{\delta {fname}}}$ (corrected by $T^{{\rm ({tag})}}$)  band {band}")
                if bi == 0:
                    ax.set_ylabel("bandpower (truth units)")
                ax.legend(fontsize=8); ax.grid(alpha=0.3)
                for y in (-1, 1):
                    axr.axhline(y, color="grey", lw=0.6, ls=":")
                axr.axhline(0, color="grey", lw=0.8); axr.set_ylim(-3.5, 3.5); axr.set_xlabel(r"$\ell$")
                if bi == 0:
                    axr.set_ylabel(r"$(\hat C-{\rm truth})/\sigma_{\rm none}$")
                axr.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(FIGDIR, f"cross_{fname}_pred{tag}_ns{nside}.pdf")); plt.close(fig)

    plot_cross(T_B, "B")
    plot_cross(T_C, "C")
    print(f"figures -> {FIGDIR}")


if __name__ == "__main__":
    if not os.environ.get("FSB_PLOTONLY"):
        compute()
    plot()
