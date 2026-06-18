"""Covariance / chi2 / unbiasedness / SNR analysis and figures.

Loads the full-sky ground truth and the masked 400-sample run, then for each
scheme (none / W2 / W1W2), filter band, and field (M0, M4E):

  * transfer T_b = <signal-only masked>_b / truth_b   (method fidelity + W1 effect)
  * unbiasedness of the noise debiasing: per-sample d = tot - sig isolates the
    noise contribution to the cross (signal cancels), so <d>=0 is the clean test;
    chi2_noise = <d> Cov(d/sqrt N)^{-1} <d>.
  * unbiasedness vs truth in truth units: corrected = tot / T, diagonal chi2 of
    (<corrected> - truth) against the 400-sample error on the mean.
  * SNR(scheme) = sqrt( truth^T Cov_corrected^{-1} truth )  (full, on the band
    subset) and per-bin error bars, to compare the three weightings.

Figures (claude/latex/figures): M0 and M4E cross-spectra with the three schemes
overplotted on the truth, plus a transfer-function panel.

Run:  FSB_NSIDE=512 python analyze.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import simlib as L

FIGDIR = L.FIGDIR
os.makedirs(FIGDIR, exist_ok=True)

# cross-multipole range used for chi2/SNR (the delta x M cross lives at low l)
ELL_MIN, ELL_MAX = 0, 768


def load():
    nside = L.NSIDE
    gt = np.load(os.path.join(L.DATA_OUT, f"ground_truth_ns{nside}.npz"))
    mk = np.load(os.path.join(L.DATA_OUT, f"masked_ns{nside}.npz"), allow_pickle=True)
    return gt, mk


def cov_mean(x):
    """Sample covariance and covariance of the mean for x[nsamp, nbin]."""
    C = np.cov(x, rowvar=False)
    return C, C / x.shape[0]


def chi2_diag(resid, err):
    m = err > 0
    return float(np.sum((resid[m] / err[m]) ** 2)), int(m.sum())


def main():
    gt, mk = load()
    ells = gt["ells"]
    bands = [tuple(int(x) for x in b) for b in mk["bands"]]
    schemes = list(mk["schemes"])
    sel = (ells >= ELL_MIN) & (ells <= ELL_MAX)
    truth_m0 = gt["cl_m0"].mean(axis=1)     # [band, bin]
    truth_m4e = gt["cl_m4e"].mean(axis=1)

    fields = {"M0": ("sig_m0", "tot_m0", truth_m0),
              "M4E": ("sig_m4e", "tot_m4e", truth_m4e)}

    report = []
    results = {}
    for fname, (sk, tk, truth) in fields.items():
        SIG, TOT = mk[sk], mk[tk]            # [scheme, band, nsamp, nbin]
        for bi, band in enumerate(bands):
            tr = truth[bi]
            for si, scheme in enumerate(schemes):
                sig = SIG[si, bi]           # [nsamp, nbin]
                tot = TOT[si, bi]
                sig_mean = sig.mean(0)
                tot_mean = tot.mean(0)
                nsamp = sig.shape[0]
                # transfer (guard tiny truth)
                T = np.where(np.abs(tr) > 1e-30 * np.abs(tr).max(), sig_mean / tr, np.nan)
                # noise-debiasing test: d = tot - sig (signal cancels per sample)
                d = tot - sig
                dmean = d.mean(0)
                _, Cd_mean = cov_mean(d)
                err_d = np.sqrt(np.clip(np.diag(Cd_mean), 0, None))
                chi2n, dofn = chi2_diag(dmean[sel], err_d[sel])
                # corrected vs truth
                corr = tot / np.where(np.isfinite(T) & (T != 0), T, np.nan)
                cmean = np.nanmean(corr, 0)
                Cc, Cc_m = cov_mean(corr)
                err_cm = np.sqrt(np.clip(np.diag(Cc_m), 0, None))
                resid = cmean - tr
                chi2t, doft = chi2_diag(resid[sel], err_cm[sel])
                # SNR (full cov on subset) of detecting the truth shape
                Cc_s = Cc[np.ix_(sel, sel)]
                trs = tr[sel]
                try:
                    snr = float(np.sqrt(trs @ np.linalg.solve(Cc_s, trs)))
                except np.linalg.LinAlgError:
                    snr = np.nan
                err_tot = np.sqrt(np.clip(np.diag(np.cov(tot, rowvar=False)), 0, None))
                results[(fname, bi, scheme)] = dict(
                    ells=ells, truth=tr, tot_mean=tot_mean, sig_mean=sig_mean,
                    err_tot=err_tot, T=T, corr_mean=cmean, err_cm=err_cm, cov=Cc)
                report.append(
                    f"{fname:4s} band{bi}{band} {scheme:5s}: "
                    f"<T>={np.nanmedian(T[sel]):.3f}  "
                    f"chi2_noise={chi2n:.1f}/{dofn}  "
                    f"chi2_truth={chi2t:.1f}/{doft}  SNR={snr:.1f}")

    rep = "\n".join(report)
    print(rep)
    with open(os.path.join(L.DATA_OUT, f"analysis_ns{L.NSIDE}.txt"), "w") as f:
        f.write(rep + "\n")

    # ---- figures: cross-spectra per field/band, schemes overplotted, with a
    #      residual panel (mean - truth) / sigma_none below each ----
    colors = {"none": "tab:red", "W2": "tab:orange", "W1W2": "tab:blue"}
    for fname in fields:
        nbnd = len(bands)
        fig, axes = plt.subplots(2, nbnd, figsize=(6 * nbnd, 5.6), squeeze=False,
                                 sharex="col",
                                 gridspec_kw={"height_ratios": [3, 1]})
        for bi, band in enumerate(bands):
            ax, axr = axes[0, bi], axes[1, bi]
            tr = results[(fname, bi, schemes[0])]["truth"]
            sig_none = results[(fname, bi, "none")]["err_cm"]
            ax.plot(ells, tr, "k-", lw=2, label="truth (full sky)", zorder=5)
            for j, scheme in enumerate(schemes):
                R = results[(fname, bi, scheme)]
                off = (j - 1) * (ells[1] - ells[0]) * 0.18
                ax.errorbar(ells + off, R["corr_mean"], yerr=R["err_cm"],
                            fmt="o", ms=3, color=colors[scheme], alpha=0.8,
                            capsize=2, label=scheme)
                # residual in units of the 'none' error on the mean
                with np.errstate(divide="ignore", invalid="ignore"):
                    resid = (R["corr_mean"] - tr) / sig_none
                    rerr = R["err_cm"] / sig_none
                axr.errorbar(ells + off, resid, yerr=rerr, fmt="o", ms=3,
                             color=colors[scheme], alpha=0.8, capsize=2)
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlim(0, ELL_MAX)
            ax.set_title(rf"$C_\ell^{{\delta {fname}}}$  filter band {band}")
            if bi == 0:
                ax.set_ylabel("bandpower (truth units)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            for y in (-1, 1):
                axr.axhline(y, color="grey", lw=0.6, ls=":")
            axr.axhline(0, color="grey", lw=0.8)
            axr.set_ylim(-3.5, 3.5)
            axr.set_xlabel(r"$\ell$")
            if bi == 0:
                axr.set_ylabel(r"$(\hat C-{\rm truth})/\sigma_{\rm none}$")
            axr.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, f"cross_{fname}_ns{L.NSIDE}.pdf"))
        plt.close(fig)

    # ---- SNR / errorbar comparison figure ----
    fig, axes = plt.subplots(1, len(bands), figsize=(6 * len(bands), 4.0), squeeze=False)
    for bi, band in enumerate(bands):
        ax = axes[0, bi]
        for scheme in schemes:
            R = results[("M0", bi, scheme)]
            ax.plot(R["ells"], R["err_cm"], "-o", ms=3, color=colors[scheme], label=scheme)
        ax.set_xlim(0, ELL_MAX)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_title(rf"$M_0$ error on the mean, band {band}")
        if bi == 0:
            ax.set_ylabel(r"$\sigma(\hat C_\ell)/\sqrt{N}$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"snr_M0_ns{L.NSIDE}.pdf"))
    plt.close(fig)

    # ---- transfer function T_b for the three schemes (M0; the field with the
    #      noise bias and the clear W1 effect -- M4E transfer is noise-dominated) ----
    nbnd = len(bands)
    fig, axes = plt.subplots(1, nbnd, figsize=(5 * nbnd, 3.8), squeeze=False, sharey=True)
    for bi, band in enumerate(bands):
        ax = axes[0, bi]
        tr = results[("M0", bi, schemes[0])]["truth"]
        good = np.abs(tr) > 0.15 * np.abs(tr).max()   # signal-bearing bins (solid markers)
        for scheme in schemes:
            R = results[("M0", bi, scheme)]
            # full ell range (open, where the cross is consistent with zero -> T noisy)
            ax.plot(R["ells"], R["T"], "-", lw=0.8, color=colors[scheme], alpha=0.4)
            ax.plot(R["ells"][good], R["T"][good], "o", ms=4,
                    color=colors[scheme], label=scheme)
        ax.axhline(1.0, color="grey", lw=0.8, ls="--")
        ax.set_xlim(0, ELL_MAX)
        ax.set_ylim(0.0, 2.0)
        ax.set_title(rf"$T_\ell$  $M_0$  band {band}", fontsize=10)
        ax.set_xlabel(r"$\ell$")
        if bi == 0:
            ax.set_ylabel(r"$T_\ell=\langle \hat C^{\rm sig}_\ell\rangle/C^{\rm truth}_\ell$")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"transfer_ns{L.NSIDE}.pdf"))
    plt.close(fig)

    # save the measured transfer (both fields) for comparison with the analytic prediction
    Tsave = {"ells": ells, "bands": np.array(bands), "schemes": np.array(schemes)}
    for fname in fields:
        for bi in range(len(bands)):
            for scheme in schemes:
                Tsave[f"T_{fname}_b{bi}_{scheme}"] = results[(fname, bi, scheme)]["T"]
                Tsave[f"truth_{fname}_b{bi}"] = results[(fname, bi, schemes[0])]["truth"]
    np.savez(os.path.join(L.DATA_OUT, f"transfer_measured_ns{L.NSIDE}.npz"), **Tsave)

    # ---- correlation matrix of the 400-realization covariance (diagonal-dominant) ----
    cm_scheme = "W1W2"
    fig, axes = plt.subplots(len(fields), len(bands),
                             figsize=(4.0 * len(bands), 3.8 * len(fields)), squeeze=False)
    for fi, fname in enumerate(fields):
        for bi, band in enumerate(bands):
            ax = axes[fi, bi]
            C = results[(fname, bi, cm_scheme)]["cov"][np.ix_(sel, sel)]
            d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
            corr = C / np.outer(d, d)
            lsel = ells[sel]
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower",
                           extent=[lsel[0], lsel[-1], lsel[0], lsel[-1]])
            ax.set_title(rf"corr$(\hat C^{{\delta {fname}}})$  band {band}", fontsize=9)
            if fi == len(fields) - 1:
                ax.set_xlabel(r"$\ell$")
            if bi == 0:
                ax.set_ylabel(r"$\ell$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(rf"correlation matrices ({cm_scheme}); off-diagonals $\ll1$",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"cormat_ns{L.NSIDE}.pdf"))
    plt.close(fig)
    print(f"\nfigures -> {FIGDIR}")


if __name__ == "__main__":
    main()
