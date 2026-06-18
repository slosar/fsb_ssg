"""Method (D): mode-coupling-matrix calibration of the W1 modulation.

The delta-correlated part of the W1-weighted squared field is the response
kappa*(W1^2 delta) (eq. kappa of the note).  If the weight were smooth on the
kernel scale this would equal W1^2 (kappa*delta): the full-sky truth response
seen through the effective window  w_D = W2 m W1^2.  Decoupling the measured
pseudo-C_l [ m delta , (W2 m) M0 ] with the NaMaster mode-coupling matrix
built for the window pair (m, w_D) -- a map-less "dummy" field carrying w_D --
then returns truth-calibrated bandpowers up to the scalar edge factor g_edge:
the inverse-variance Jensen factor N_weight of method (C) is absorbed into the
coupling matrix as a full ell-coupling rather than a scalar.

Doing the commutator properly, the pseudo-spectrum probes the weight at the
*inner* point of the kernel convolution, so the correct multiplicative window
carries the kernel-smoothed weight,

    w_D^smooth = W2 m (kappa * W1^2)        [variant D2]

rather than the raw  w_D^raw = W2 m W1^2    [variant D1, the literal proposal].

D1 over-counts the small-scale weight structure that the kernel filters out of
the band; D2 keeps exactly the part of the modulation the estimator responds
to.  Because kappa*W1^2 also dips where the kernel support leaves the mask,
the D2 window contains the edge loss as well, so its expected residual
calibration is ~1 (no transfer function at all), while D1's idealized residual
is g_edge.  Both are computed; the run reports the measured residuals.

Decoupling is linear in the pseudo-spectrum, so the saved standard-decoupled
bandpowers are mapped exactly without re-measuring:

    Chat_D = M_{D,b}^{-1} M_{std,b} Chat_std = A Chat_std,

with A built by pushing unit bandpower vectors through
unbin -> couple(ws_std) -> decouple(ws_D).  ws_std reproduces the run's
workspaces exactly (W2 from the ensemble-mean noise); the dummy windows use
W1 of the fixed SRC_FIX survey window, the same fixed-window treatment as
methods (B)/(C) (the ensemble-mean W1 is nearly flat and must not be used).
The spin-4 transform uses the EE block of A (the decoupled B was not stored);
the E<-B leakage block vanishes identically, and the whole transform is
spot-checked against direct D-decoupling of one realization.

Outputs:
  data/method_d_ns{NSIDE}.npz          (D1/D2 bandpowers, A matrices, checks)
  data/method_d_ns{NSIDE}.txt          (residual-transfer / chi2 / SNR report)
  figures/cross_{M0,M4E}_predD_ns{NSIDE}.pdf
  figures/transfer_D_ns{NSIDE}.pdf

Run:  OMP_NUM_THREADS=6 FSB_NSIDE=1024 python method_d.py
      FSB_NSIDE=1024 FSB_PLOTONLY=1 python method_d.py   # figures from saved npz
"""
import os
import time
import numpy as np
import healpy as hp
import pymaster as nmt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import get_context
import simlib as L

NREAL = 100
NMASK = 4
SRC_FIX = int(os.environ.get("FSB_SRCFIX", "1"))   # as predict_transfer
SI = L.SCHEMES.index("W1W2")
FIGDIR = L.FIGDIR
ELL_MAX = 768
COLORS = {"none": "tab:red", "W2": "tab:orange", "W1W2": "tab:blue"}
VARIANTS = ["D1", "D2"]            # raw window / kernel-smoothed window

# globals shared with fork()ed workers
MASKS, N2BAR_GLOB, DS_FIX, CL_EE, CL_BB = None, None, None, None, None


def ensemble_n2(nside):
    """Mask-independent ensemble-mean noise variance density (as in run_masked).
    Used ONLY to reproduce the run's W2 / standard workspaces; the dummy-window
    W1 uses the fixed SRC_FIX realization (the ensemble mean is nearly flat)."""
    acc = np.zeros(L.NPIX)
    for r in range(1, NREAL + 1):
        ds = L.load_delta_source(r, nside)
        acc += L.SIGMA_E**2 / np.clip(L.N_BAR_SR * L.OMEGA_PIX * (1.0 + ds), 1e-3, None)
    return acc / NREAL


def build_A(ws_std, ws_D, bins, ncls):
    """Binned re-decoupling matrix A = M_{D,b}^{-1} M_{std,b}: columns are unit
    bandpower vectors pushed through couple(std) -> decouple(D).  Exact (the
    same stepwise-bandpower convention as NaMaster's binned MCM)."""
    nb = bins.get_n_bands()
    A = np.zeros((ncls * nb, ncls * nb))
    for j in range(ncls * nb):
        e = np.zeros((ncls, nb))
        e[j // nb, j % nb] = 1.0
        A[:, j] = ws_D.decouple_cell(ws_std.couple_cell(bins.unbin_cell(e))).ravel()
    return A


def worker(mi):
    """One mask: per band build the standard and the two dummy-window
    workspaces (spin0x0 and spin0x4) and form the re-decoupling matrices A;
    on mask 0 also spot-check A against direct D-decoupling of realization 1."""
    t0 = time.time()
    nside, lmax = L.NSIDE, L.LMAX
    bins = L.make_bins(lmax)
    nb = bins.get_n_bands()
    bands = L.FILTER_BANDS
    mask = MASKS[mi]
    n2bar = N2BAR_GLOB * mask
    # fixed survey window for the dummy-mask weight (as methods B/C)
    W1fix = L.make_W1(L.noise_variance(DS_FIX, mask), mask, "W1W2")
    cgamma = 0.5 * (CL_EE + CL_BB)
    A00 = {v: np.zeros((len(bands), nb, nb)) for v in VARIANTS}
    A04 = {v: np.zeros((len(bands), 2 * nb, 2 * nb)) for v in VARIANTS}
    checks = {}

    if mi == 0:   # spot-check inputs (per-realization W1, as in the run)
        g1, g2, dl, ds = L.load_cached(1, nside)
        n2r = L.noise_variance(ds, mask)
        dlm = (dl - np.sum(dl * mask) / np.sum(mask)) * mask
        f0d = L.field0(mask, dlm)
        aE, aB = L.shear_to_bandalm(g1, g2, L.make_W1(n2r, mask, "W1W2"), lmax)

    for bi, (l0, l1) in enumerate(bands):
        c = L.band_signal_c(CL_EE, CL_BB, l0, l1)
        W2 = L.make_W2(n2bar, mask, "W1W2", l0, l1, c)   # identical to run_masked
        bk = L.response_beam(l0, l1, cgamma, lmax)       # unit-mean kernel kappa
        wD = {"D1": W2 * W1fix**2,
              "D2": W2 * hp.smoothing(W1fix**2, beam_window=bk)}
        ws00_std = L.workspace00(mask, W2, bins)
        ws04_std = L.workspace04(mask, W2, bins)
        for v in VARIANTS:
            ws00_D = L.workspace00(mask, wD[v], bins)
            ws04_D = L.workspace04(mask, wD[v], bins)
            A00[v][bi] = build_A(ws00_std, ws00_D, bins, 1)
            A04[v][bi] = build_A(ws04_std, ws04_D, bins, 2)
            if mi == 0:
                m0, m4e, m4b = L.band_fields(aE, aB, l0, l1, nside, lmax)
                fM = nmt.NmtField(W2, [m0], lmax=lmax)
                pcl = nmt.compute_coupled_cell(f0d, fM)
                std = ws00_std.decouple_cell(pcl)[0]
                dirD = ws00_D.decouple_cell(pcl)[0]
                checks[f"err00_{v}_b{bi}"] = (np.max(np.abs(A00[v][bi] @ std - dirD))
                                              / np.max(np.abs(dirD)))
                f4 = nmt.NmtField(W2, [m4e, m4b], lmax=lmax, spin=4)
                pcl4 = nmt.compute_coupled_cell(f0d, f4)
                std4 = ws04_std.decouple_cell(pcl4)
                dir4 = ws04_D.decouple_cell(pcl4)
                eonly = A04[v][bi][:nb, :nb] @ std4[0]
                checks[f"err04_Eonly_{v}_b{bi}"] = (np.max(np.abs(eonly - dir4[0]))
                                                   / np.max(np.abs(dir4[0])))
            del ws00_D, ws04_D
        del ws00_std, ws04_std
        print(f"  mask{mi} band{bi} done ({time.time() - t0:.0f}s)", flush=True)
    return mi, A00, A04, checks


def compute():
    global MASKS, N2BAR_GLOB, DS_FIX, CL_EE, CL_BB
    t0 = time.time()
    nside = L.NSIDE
    bins = L.make_bins(L.LMAX)
    nb = bins.get_n_bands()
    nbands = len(L.FILTER_BANDS)
    MASKS = L.load_masks(nside)
    gt = np.load(os.path.join(L.DATA_OUT, f"ground_truth_ns{nside}.npz"))
    CL_EE, CL_BB = gt["cl_ee"], gt["cl_bb"]
    DS_FIX = L.load_delta_source(SRC_FIX, nside)
    N2BAR_GLOB = ensemble_n2(nside)
    print(f"ensemble n2 built ({time.time() - t0:.0f}s)", flush=True)

    with get_context("fork").Pool(NMASK) as pool:
        results = pool.map(worker, range(NMASK))
    A00 = {v: np.zeros((NMASK, nbands, nb, nb)) for v in VARIANTS}
    A04 = {v: np.zeros((NMASK, nbands, 2 * nb, 2 * nb)) for v in VARIANTS}
    checks = {}
    for mi, a00, a04, ch in results:
        for v in VARIANTS:
            A00[v][mi], A04[v][mi] = a00[v], a04[v]
        checks.update(ch)
    print(f"A matrices built ({time.time() - t0:.0f}s)", flush=True)
    for k, v in sorted(checks.items()):
        print(f"  spot-check {k}: {v:.2e}", flush=True)
    # E<-B leakage of the spin-4 transform (B not stored; must be negligible)
    eb = max(np.max(np.abs(A04[v][:, :, :nb, nb:])) for v in VARIANTS)
    print(f"  max |A04_EB| = {eb:.2e}", flush=True)

    # transform the saved per-mask W1W2 bandpowers, interleave samp = k*4 + mi
    out = {f"{k}_{v}": np.zeros((nbands, NREAL * NMASK, nb))
           for k in ["sig_m0", "tot_m0", "sig_m4e", "tot_m4e"] for v in VARIANTS}
    for mi in range(NMASK):
        part = np.load(os.path.join(L.DATA_OUT, f"masked_ns{nside}_m{mi}.npz"),
                       allow_pickle=True)
        for bi in range(nbands):
            for key in ["sig_m0", "tot_m0", "sig_m4e", "tot_m4e"]:
                x = part[key][SI, bi]                       # [nreal, nb]
                for v in VARIANTS:
                    Am = (A00[v][mi, bi] if key.endswith("m0")
                          else A04[v][mi, bi, :nb, :nb])
                    out[f"{key}_{v}"][bi, mi::NMASK] = x @ Am.T

    # residual scalar calibration: g_edge = T_C of the unweighted scheme (N_w=1)
    pr = np.load(os.path.join(L.DATA_OUT, f"transfer_pred_ns{nside}.npz"))
    g_edge = pr["T_C"][0, :, 0]

    np.savez(os.path.join(L.DATA_OUT, f"method_d_ns{nside}.npz"),
             ells=bins.get_effective_ells(), bands=np.array(L.FILTER_BANDS),
             g_edge=g_edge, max_A04_EB=eb,
             **{f"A00_{v}": A00[v] for v in VARIANTS},
             **{f"check_{k}": val for k, val in checks.items()},
             **out)
    print(f"saved method_d_ns{nside}.npz ({time.time() - t0:.0f}s)", flush=True)


def report_and_plot():
    nside = L.NSIDE
    gt = np.load(os.path.join(L.DATA_OUT, f"ground_truth_ns{nside}.npz"))
    mk = np.load(os.path.join(L.DATA_OUT, f"masked_ns{nside}.npz"), allow_pickle=True)
    md = np.load(os.path.join(L.DATA_OUT, f"method_d_ns{nside}.npz"))
    pr = np.load(os.path.join(L.DATA_OUT, f"transfer_pred_ns{nside}.npz"))
    tm = np.load(os.path.join(L.DATA_OUT, f"transfer_measured_ns{nside}.npz"))
    ells = md["ells"]
    bands = [tuple(int(x) for x in b) for b in md["bands"]]
    g_edge, T_C = md["g_edge"], pr["T_C"]
    sel = ells <= ELL_MAX
    truth = {"M0": gt["cl_m0"].mean(1), "M4E": gt["cl_m4e"].mean(1)}
    # idealized residual scalar per variant: D1 -> g_edge, D2 -> 1 (edge absorbed)
    resid_cal = {"D1": g_edge, "D2": np.ones_like(g_edge)}

    lines = [f"method (D): W1W2 matrix-decoupled; g_edge = "
             + np.array2string(g_edge, precision=4)]
    for k in md.files:
        if k.startswith("check_"):
            lines.append(f"  spot-check {k[6:]}: {float(md[k]):.2e}")
    lines.append(f"  max |A04_EB| (spin-4 E<-B leakage, B dropped): {float(md['max_A04_EB']):.2e}")
    for v in VARIANTS:
        for fname, key in [("M0", "m0"), ("M4E", "m4e")]:
            for bi, band in enumerate(bands):
                tr = truth[fname][bi]
                corr = md[f"tot_{key}_{v}"][bi] / resid_cal[v][bi]
                cmean = corr.mean(0)
                C = np.cov(corr, rowvar=False)
                err_cm = np.sqrt(np.clip(np.diag(C), 0, None) / corr.shape[0])
                m = sel & (err_cm > 0)
                chi2 = float(np.sum(((cmean - tr)[m] / err_cm[m]) ** 2))
                trs = tr[sel]
                snr = float(np.sqrt(trs @ np.linalg.solve(C[np.ix_(sel, sel)], trs)))
                TD = md[f"sig_{key}_{v}"][bi].mean(0) / tr
                good = np.abs(tr) > 0.15 * np.abs(tr).max()
                lines.append(f"{v} {fname:4s} band{bi}{band}: <T_{v}>="
                             f"{np.nanmedian(TD[good]):.3f} (g_edge={g_edge[bi]:.3f})"
                             f"  chi2_truth={chi2:.1f}/{int(m.sum())}  SNR={snr:.1f}")
    rep = "\n".join(lines)
    print(rep)
    with open(os.path.join(L.DATA_OUT, f"method_d_ns{nside}.txt"), "w") as f:
        f.write(rep + "\n")

    # ---- residual-transfer figure: std vs D1/D2 decoupling, W1W2 (M0; the
    #      M4E transfer is too noise-dominated to be informative here) ----
    vcol = {"D1": "tab:green", "D2": "tab:purple"}
    fig, axes = plt.subplots(1, len(bands), figsize=(5 * len(bands), 4.0),
                             squeeze=False, sharey=True)
    for bi, band in enumerate(bands):
        ax = axes[0, bi]
        tr = truth["M0"][bi]
        good = np.abs(tr) > 0.15 * np.abs(tr).max()
        ax.plot(ells[good], tm[f"T_M0_b{bi}_W1W2"][good], "o", ms=6, color="tab:blue",
                label="std decoupling (measured $T$)")
        ax.plot(ells, T_C[2, bi], "--", lw=1.4, color="tab:blue", alpha=0.8,
                label=r"(C): $g_{\rm edge}N_{\rm weight}$")
        for v, lab in [("D1", r"(D) raw window $W_2 m W_1^2$"),
                       ("D2", r"(D) smoothed window $W_2 m\,(\kappa\!\star\!W_1^2)$")]:
            TD0 = md[f"sig_m0_{v}"][bi].mean(0) / tr
            ax.plot(ells[good], TD0[good], "o", ms=6, color=vcol[v], label=lab)
        ax.axhline(g_edge[bi], color="k", lw=1.2, ls=":", label=r"$g_{\rm edge}$")
        ax.axhline(1.0, color="grey", lw=0.8, ls="-", alpha=0.6)
        ax.set_xlim(0, ELL_MAX); ax.set_ylim(0.75, 1.25)
        ax.set_title(rf"residual transfer, W1W2, band {band}", fontsize=10)
        ax.set_xlabel(r"$\ell$")
        if bi == 0:
            ax.set_ylabel(r"$\langle\hat C^{\rm sig}_\ell\rangle/C^{\rm truth}_\ell$")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"transfer_D_ns{nside}.pdf")); plt.close(fig)

    # ---- cross-spectra figures (equivalents of the predB/predC ones):
    #      none/W2 as in (C); W1W2 from the (D2) matrix decoupling, no transfer ----
    for fname, key in [("M0", "m0"), ("M4E", "m4e")]:
        fig, axes = plt.subplots(2, len(bands), figsize=(6 * len(bands), 5.6), squeeze=False,
                                 sharex="col", gridspec_kw={"height_ratios": [3, 1]})
        for bi, band in enumerate(bands):
            ax, axr = axes[0, bi], axes[1, bi]
            tr = truth[fname][bi]
            corr, errm = {}, {}
            for si, s in enumerate(L.SCHEMES):
                if s == "W1W2":
                    cv = md[f"tot_{key}_D2"][bi]
                else:
                    cv = mk[f"tot_{key}"][si, bi] / T_C[si, bi]
                corr[s] = cv.mean(0)
                errm[s] = np.sqrt(np.clip(np.diag(np.cov(cv, rowvar=False)), 0, None)
                                  / cv.shape[0])
            sig0 = errm["none"]
            ax.plot(ells, tr, "k-", lw=2, label="truth (full sky)", zorder=5)
            for si, s in enumerate(L.SCHEMES):
                off = (si - 1) * (ells[1] - ells[0]) * 0.18
                lab = s + (" (D, matrix)" if s == "W1W2" else "")
                ax.errorbar(ells + off, corr[s], yerr=errm[s], fmt="o", ms=3, color=COLORS[s],
                            alpha=0.8, capsize=2, label=lab)
                with np.errstate(divide="ignore", invalid="ignore"):
                    axr.errorbar(ells + off, (corr[s] - tr) / sig0, yerr=errm[s] / sig0,
                                 fmt="o", ms=3, color=COLORS[s], alpha=0.8, capsize=2)
            ax.axhline(0, color="grey", lw=0.6); ax.set_xlim(0, ELL_MAX)
            ax.set_title(rf"$C_\ell^{{\delta {fname}}}$ (method (D))  band {band}")
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
        fig.savefig(os.path.join(FIGDIR, f"cross_{fname}_predD_ns{nside}.pdf")); plt.close(fig)
    print(f"figures -> {FIGDIR}")


if __name__ == "__main__":
    if not os.environ.get("FSB_PLOTONLY"):
        compute()
    report_and_plot()
