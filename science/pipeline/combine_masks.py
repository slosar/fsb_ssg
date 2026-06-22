"""Combine per-mask run_masked outputs (masked_ns{NSIDE}_m{i}.npz) into the
standard masked_ns{NSIDE}.npz with the 4 masks concatenated along the sample
axis, interleaved as (realization, mask) so the 400 samples match the
single-process layout.

Run:  FSB_NSIDE=1024 python combine_masks.py
"""
import os
import numpy as np
import simlib as L


def main():
    nside = L.NSIDE
    parts = []
    for i in range(4):
        p = os.path.join(L.DATA_OUT, f"masked_ns{nside}_m{i}.npz")
        parts.append(np.load(p, allow_pickle=True))
    nreal = int(parts[0]["nreal"])
    keys = ["sig_m0", "tot_m0", "sig_m4e", "tot_m4e", "sig_m4b", "tot_m4b"]
    out = {}
    for key in keys:
        arrs = [p[key] for p in parts]          # each [nsch, nband, nreal, nbin]
        nsch, nband, _, nbin = arrs[0].shape
        comb = np.zeros((nsch, nband, nreal * 4, nbin))
        for mi, a in enumerate(arrs):
            comb[:, :, mi::4, :] = a            # interleave: samp = k*4 + mi
        out[key] = comb
    np.savez(os.path.join(L.DATA_OUT, f"masked_ns{nside}.npz"),
             schemes=parts[0]["schemes"], bands=parts[0]["bands"],
             ells=parts[0]["ells"], nside=nside, lmax=parts[0]["lmax"],
             nreal=nreal, mask_ids=np.array([0, 1, 2, 3]), **out)
    print(f"combined 4 masks -> masked_ns{nside}.npz  "
          f"({out['tot_m0'].shape[2]} samples)")


if __name__ == "__main__":
    main()
