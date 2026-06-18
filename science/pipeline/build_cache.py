"""Pre-degrade the 100 realizations to the working NSIDE and cache as float32.

Both the full-sky ground truth and the masked 400-realization run reuse these,
so paying the spin-2 / scalar degrade once here removes the dominant per-run
cost.  Output: data/cache_ns{NSIDE}/r{idx:03d}.npz with g1,g2,delta_lens,delta_src.

Run:  FSB_NSIDE=512 python build_cache.py
"""
import os
import time
import numpy as np
import simlib as L

NREAL = int(os.environ.get("FSB_NREAL", "100"))


def main():
    cdir = os.path.join(L.DATA_OUT, f"cache_ns{L.NSIDE}")
    os.makedirs(cdir, exist_ok=True)
    t0 = time.time()
    for r in range(1, NREAL + 1):
        out = os.path.join(cdir, f"r{r:03d}.npz")
        if os.path.exists(out):
            continue
        g1, g2 = L.load_shear(r, L.NSIDE)
        dl = L.load_delta_lens(r, L.NSIDE)
        ds = L.load_delta_source(r, L.NSIDE)
        np.savez(out,
                 g1=g1.astype(np.float32), g2=g2.astype(np.float32),
                 delta_lens=dl.astype(np.float32), delta_src=ds.astype(np.float32))
        if r % 10 == 0 or r == 1:
            print(f"  cached r{r:03d}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"done -> {cdir}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
