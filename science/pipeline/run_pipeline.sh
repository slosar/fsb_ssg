#!/bin/bash
# Full FSB-SSG science-validation pipeline at a given resolution.
#   FSB_SIM_DIR=/path/to/sims FSB_NSIDE=256  bash run_pipeline.sh   # validation
#   FSB_SIM_DIR=/path/to/sims FSB_NSIDE=1024 bash run_pipeline.sh   # production
#
# Set FSB_PYTHON to the interpreter with pymaster/healpy/fsb_ssg installed
# (default: `python`).  Stages each read the previous stage's products in ./data/
# and write figures into ../report/figures/:
#   build_cache       degraded-map cache (skipped at NSIDE=1024, maps are native)
#   ground_truth      full-sky, no-noise, unit-weight truth
#   run_masked        100 sims x 4 masks = 400 (parallel per-mask at NSIDE>=1024)
#   combine_masks     merge per-mask outputs
#   analyze           chi2, (A)-corrected cross-spectra, SNR, cormat; saves T_A
#   predict_transfer  T_B/T_C, transfer comparison, fiducial (B)-corrected cross
#   method_d          (D) dummy-window mode-coupling calibration + figures
#   nmt_covariance    NaMaster Gaussian covariance vs sample scatter
set -e
PY=${FSB_PYTHON:-python}
cd "$(dirname "$0")"
NS=${FSB_NSIDE:-512}
export FSB_NSIDE=$NS
log(){ echo "=== [$(date +%H:%M:%S)] $* ==="; }

if [ "$NS" -lt 1024 ]; then
  log "cache (NSIDE=$NS)";      OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} $PY build_cache.py
else
  log "cache: skipped (NSIDE>=1024 maps are native; load_cached reads FSB_SIM_DIR directly)"
fi
log "ground truth";            OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} $PY ground_truth.py

if [ "$NS" -ge 1024 ]; then
  log "masked run: 4 masks in parallel (6 threads each)"
  for i in 0 1 2 3; do
    OMP_NUM_THREADS=6 FSB_MASKS=$i $PY run_masked.py > data/masked_ns${NS}_m$i.log 2>&1 &
  done
  wait
  log "combine"; $PY combine_masks.py
else
  log "masked run (all masks)"; OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} $PY run_masked.py
fi

log "analyze";          OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} $PY analyze.py
log "predict transfer"; OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} $PY predict_transfer.py
log "method D";         OMP_NUM_THREADS=6 $PY method_d.py
log "namaster cov";     OMP_NUM_THREADS=${OMP_NUM_THREADS:-24} $PY nmt_covariance.py
log "DONE NSIDE=$NS"
