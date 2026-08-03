#!/bin/bash
# Close the three gaps in the probe tables that a reviewer would find.
#
# GPU access ends in ~5 days, so this queues the work that actually NEEDS a
# GPU. The cross-dataset rotation diagnostic is CPU/data work and can happen
# afterwards.
#
# GAP 1 -- SEEDS. The headline table (HANDOFF 2.2) reports k=2/4/8/16 from
# encoder seed 42 ONLY, and 2.6 shows seed 42 is the outlier high (+0.0171 vs
# +0.0124 and +0.0120 at k=8, mean +0.0138). Right now the paper's central
# table is its best seed at three of four horizons. Running seeds 0 and 1 at
# k=2/4/16 makes every cell a mean over three encoders.
#
# GAP 2 -- THE RANDOM-ENCODER CONTROL, at every horizon. It exists only at
# k=8 (2.19e), where a random frozen tactile encoder gives +0.0122 against
# pretrained seeds spanning +0.0120 to +0.0171. If the paper claims the
# predictive signal is in the tactile INPUT rather than the learned
# representation, that control has to hold across horizons, not one point.
#
# GAP 3 -- PARTICIPANT-DISJOINT, at every horizon. Also k=8 only. The
# robustness table should span the same horizons as the headline.
#
# 12 runs, ~20 min each, three at a time.
#
#   tmux new-session -d -s probesweep 'bash scripts/probe_paper_sweep.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

CK42=$(ls -d logs/2026_06_22-21_01_54*/checkpoints/epoch_300.pt 2>/dev/null | head -1)
CK0=$(ls -d logs/2026_07_06-22_35_40*/checkpoints/epoch_latest.pt 2>/dev/null | head -1)
CK1=$(ls -d logs/2026_07_06-22_36_12*/checkpoints/epoch_latest.pt 2>/dev/null | head -1)
CKSCENE=logs/p2t_scene_gru/checkpoints/epoch_300.pt

for c in "$CK42" "$CK0" "$CK1" "$CKSCENE"; do
  [ -f "$c" ] || { log "FATAL: missing checkpoint $c"; exit 1; }
done
log "all four encoders present"

log "waiting for the aperture horizon sweep to release GPUs"
while pgrep -u "$(whoami)" -f 'aperture_horizons\.sh' >/dev/null; do sleep 240; done
log "queue clear"

probe() {
  local out=$1 gpu=$2; shift 2
  if [ -f "$out" ]; then log "  $out exists, skipping"; return 0; fi
  log "  START $out (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 19 $PY scripts/probe_rigid.py \
    --data "$DATA" --sequence-length 36 --output "$out" "$@" \
    > "/tmp/${out%.json}.log" 2>&1 \
    && log "  DONE  $out" || log "  FAILED $out -- /tmp/${out%.json}.log"
}

# --- gap 1: encoder seeds 0 and 1 at the horizons that only have seed 42
stream_seeds() {
  for k in 2 4 16; do
    probe "results_probe_rigid_k${k}_seed0.json" 0 --checkpoint "$CK0" --horizon-k "$k"
    probe "results_probe_rigid_k${k}_seed1.json" 0 --checkpoint "$CK1" --horizon-k "$k"
  done
  log "stream seeds finished"
}

# --- gap 2: random-encoder control at the other horizons
stream_random() {
  for k in 2 4 16; do
    probe "results_probe_rigid_k${k}_RANDENC.json" 1 \
      --checkpoint "$CK42" --horizon-k "$k" --random-tactile-encoder
  done
  log "stream random finished"
}

# --- gap 3: participant-disjoint, clean encoder, at the other horizons
stream_scene() {
  for k in 2 4 16; do
    probe "results_probe_rigid_k${k}_SCENE_CLEANENC.json" 4 \
      --checkpoint "$CKSCENE" --horizon-k "$k" --split-group-by scene
  done
  log "stream scene finished"
}

log "probe paper sweep: 12 runs closing the seed / random-control / participant gaps"
stream_seeds & stream_random & stream_scene &
wait
log "complete"
