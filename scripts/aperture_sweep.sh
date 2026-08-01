#!/bin/bash
# Does the low-entropy target rescue the forecaster, and does touch help on it?
#
# The diagnosis: the 63-d delta target is ~81% unpredictable (best linear model
# R^2=0.19), and MSE on a high-entropy near-symmetric target recovers a
# conditional mean near zero -- hence copy-zero being so hard to beat, and
# every tactile contribution fighting over an 8-9% sliver. Grip aperture is one
# scalar a policy actually acts on, and being a distance it is rotation
# invariant, so it needs none of the Kabsch correction the delta targets do.
#
# A 2-epoch smoke run already reached R^2=0.26 and AUC_sign=0.69 against the
# 63-d target's 0.19 ceiling after 300 epochs, which is what motivates the
# full sweep.
#
# 15 runs at k=8, scene-disjoint, frozen scene-trained encoder:
#   pose-only x3            (fusion-independent: no tactile branch to fuse)
#   {gate, film} x {frz, frzshuf} x 3 seeds
#
# Read frz-vs-frzshuf WITHIN a fusion -- that pair is capacity-matched. Across
# fusions the trainable counts differ.
#
# Queued behind the fusion sweep rather than run alongside it: the box is
# shared and that sweep already holds three GPUs.
#
#   tmux new-session -d -s aperture 'bash scripts/aperture_sweep.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
SCENE_CK=logs/p2t_scene_gru/checkpoints/epoch_300.pt
K=8

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

[ -f "$SCENE_CK" ] || { log "FATAL: missing $SCENE_CK"; exit 1; }

log "waiting for the fusion sweep to release its GPUs"
while pgrep -u "$(whoami)" -f "fusion_sweep.sh" >/dev/null; do sleep 300; done
log "fusion sweep clear, starting aperture sweep"

run_one() {
  local arm=$1 fusion=$2 seed=$3 gpu=$4
  local name
  if [ "$arm" = "pose" ]; then
    name="ap_pose_k${K}_s${seed}"
  else
    name="ap_${fusion}_${arm}_k${K}_s${seed}"
  fi

  if [ -f "logs/$name/checkpoints/epoch_300.pt" ]; then
    log "  $name already done, skipping"; return 0
  fi

  local extra=()
  if [ "$arm" = "pose" ]; then
    extra=(--pose-only)
  else
    extra=(--tactile-init-checkpoint "$SCENE_CK" --freeze-tactile-encoder --fusion "$fusion")
    [ "$arm" = "frzshuf" ] && extra+=(--shuffle-tactile)
  fi

  log "  START $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 19 $PY -m opentouch_train.regression_main \
    --train-data "$DATA" \
    --horizon-k "$K" --sequence-length 36 --causal --min-history 10 \
    --target-mode grip_aperture \
    --tactile-correction-input tactile_only \
    --grad-clip-scope per_branch \
    --split-group-by scene --split-seed 42 --seed "$seed" \
    --epochs 300 --val-frequency 50 --save-frequency 300 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

# Three streams. The pose-only baseline goes first in its stream so the
# comparison point exists as early as possible.
stream_a() { for s in 1 2 3; do run_one pose "" "$s" 0; done
             for s in 1 2 3; do run_one frz gate "$s" 0; done
             log "stream A finished"; }
stream_b() { for s in 1 2 3; do run_one frzshuf gate "$s" 1; done
             for s in 1 2 3; do run_one frzshuf film "$s" 1; done
             log "stream B finished"; }
stream_c() { for s in 1 2 3; do run_one frz film "$s" 2; done
             log "stream C finished"; }

stream_a & stream_b & stream_c &
wait
log "aperture sweep complete"

$PY scripts/summarize_aperture.py --logs logs --out results_aperture_sweep.json \
  | tee /tmp/aperture_summary.txt
