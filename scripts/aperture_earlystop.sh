#!/bin/bash
# Grip aperture with an HONEST early-stopping protocol.
#
# THE PROBLEM THIS SOLVES. The dense-validation run showed the touch arm peaks
# around epoch 4 (R^2=0.248) and decays to 0.155 by epoch 300, while pose-only
# climbs slowly and is still improving at 60. Reporting either arm at a fixed
# shared epoch mis-states both. But simply reporting each arm at its best
# VALIDATION epoch, with validation metrics, is selection on the eval set --
# the same error as quoting the seed that happened to be highest.
#
# So: the stopping epoch is chosen on VAL, and the number is reported on TEST.
# The test split has been touched by almost nothing in this project, which is
# what makes it a genuine out-of-sample measurement rather than a repeat of the
# set every decision was made against.
#
# 60 epochs, checkpoint every 2, three arms, three seeds. Phase 2 then reads
# each run's val trajectory, picks the best epoch, and evaluates ONLY that
# checkpoint on test.
#
#   tmux new-session -d -s apestop 'bash scripts/aperture_earlystop.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
SCENE_CK=logs/p2t_scene_gru/checkpoints/epoch_300.pt
K=8
EPOCHS=60

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

[ -f "$SCENE_CK" ] || { log "FATAL: missing $SCENE_CK"; exit 1; }

log "waiting for any running sweep to release its GPUs"
while pgrep -u "$(whoami)" -f 'onset_sweep\.sh' >/dev/null; do sleep 180; done
log "queue clear"

run_one() {
  local arm=$1 seed=$2 gpu=$3
  local name="aes_${arm}_k${K}_s${seed}"
  [ -d "logs/$name" ] && { log "  $name exists, skipping"; return 0; }

  local extra=()
  if [ "$arm" = "pose" ]; then
    extra=(--pose-only)
  else
    extra=(--tactile-init-checkpoint "$SCENE_CK" --freeze-tactile-encoder)
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
    --epochs "$EPOCHS" --val-frequency 2 --save-frequency 2 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

stream() {
  local arm=$1 gpu=$2
  for seed in 1 2 3; do run_one "$arm" "$seed" "$gpu"; done
  log "stream $arm finished"
}

log "phase 1: training, $EPOCHS epochs, checkpoint every 2"
stream pose    0 &
stream frz     1 &
stream frzshuf 4 &
wait
log "phase 1 complete"

log "phase 2: select on val, evaluate on test"
$PY scripts/aperture_select_and_test.py \
  --logs logs --data "$DATA" --prefix aes_ --out results_aperture_earlystop.json \
  | tee /tmp/aperture_earlystop.txt
log "complete"
