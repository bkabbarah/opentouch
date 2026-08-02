#!/bin/bash
# Does the grip-aperture result survive redrawing the participant partition?
#
# THE TEST THAT MATTERS. On split_seed 42 the aperture result is clean and out
# of sample: touch nearly doubles pose-only's explained variance (test R^2
# 0.102 -> 0.187) and lifts open/close AUC 0.601 -> 0.665. Section 2.19 looked
# exactly this clean before redrawing the partition moved it from -15% to +5%,
# so a single-partition result is not yet a result.
#
# CHEAP THIS TIME. The split-seed study already trained a scene-disjoint
# retrieval encoder per partition (p2t_scene_gru_ss{1,2,3}), so no 6.5-hour
# encoder retraining is needed -- each partition just uses ITS OWN encoder.
# Reusing seed 42's encoder on a redrawn split would hand the new validation
# scenes to an encoder that trained on some of them, which is the leak the
# whole scene-disjoint apparatus exists to prevent.
#
# Same honest protocol as the seed-42 run: 60 epochs, checkpoint every 2,
# stopping epoch chosen on VAL by R^2, number reported on TEST.
#
# 18 runs: {pose, frz, frzshuf} x split seeds {1,2,3} x model seeds {1,2}.
# Two model seeds because the seed-42 spread was tiny (test R^2 0.194/0.192/
# 0.174) -- the unknown here is the PARTITION, so the budget goes there.
#
#   tmux new-session -d -s apsplit 'bash scripts/aperture_splitseed.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
K=8
EPOCHS=60
SPLIT_SEEDS="${SPLIT_SEEDS:-1 2 3}"

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for ss in $SPLIT_SEEDS; do
  [ -f "logs/p2t_scene_gru_ss${ss}/checkpoints/epoch_300.pt" ] || {
    log "FATAL: encoder for split seed $ss missing"; exit 1; }
done
log "all three split-seed encoders present"

run_one() {
  local ss=$1 arm=$2 seed=$3 gpu=$4
  local name="asp${ss}_${arm}_k${K}_s${seed}"
  local ck="logs/p2t_scene_gru_ss${ss}/checkpoints/epoch_300.pt"
  [ -d "logs/$name" ] && { log "  $name exists, skipping"; return 0; }

  local extra=()
  if [ "$arm" = "pose" ]; then
    extra=(--pose-only)
  else
    # Each partition uses ITS OWN encoder -- see header.
    extra=(--tactile-init-checkpoint "$ck" --freeze-tactile-encoder)
    [ "$arm" = "frzshuf" ] && extra+=(--shuffle-tactile)
  fi

  log "  START $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 19 $PY -m opentouch_train.regression_main \
    --train-data "$DATA" \
    --horizon-k "$K" --sequence-length 36 --causal --min-history 10 \
    --target-mode grip_aperture \
    --tactile-correction-input tactile_only \
    --grad-clip-scope per_branch \
    --split-group-by scene --split-seed "$ss" --seed "$seed" \
    --epochs "$EPOCHS" --val-frequency 2 --save-frequency 2 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

stream() {
  local arm=$1 gpu=$2
  for ss in $SPLIT_SEEDS; do
    for seed in 1 2; do run_one "$ss" "$arm" "$seed" "$gpu"; done
  done
  log "stream $arm finished"
}

log "aperture split-seed robustness: partitions $SPLIT_SEEDS, 60 epochs, val-select/test-report"
stream pose    0 &
stream frz     1 &
stream frzshuf 4 &
wait
log "training complete; selecting on val and scoring on test"

for ss in $SPLIT_SEEDS; do
  log "  partition $ss"
  $PY scripts/aperture_select_and_test.py \
    --logs logs --data "$DATA" --prefix "asp${ss}_" \
    --out "results_aperture_splitseed_ss${ss}.json"
done | tee /tmp/aperture_splitseed.txt
log "complete"
