#!/bin/bash
# Split-seed robustness for the participant-disjoint forecasting result.
#
# THE PROBLEM. Every participant-disjoint number in HANDOFF 2.19/2.20 comes
# from ONE partition: split_seed=42 deals 26 scenes into train 20 / val 3 /
# test 3. "Touch helps for a person the model has never seen" therefore rests
# on three held-out person-locations. A clip-clustered bootstrap cannot see
# this -- it resamples clips *within* those same three scenes. The only way to
# measure it is to redraw the partition.
#
# THE CATCH, and the reason this is not simply 18 more regression runs: the
# frozen tactile encoder (p2t_scene_gru) was itself trained on split_seed=42's
# TRAIN scenes. Re-drawing the regression split alone would hand the new val
# scenes to an encoder that had trained on some of them -- reintroducing
# exactly the leak 2.19 exists to remove, while still being labelled
# participant-disjoint. So each new split seed needs its OWN retrieval encoder.
#
# Cost: 3 encoders (parallel, ~6.5h) then 18 regression runs (3 streams, ~5h).
#
# Model seeds are dropped to 1 per condition, deliberately. Seed spread in
# 2.19 was 1e-6 to 5e-6 on means of ~3e-4 -- utterly negligible. The unknown
# is the PARTITION, so the budget goes to split seeds rather than re-measuring
# a variance already known to be tiny.
#
#   tmux new-session -d -s splitseed 'bash scripts/split_seed_study.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
SPLIT_SEEDS="${SPLIT_SEEDS:-1 2 3}"

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ------------------------------------------------------------------ phase 1
# One scene-disjoint retrieval encoder per split seed, same config as
# p2t_scene_gru (which is the split_seed=42 member of this family).
train_encoder() {
  local ss=$1 gpu=$2 name="p2t_scene_gru_ss${ss}"
  if [ -f "logs/$name/checkpoints/epoch_300.pt" ]; then
    log "  encoder $name already done, skipping"; return 0
  fi
  log "  START encoder $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 19 $PY -m opentouch_train.main \
    --train-data "$DATA" --task-type p2t \
    --model OpenTouch-DINOv3-B16-Retrieval \
    --sequence-length 20 --batch-size 256 --epochs 300 \
    --lr 0.0001 --wd 0.01 --warmup 0.05 --lr-scheduler cosine \
    --precision amp --workers 8 --seed 42 \
    --split-group-by scene --split-seed "$ss" \
    --save-frequency 300 --name "$name" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  encoder $name" || log "  FAILED encoder $name -- /tmp/${name}.log"
}

log "phase 1: scene-disjoint retrieval encoders for split seeds: $SPLIT_SEEDS"
GPU=0
for SS in $SPLIT_SEEDS; do
  train_encoder "$SS" "$GPU" &
  GPU=$((GPU + 1))
  [ "$GPU" -ge 3 ] && GPU=0
done
wait
log "phase 1 complete"

# ------------------------------------------------------------------ phase 2
# Forecasting arms at each split seed, each against ITS OWN encoder.
run_one() {
  local ss=$1 arm=$2 k=$3 gpu=$4
  local name="ss${ss}_${arm}_k${k}"
  local ck="logs/p2t_scene_gru_ss${ss}/checkpoints/epoch_300.pt"

  if [ -f "logs/$name/checkpoints/epoch_300.pt" ]; then
    log "  $name already done, skipping"; return 0
  fi
  if [ "$arm" != "pose" ] && [ ! -f "$ck" ]; then
    log "  $name: encoder $ck missing, SKIPPING (would otherwise train from random init)"
    return 1
  fi

  local extra=()
  case "$arm" in
    frz)     extra=(--tactile-init-checkpoint "$ck" --freeze-tactile-encoder) ;;
    frzshuf) extra=(--tactile-init-checkpoint "$ck" --freeze-tactile-encoder --shuffle-tactile) ;;
    pose)    extra=(--pose-only) ;;
  esac

  log "  START $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 19 $PY -m opentouch_train.regression_main \
    --train-data "$DATA" \
    --horizon-k "$k" --sequence-length 36 --causal --min-history 10 \
    --target-mode rigid_articulation \
    --tactile-correction-input tactile_only \
    --grad-clip-scope per_branch \
    --split-group-by scene --split-seed "$ss" --seed 1 \
    --epochs 300 --val-frequency 50 --save-frequency 300 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

log "phase 2: forecasting arms per split seed"
stream() {
  local arm=$1 gpu=$2
  for ss in $SPLIT_SEEDS; do
    for k in 8 16; do
      run_one "$ss" "$arm" "$k" "$gpu"
    done
  done
  log "stream $arm finished"
}
stream frz     0 &
stream frzshuf 1 &
stream pose    2 &
wait
log "phase 2 complete"

log "split-seed study complete -- summarise with scripts/forecast_ci.py --prefix ss<N>_"
