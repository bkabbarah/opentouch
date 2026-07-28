#!/bin/bash
# Participant-disjoint frozen-encoder forecasting -- HANDOFF open question #1.
#
# This is section 2.17 rerun with every trace of participant leakage removed:
#   - the tactile encoder is initialised from p2t_scene_gru, a retrieval run
#     trained with --split-group-by scene, so it never saw a val participant
#   - the regression itself uses --split-group-by scene, so val/test contain
#     participants absent from train
#
# 2.17 could not make the "unseen person" claim because its frozen encoder came
# from the clip-disjoint retrieval run. This closes that.
#
# 18 runs: {frz, frzshuf, pose} x k in {8,16} x seeds {1,2,3}. pose-only is
# rerun rather than reused -- the existing rr_pose_* baselines are clip-split
# and are not a valid comparison against scene-split arms.
#
# frzshuf is the load-bearing control: identical architecture, identical
# trainable parameter count (66,045), same frozen encoder, but the tactile
# stream is deterministically re-paired with the wrong windows. frz minus
# frzshuf is tactile CONTENT with capacity held fixed.
#
#   tmux new-session -d -s scenefc 'bash scripts/scene_forecast_sweep.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
SCENE_CK=logs/p2t_scene_gru/checkpoints/epoch_300.pt

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [ ! -f "$SCENE_CK" ]; then
  log "FATAL: scene-disjoint retrieval checkpoint missing: $SCENE_CK"
  exit 1
fi

# Etiquette: shared box. Everything is nice 19, capped at 4 BLAS threads, and
# runs as three GPU-pinned streams rather than 18 concurrent jobs. GPUs 5-7
# belong to another user's VLLM; 3 is partly occupied; 4 is left free.
run_one() {
  local arm=$1 k=$2 seed=$3 gpu=$4
  local name="sf_${arm}_k${k}_s${seed}"

  if [ -f "logs/$name/checkpoints/epoch_300.pt" ]; then
    log "  $name already complete, skipping"
    return 0
  fi

  local extra=()
  case "$arm" in
    frz)     extra=(--tactile-init-checkpoint "$SCENE_CK" --freeze-tactile-encoder) ;;
    frzshuf) extra=(--tactile-init-checkpoint "$SCENE_CK" --freeze-tactile-encoder --shuffle-tactile) ;;
    pose)    extra=(--pose-only) ;;
    *)       log "  unknown arm $arm"; return 1 ;;
  esac

  log "  START $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 19 $PY -m opentouch_train.regression_main \
    --train-data "$DATA" \
    --horizon-k "$k" \
    --sequence-length 36 --causal --min-history 10 \
    --target-mode rigid_articulation \
    --tactile-correction-input tactile_only \
    --grad-clip-scope per_branch \
    --split-group-by scene \
    --split-seed 42 \
    --seed "$seed" \
    --epochs 300 --val-frequency 50 --save-frequency 300 \
    --name "$name" \
    "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1

  if [ $? -ne 0 ]; then
    log "  FAILED $name -- see /tmp/${name}.log"
    return 1
  fi
  log "  DONE  $name"
}

# One stream per arm, so the three arms progress together and a partial sweep
# is still a complete comparison at whatever seeds have finished.
stream() {
  local arm=$1 gpu=$2
  for k in 8 16; do
    for seed in 1 2 3; do
      run_one "$arm" "$k" "$seed" "$gpu"
    done
  done
  log "stream $arm finished"
}

log "scene-disjoint forecasting sweep starting (18 runs, 3 streams)"
log "encoder: $SCENE_CK"

stream frz     0 &
P1=$!
stream frzshuf 1 &
P2=$!
stream pose    2 &
P3=$!

wait $P1 $P2 $P3
log "all streams finished"

# ------------------------------------------------------------------ summary
# Parses the final epoch-300 rigid/moving fingertip MSE out of each run and
# writes the comparison table. Also asserts the motion threshold is identical
# across arms within a horizon -- it is auto-computed from the train split, so
# a mismatch would silently make the arms non-comparable.
log "writing summary"
$PY scripts/summarize_scene_forecast.py \
  --logs logs --out results_scene_forecast.json | tee /tmp/scene_forecast_summary.txt

log "sweep complete"
