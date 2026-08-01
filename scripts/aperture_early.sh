#!/bin/bash
# Where does the grip-aperture model actually peak?
#
# The 300-epoch sweep ends at R^2=0.158 for the touch arm, but a 2-epoch smoke
# run reached 0.258 and by epoch 50 it was already down to 0.161. With
# --val-frequency 50 the entire rise and fall between epochs 1 and 49 is
# invisible, so the reported number may be a long way past the peak.
#
# That is what the effective-sample-size problem predicts: windows overlap by
# 19 of 20 frames, so 116k training samples are nowhere near 116k independent
# ones, and 300 epochs was inherited from the 63-d delta target rather than
# chosen for this one.
#
# 60 epochs, validation every 2, all three arms, 3 seeds. Checkpoints are not
# saved -- this run exists to locate the peak, not to produce a model.
#
#   tmux new-session -d -s apearly 'bash scripts/aperture_early.sh'

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

run_one() {
  local arm=$1 seed=$2 gpu=$3
  local name="ape_${arm}_k${K}_s${seed}"
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
    --epochs "$EPOCHS" --val-frequency 2 --save-frequency 0 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

stream() {
  local arm=$1 gpu=$2
  for seed in 1 2 3; do run_one "$arm" "$seed" "$gpu"; done
  log "stream $arm finished"
}

log "aperture early-validation: $EPOCHS epochs, val every 2, 3 arms x 3 seeds"
stream pose    0 &
stream frz     1 &
stream frzshuf 4 &
wait
log "complete"

$PY scripts/summarize_aperture_early.py --logs logs | tee /tmp/aperture_early.txt
