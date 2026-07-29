#!/bin/bash
# Does the forecasting gain track TACTILE REPRESENTATION QUALITY?
#
# THE CONFOUND THIS RESOLVES. The split-seed study varied two things at once:
# which participants were held out, AND how well that partition's retrieval
# encoder trained (val T->P mAP 29.1 / 24.3 / 20.8 / 18.7). The forecasting
# gain tracked the second, with the weakest encoder giving the only sign
# reversal -- but partition and encoder quality moved together, so neither can
# be credited.
#
# THE DESIGN. Hold the partition FIXED (split_seed 42, the original) and vary
# only encoder quality, using intermediate checkpoints of p2t_scene_gru as a
# natural quality ladder:
#
#     epoch  20 -> t2p mAP 10.81
#     epoch  40 -> 16.07
#     epoch  60 -> 20.26
#     epoch 100 -> 25.19
#     epoch 300 -> 28.24
#
# That range BRACKETS the three redrawn partitions' encoders (18.74, 20.78,
# 24.29), so this is a prediction, not just a correlation: if quality is what
# drives the gain, the ladder should reproduce their forecasting values at the
# matching mAP. If the gain is instead flat across the ladder, then the
# partition is what matters and the forecasting result is simply fragile.
# Either answer settles it.
#
# frzshuf is run at every rung too, so the capacity-matched control always
# shares its arm's encoder. pose-only needs no rerun -- sf_pose_k8_s{1,2,3}
# is already this exact partition and horizon (0.000290).
#
#   tmux new-session -d -s ladder 'bash scripts/encoder_ladder.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
EPOCHS_LADDER="${EPOCHS_LADDER:-20 40 60 100 300}"
K=8

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

run_one() {
  local ep=$1 arm=$2 gpu=$3
  local name="el${ep}_${arm}_k${K}"
  local ck="logs/p2t_scene_gru/checkpoints/epoch_${ep}.pt"

  if [ -f "logs/$name/checkpoints/epoch_300.pt" ]; then
    log "  $name already done, skipping"; return 0
  fi
  if [ ! -f "$ck" ]; then
    log "  $name: encoder $ck missing, SKIPPING"; return 1
  fi

  local extra=(--tactile-init-checkpoint "$ck" --freeze-tactile-encoder)
  [ "$arm" = "frzshuf" ] && extra+=(--shuffle-tactile)

  log "  START $name (gpu $gpu, encoder epoch $ep)"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 19 $PY -m opentouch_train.regression_main \
    --train-data "$DATA" \
    --horizon-k "$K" --sequence-length 36 --causal --min-history 10 \
    --target-mode rigid_articulation \
    --tactile-correction-input tactile_only \
    --grad-clip-scope per_branch \
    --split-group-by scene --split-seed 42 --seed 1 \
    --epochs 300 --val-frequency 50 --save-frequency 300 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

# One stream per GPU, walking the ladder. Both arms of a rung land close
# together in time so a partial run is still a usable pair.
stream() {
  local arm=$1 gpu=$2
  for ep in $EPOCHS_LADDER; do
    run_one "$ep" "$arm" "$gpu"
  done
  log "stream $arm finished"
}

log "encoder-quality ladder: partition fixed at split_seed 42, encoder epochs: $EPOCHS_LADDER"
stream frz     0 &
stream frzshuf 1 &
wait
log "ladder complete"

log "summary"
$PY scripts/summarize_ladder.py --logs logs --out results_encoder_ladder.json \
  | tee /tmp/ladder_summary.txt
