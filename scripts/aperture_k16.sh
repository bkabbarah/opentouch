#!/bin/bash
# Grip aperture at the LONGER horizon: does the effect strengthen or decay?
#
# At k=8 the result replicated across all four participant partitions -- touch
# beats a capacity-matched control by +0.11 AUC (positive 4/4) and pose-only by
# +0.054 AUC, out of sample. k=16 (533ms) is the obvious question: on the delta
# target the tactile contribution was larger at k=16 for curl but the pose-only
# margin vanished there, so the horizon genuinely changes the answer.
#
# Identical protocol to the k=8 runs so the two are directly comparable:
# 60 epochs, checkpoint every 2, stopping epoch chosen on VAL by R^2, number
# reported on TEST, each partition using ITS OWN scene-disjoint encoder.
#
# 24 runs: 4 partitions x {pose, frz, frzshuf} x 2 model seeds.
#
# k=16 is fine for this target -- unlike motion_onset it needs no past window,
# so the only cost is a shorter valid-t range (11 timesteps per window rather
# than 19), which means fewer samples but no structural problem.
#
#   tmux new-session -d -s ap16 'bash scripts/aperture_k16.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
K=16
EPOCHS=60
PARTITIONS="${PARTITIONS:-42 1 2 3}"

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

encoder_for() {
  # Partition 42 is the original scene-disjoint encoder; the rest come from
  # the split-seed study. Each partition MUST use its own, or the redrawn
  # validation scenes are handed to an encoder that trained on some of them.
  if [ "$1" = "42" ]; then echo "logs/p2t_scene_gru/checkpoints/epoch_300.pt"
  else echo "logs/p2t_scene_gru_ss$1/checkpoints/epoch_300.pt"; fi
}

for ss in $PARTITIONS; do
  ck=$(encoder_for "$ss")
  [ -f "$ck" ] || { log "FATAL: encoder for partition $ss missing ($ck)"; exit 1; }
done
log "all partition encoders present"

run_one() {
  local ss=$1 arm=$2 seed=$3 gpu=$4
  local name="a16_${ss}_${arm}_k${K}_s${seed}"
  local ck; ck=$(encoder_for "$ss")
  [ -d "logs/$name" ] && { log "  $name exists, skipping"; return 0; }

  local extra=()
  if [ "$arm" = "pose" ]; then
    extra=(--pose-only)
  else
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
  for ss in $PARTITIONS; do
    for seed in 1 2; do run_one "$ss" "$arm" "$seed" "$gpu"; done
  done
  log "stream $arm finished"
}

log "aperture k=$K across partitions $PARTITIONS, val-select/test-report"
stream pose    0 &
stream frz     1 &
stream frzshuf 4 &
wait
log "training complete; selecting on val and scoring on test"

for ss in $PARTITIONS; do
  log "  partition $ss"
  $PY scripts/aperture_select_and_test.py \
    --logs logs --data "$DATA" --prefix "a16_${ss}_" \
    --out "results_aperture_k16_ss${ss}.json"
done | tee /tmp/aperture_k16.txt
log "complete"
