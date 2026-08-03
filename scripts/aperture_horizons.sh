#!/bin/bash
# Grip aperture at the SHORT horizons, completing the curve.
#
# k=8 gave touch - shuffled +0.110 AUC and k=16 gave +0.143, both positive on
# all four participant partitions. If the effect keeps declining toward k=2,
# the curve is monotone in horizon and supports the reading that touch carries
# information about where the grip is HEADING rather than about the instant of
# contact. If it is flat or inverted at short horizons that reading is wrong,
# and the paper should not make it.
#
# Identical protocol to the k=8 and k=16 runs: 60 epochs, checkpoint every 2,
# stopping epoch chosen on VAL by R^2, reported on TEST, each partition on its
# own scene-disjoint encoder.
#
# 24 runs: 2 horizons x 4 partitions x 3 arms x 1 model seed. One seed rather
# than two because the seed spread at k=8/k=16 was ~0.01 AUC while the spread
# ACROSS partitions was several times that -- the budget belongs on partitions.
#
#   tmux new-session -d -s aphz 'bash scripts/aperture_horizons.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
EPOCHS=60
HORIZONS="${HORIZONS:-2 4}"
PARTITIONS="${PARTITIONS:-42 1 2 3}"

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

encoder_for() {
  if [ "$1" = "42" ]; then echo "logs/p2t_scene_gru/checkpoints/epoch_300.pt"
  else echo "logs/p2t_scene_gru_ss$1/checkpoints/epoch_300.pt"; fi
}

for ss in $PARTITIONS; do
  ck=$(encoder_for "$ss")
  [ -f "$ck" ] || { log "FATAL: encoder for partition $ss missing"; exit 1; }
done
log "all partition encoders present"

run_one() {
  local k=$1 ss=$2 arm=$3 gpu=$4
  local name="ahz${k}_${ss}_${arm}_k${k}_s1"
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
    --horizon-k "$k" --sequence-length 36 --causal --min-history 10 \
    --target-mode grip_aperture \
    --tactile-correction-input tactile_only \
    --grad-clip-scope per_branch \
    --split-group-by scene --split-seed "$ss" --seed 1 \
    --epochs "$EPOCHS" --val-frequency 2 --save-frequency 2 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

stream() {
  local arm=$1 gpu=$2
  for k in $HORIZONS; do
    for ss in $PARTITIONS; do
      run_one "$k" "$ss" "$arm" "$gpu"
    done
  done
  log "stream $arm finished"
}

log "aperture horizons [$HORIZONS] x partitions [$PARTITIONS], 1 seed"
stream pose    0 &
stream frz     1 &
stream frzshuf 4 &
wait
log "training complete; selecting on val and scoring on test"

for k in $HORIZONS; do
  for ss in $PARTITIONS; do
    log "  k=$k partition $ss"
    $PY scripts/aperture_select_and_test.py \
      --logs logs --data "$DATA" --prefix "ahz${k}_${ss}_" \
      --out "results_aperture_k${k}_ss${ss}.json"
  done
done | tee /tmp/aperture_horizons.txt
log "complete"
