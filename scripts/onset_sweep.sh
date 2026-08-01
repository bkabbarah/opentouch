#!/bin/bash
# Can touch ANTICIPATE motion that has not started yet?
#
# Given a hand currently STILL -- articulation motion over the previous k
# frames below --motion-threshold -- will it be moving over the next k? Both
# sides use the same statistic over the same duration, so one threshold means
# the same thing forwards and backwards.
#
# Conditioning on still is what makes this measurable. Over all samples "will
# it move" is answered by "it already is": motion is strongly autocorrelated,
# a pose-only model sits near ceiling, and no headroom is left. Restricting to
# still samples isolates anticipation -- the quantity a controller needs, and
# the one contact forces plausibly precede.
#
# A 2-epoch smoke run gave: still n=3077 of 14,554 val samples, base rate
# 0.463 (nearly balanced, so the task is not degenerate), AUC 0.59 with room
# to climb. That is the profile of a task where a tactile contribution would
# actually be visible.
#
# 9 runs at k=8, gate fusion, scene-disjoint, frozen scene-trained encoder:
#   pose-only / frz / frzshuf, 3 seeds each.
# Whether film helps is being answered on the delta target by the fusion
# sweep; film arms can be added here afterwards rather than guessed at now.
#
# k=8 only: requiring a real k-frame past excludes t < k, which at k=16 would
# leave 4 of 11 timesteps per window.
#
#   tmux new-session -d -s onset 'bash scripts/onset_sweep.sh'

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

log "waiting for the fusion and aperture sweeps to release their GPUs"
while pgrep -u "$(whoami)" -f 'fusion_sweep\.sh|aperture_sweep\.sh' >/dev/null; do sleep 300; done
log "queue clear, starting onset sweep"

run_one() {
  local arm=$1 seed=$2 gpu=$3
  local name="on_${arm}_k${K}_s${seed}"

  if [ -f "logs/$name/checkpoints/epoch_300.pt" ]; then
    log "  $name already done, skipping"; return 0
  fi

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
    --target-mode motion_onset \
    --tactile-correction-input tactile_only \
    --grad-clip-scope per_branch \
    --split-group-by scene --split-seed 42 --seed "$seed" \
    --epochs 300 --val-frequency 50 --save-frequency 300 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

stream() {
  local arm=$1 gpu=$2
  for seed in 1 2 3; do run_one "$arm" "$seed" "$gpu"; done
  log "stream $arm finished"
}

stream pose    0 &
stream frz     1 &
stream frzshuf 2 &
wait
log "onset sweep complete"

$PY scripts/summarize_onset.py --logs logs --out results_onset_sweep.json \
  | tee /tmp/onset_summary.txt
