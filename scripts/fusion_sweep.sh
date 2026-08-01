#!/bin/bash
# Does a better FUSION rescue the forecaster?
#
# The diagnosis this tests: the forecasting ceiling is low but not zero (best
# linear model explains R^2=0.19; pose-only beats copy-zero by 8-9%), and the
# scalar gate is the most likely fixable bottleneck. It can only say "touch
# shifts the output by this much" -- it cannot say "touch changes how pose
# should be read", which is the conditional form the effect plausibly takes.
#
# Partition and everything else are held at the 2.19 settings (split_seed 42,
# scene-disjoint, frozen scene-trained encoder), so the film arms are directly
# comparable to the gate arms already on disk as sf_*.
#
# 12 runs: {frz, frzshuf} x {gate, film} x 3 seeds, k=8.
#   - the gate arms are RERUN rather than reused so both fusions come from the
#     same code revision; sf_* predates the film change
#   - pose-only is NOT rerun: it has no tactile branch, so fusion cannot touch
#     it, and sf_pose_k8_s{1,2,3} is already this exact configuration
#
# Read frz-vs-frzshuf WITHIN each fusion. Across fusions the trainable counts
# differ (66,045 gate vs 74,430 film), so only the within-fusion contrast is
# capacity-matched.
#
#   tmux new-session -d -s fusion 'bash scripts/fusion_sweep.sh'

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

run_one() {
  local fusion=$1 arm=$2 seed=$3 gpu=$4
  local name="fu_${fusion}_${arm}_k${K}_s${seed}"

  if [ -f "logs/$name/checkpoints/epoch_300.pt" ]; then
    log "  $name already done, skipping"; return 0
  fi

  local extra=(--tactile-init-checkpoint "$SCENE_CK" --freeze-tactile-encoder)
  [ "$arm" = "frzshuf" ] && extra+=(--shuffle-tactile)

  log "  START $name (gpu $gpu)"
  CUDA_VISIBLE_DEVICES=$gpu nice -n 19 $PY -m opentouch_train.regression_main \
    --train-data "$DATA" \
    --horizon-k "$K" --sequence-length 36 --causal --min-history 10 \
    --target-mode rigid_articulation \
    --tactile-correction-input tactile_only \
    --grad-clip-scope per_branch \
    --fusion "$fusion" \
    --split-group-by scene --split-seed 42 --seed "$seed" \
    --epochs 300 --val-frequency 50 --save-frequency 300 \
    --name "$name" "${extra[@]}" \
    > "/tmp/${name}.log" 2>&1 \
    && log "  DONE  $name" || log "  FAILED $name -- /tmp/${name}.log"
}

# One stream per (fusion, arm) pair so a partial sweep still yields complete
# frz/frzshuf pairs at whatever seeds have landed.
stream() {
  local fusion=$1 arm=$2 gpu=$3
  for seed in 1 2 3; do
    run_one "$fusion" "$arm" "$seed" "$gpu"
  done
  log "stream ${fusion}/${arm} finished"
}

log "fusion sweep: gate vs film, frozen scene-disjoint encoder, k=$K"
stream film frz        0 &
stream film frzshuf    1 &
stream gate frz        2 &
wait
log "first three streams done; running the last arm"
stream gate frzshuf    0
log "fusion sweep complete"

$PY scripts/summarize_fusion.py --logs logs --out results_fusion_sweep.json \
  | tee /tmp/fusion_summary.txt
