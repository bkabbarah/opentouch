#!/bin/bash
# Everything that should run once the participant-disjoint forecasting sweep
# frees the GPUs. Open questions #2 and #3, plus one stale artifact refresh.
#
# Deliberately queued behind the sweep rather than run alongside it: the box is
# shared and the sweep already holds three GPUs. This waits, then runs strictly
# serially at nice 19.
#
#   tmux new-session -d -s postsweep 'bash scripts/post_sweep.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
CLIP_GRU=logs/2026_06_22-21_01_54-model_OpenTouch-DINOv3-B16-Retrieval-lr_0.0001-b_256-j_8-p_amp/checkpoints/epoch_300.pt

export PYTHONPATH=src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4
export WANDB_MODE=offline

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# Wait on the SWEEP SCRIPT, not on regression_main. The sweep launches its 18
# runs back to back, so polling for a training process would eventually sample
# the gap between two runs and start this work while 17 runs remain.
log "waiting for the forecasting sweep to finish"
while pgrep -u "$(whoami)" -f "scene_forecast_sweep.sh" >/dev/null; do sleep 300; done
log "sweep clear, starting post-sweep work"

# ------------------------------------------------------------------ open #3
# Per-joint AUC export at k=16. Run at k=8 only so far, and k=16 is where the
# curl effect is largest (+0.0198 vs +0.0187 at k=8).
log "open #3: per-joint export at k=16"
nice -n 19 $PY scripts/export_per_joint.py \
  --checkpoint "$CLIP_GRU" --data "$DATA" \
  --horizon-k 16 --sequence-length 36 --split-group-by clip --eval-split val \
  --output-prefix per_joint_k16 \
  > /tmp/per_joint_k16.log 2>&1 \
  && log "  wrote per_joint_k16.{json,csv}" \
  || log "  FAILED -- see /tmp/per_joint_k16.log"

# ------------------------------------------------------------------ open #2
# Retrieval bootstrap CIs. Retrieval is the SOTA-relative-to-OpenTouch claim
# and has only a 3-seed std behind it.
#
# Clip-clustered and fixed-gallery by default (see bootstrap_eval.py). The
# scene-disjoint pair matters most: 2.18's 4.40x ratio is the headline that
# currently has no interval at all. --split-group-by is left unset so each
# checkpoint is bootstrapped over the split it was actually trained against.
log "open #2: retrieval bootstrap CIs"
run_boot() {
  local tag=$1 ck=$2 split=$3
  if [ ! -f "$ck" ]; then log "  missing checkpoint for $tag, skipping"; return; fi
  log "  $tag / $split"
  nice -n 19 $PY bootstrap_eval.py \
    --checkpoint "$ck" --data "$DATA" \
    --split "$split" --n-bootstrap 1000 --cluster clip \
    --output "results_bootstrap_${tag}_${split}.json" \
    > "/tmp/boot_${tag}_${split}.log" 2>&1 \
    && log "    ok" || log "    FAILED -- see /tmp/boot_${tag}_${split}.log"
}

for SPLIT in val test; do
  run_boot clip_gru        "$CLIP_GRU"                                    "$SPLIT"
  run_boot scene_gru       logs/p2t_scene_gru/checkpoints/epoch_300.pt    "$SPLIT"
  run_boot scene_avgpool   logs/p2t_scene_avgpool_norelu/checkpoints/epoch_300.pt "$SPLIT"
done

# ------------------------------------------------------------- stale artifact
# results_handframe_validation.json still carries the pre-rename verdict, which
# reads as a failed validation when it is the opposite. Refresh it.
log "refreshing handframe validation"
nice -n 19 $PY scripts/validate_handframe.py \
  --data "$DATA" --split val --horizon-k 8 \
  --output results_handframe_validation.json \
  > /tmp/handframe.log 2>&1 \
  && log "  refreshed" || log "  FAILED -- see /tmp/handframe.log"

log "regenerating MORNING_REPORT.md"
nice -n 19 $PY scripts/collect_results.py --output MORNING_REPORT.md >/dev/null 2>&1 || true

log "post-sweep work complete"
