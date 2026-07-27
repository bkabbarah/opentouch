#!/bin/bash
# Overnight orchestration. Everything already in flight finishes on its own;
# this handles the work that DEPENDS on the scene-disjoint training, then
# regenerates the report so there is always something current to read.
#
# Deliberately serial and low-priority: the box is shared, was at load ~700
# with three other users active, so every step is nice 19 and capped at 4 BLAS
# threads. Nothing here starts a new training run.
#
#   tmux new-session -d -s overnight 'bash scripts/overnight.sh'

set -u
cd ~/scratch/bashar/opentouch-gru

PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
RETRIEVAL_CK=logs/2026_06_22-21_01_54-model_OpenTouch-DINOv3-B16-Retrieval-lr_0.0001-b_256-j_8-p_amp/checkpoints/epoch_300.pt
export PYTHONPATH=src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }
refresh() { $PY scripts/collect_results.py --output MORNING_REPORT.md >/dev/null 2>&1 || true; }

# Keep the report current every 10 minutes no matter what else happens, so a
# hang in any single step still leaves a readable summary.
( while true; do refresh; sleep 600; done ) &
REFRESHER=$!
trap 'kill $REFRESHER 2>/dev/null' EXIT

log "overnight orchestration started"
refresh

# ---------------------------------------------------------------- phase 1
# Wait for the two scene-disjoint retrieval trainings. They were launched at
# 16:50 for 300 epochs at ~79s/epoch, so expect completion around 23:15.
log "phase 1: waiting for scene-disjoint training"
while pgrep -f "opentouch_train.main" >/dev/null; do sleep 300; done
log "phase 1: training finished"

# ---------------------------------------------------------------- phase 2
# Retrieval eval on both arms under the SAME scene-disjoint split they were
# trained on. This is the first same-codebase, participant-disjoint version of
# the headline comparison.
log "phase 2: retrieval eval on scene-disjoint checkpoints"
for RUN in p2t_scene_gru p2t_scene_avgpool; do
  CK=$(ls -t logs/$RUN/checkpoints/epoch_*.pt 2>/dev/null | head -1)
  if [ -z "$CK" ]; then log "  no checkpoint for $RUN, skipping"; continue; fi
  for SPLIT in val test; do
    log "  $RUN / $SPLIT  ($CK)"
    nice -n 19 $PY -m opentouch_train.eval \
      --checkpoint "$CK" --data "$DATA" --task-type p2t --split "$SPLIT" \
      --sequence-length 20 --seed 42 --split-group-by scene \
      --output "results_retrieval_scene_${RUN}_${SPLIT}.json" \
      > "/tmp/eval_${RUN}_${SPLIT}.log" 2>&1 || log "  eval failed, see /tmp/eval_${RUN}_${SPLIT}.log"
  done
done
refresh

# ---------------------------------------------------------------- phase 3
# The fully clean participant test: probe using the encoder that was ITSELF
# trained on scene-disjoint data, evaluated on scene-disjoint splits. The
# earlier SCENE run reused a clip-split-trained encoder, so it only half
# answered this.
log "phase 3: probe with the scene-disjoint-trained encoder"
CLEAN_CK=$(ls -t logs/p2t_scene_gru/checkpoints/epoch_*.pt 2>/dev/null | head -1)
if [ -n "$CLEAN_CK" ]; then
  nice -n 19 $PY scripts/probe_rigid.py \
    --checkpoint "$CLEAN_CK" --data "$DATA" \
    --horizon-k 8 --sequence-length 36 --causal-window 20 --min-history 10 \
    --split-group-by scene \
    --output results_probe_rigid_k8_SCENE_CLEANENC.json \
    > /tmp/rigid_scene_cleanenc.log 2>&1 || log "  clean-encoder probe failed"
else
  log "  no scene-disjoint checkpoint, skipping"
fi
refresh

# ---------------------------------------------------------------- phase 4
# Let everything else drain: raw kinematics, magnitude, test-split
# confirmation, both subset probes.
log "phase 4: waiting for remaining probes"
while pgrep -f "probe_rawpose.py|probe_magnitude.py|probe_rigid.py|tactile_subset_probe.py" >/dev/null; do
  sleep 300
done
log "phase 4: all probes finished"

refresh
log "overnight orchestration complete -- see MORNING_REPORT.md"
