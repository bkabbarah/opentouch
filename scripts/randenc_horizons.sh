#!/bin/bash
# THE FORECASTING RANDOM-ENCODER CONTROL AT k=2 AND k=4.
#
# WHY. HANDOFF 2.19d reports that a randomly-initialised frozen tactile
# encoder BEATS the pretrained one on forecasting -- and that is the second
# leg of the paper's contribution 2 ("retrieval quality is a poor proxy for
# downstream utility"). It is k=8, one seed.
#
# HANDOFF 2.27 has since shown that the PROBE version of exactly this claim
# was a k=8 ARTIFACT: at k=2 and k=4 the pretrained encoder gives roughly
# twice the marginal of a random one, 7-9 encoder-seed sd outside noise, and
# k=8 is the single horizon where the two coincide. There is no reason to
# assume the forecasting version behaves differently, and if it does not, a
# claim the paper leans on is horizon-specific.
#
# Every delta-target scene-split run in this repo is k=8, so all arms have to
# be run fresh at each new horizon.
#
# PROTOCOL matches logs/rnd_frz_k8/params.txt exactly so the new numbers are
# comparable to 2.19d: rigid_articulation target, scene-disjoint split, seed
# 42 partition, 300 epochs, seed 1, sequence_length 36, causal.
#
#   bash scripts/randenc_horizons.sh
#
# ~55 min per run, 10 runs, 3 streams -> about 3.5 hours.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
CKPT=logs/p2t_scene_gru/checkpoints/epoch_300.pt
HORIZONS="2 4"

log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

if [ ! -f "$CKPT" ]; then
    log "FATAL: pretrained encoder $CKPT not found"
    exit 1
fi

run_one() {
    local name="$1" gpu="$2"; shift 2
    if [ -f "logs/$name/out.log" ] && grep -q "Eval Epoch: 300" "logs/$name/out.log" 2>/dev/null; then
        log "  SKIP  $name (already complete)"
        return
    fi
    log "  START $name (gpu $gpu)"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m opentouch_train.regression_main \
        --train-data "$DATA" \
        --sequence-length 36 --causal --min-history 10 \
        --target-mode rigid_articulation \
        --tactile-correction-input tactile_only \
        --grad-clip-scope per_branch \
        --split-group-by scene --split-seed 42 --seed 1 \
        --epochs 300 --val-frequency 50 \
        --name "$name" "$@" > "/tmp/${name}.log" 2>&1
    log "  DONE  $name (exit $?)"
}

# Stream A: the pretrained and random arms whose comparison IS the question.
stream_main() {
    for k in $HORIZONS; do
        run_one "rh${k}_frz"    0 --horizon-k "$k" \
            --tactile-init-checkpoint "$CKPT" --freeze-tactile-encoder
        run_one "rh${k}_rndfrz" 0 --horizon-k "$k" \
            --freeze-random-tactile-encoder --freeze-tactile-encoder
    done
    log "stream main finished"
}

# Stream B: the capacity-matched (shuffled) control for each of the above.
# 2.19d reads frz-vs-frzshuf, so both arms need their own shuffled partner.
stream_shuf() {
    for k in $HORIZONS; do
        run_one "rh${k}_frzshuf"    3 --horizon-k "$k" \
            --tactile-init-checkpoint "$CKPT" --freeze-tactile-encoder --shuffle-tactile
        run_one "rh${k}_rndfrzshuf" 3 --horizon-k "$k" \
            --freeze-random-tactile-encoder --freeze-tactile-encoder --shuffle-tactile
    done
    log "stream shuf finished"
}

# Stream C: pose-only baseline, for the vs-pose contrast.
stream_pose() {
    for k in $HORIZONS; do
        run_one "rh${k}_pose" 4 --horizon-k "$k" --pose-only
    done
    log "stream pose finished"
}

log "random-encoder forecasting control at horizons [$HORIZONS], 5 arms, 1 seed"
stream_main & stream_shuf & stream_pose &
wait

log "training complete; collecting"
$PY scripts/collect_randenc.py --horizons $HORIZONS \
    --out results_randenc_horizons.json | tee /tmp/randenc_summary.txt
log "complete"
