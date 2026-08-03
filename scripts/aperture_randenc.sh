#!/bin/bash
# RANDOM-ENCODER CONTROL ON GRIP APERTURE, ALL FOUR HORIZONS.
#
# WHY THIS AND NOT THE DELTA-MSE VERSION. HANDOFF 2.22 argues MSE is the wrong
# instrument for this effect: the corrected delta is close to unpredictable
# (best linear model R^2 = 0.19), MSE-optimal prediction collapses toward zero,
# and every contribution fights over an 8-9% sliver. Testing "does the learned
# representation matter downstream" with that instrument is low power, which is
# plausibly why the k=8 result did not replicate at k=2 (see 2.31).
#
# Grip aperture is the better vehicle on every axis:
#   - AUC and R^2 on a scalar with a real sign, not MSE on a 63-d target
#   - it is where the signal actually is (+0.11 AUC capacity-matched, 16/16)
#   - it is the downstream result the paper would lead with (2.23-2.26)
#   - 60 epochs, not 300, so it is ~4x cheaper per run
#
# WHAT IS RUN. Only the two arms that do not exist yet -- a random frozen
# encoder and its own shuffled control -- at every horizon and every partition,
# so the comparison against the pretrained arms already in results/ is
# like-for-like. 4 horizons x 4 partitions x 2 arms = 32 runs, ~18 min each.
#
# Protocol matches the existing aperture runs exactly (60 epochs, val every 2,
# epoch chosen on val by R^2 and scored on TEST by aperture_select_and_test.py).
#
#   bash scripts/aperture_randenc.sh
#
# Waits for the randenc tmux session to release the GPUs before starting.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
HORIZONS="2 4 8 16"
PARTITIONS="42 1 2 3"
GPUS=(0 3 4)

log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

while tmux has-session -t randenc 2>/dev/null; do
    log "waiting for the randenc sweep to release the GPUs"
    sleep 300
done
log "GPUs free; starting"

JOBS=()
for k in $HORIZONS; do
    for ss in $PARTITIONS; do
        for arm in rndfrz rndfrzshuf; do
            JOBS+=("$k $ss $arm")
        done
    done
done
log "random-encoder aperture control: ${#JOBS[@]} runs over horizons [$HORIZONS] x partitions [$PARTITIONS]"

run_one() {
    local k="$1" ss="$2" arm="$3" gpu="$4"
    local name="arn${k}_${ss}_${arm}_k${k}_s1"
    if [ -f "logs/$name/out.log" ] && grep -q "Eval Epoch: 60" "logs/$name/out.log" 2>/dev/null; then
        log "  SKIP  $name"
        return
    fi
    local extra=""
    [ "$arm" = "rndfrzshuf" ] && extra="--shuffle-tactile"
    log "  START $name (gpu $gpu)"
    CUDA_VISIBLE_DEVICES="$gpu" $PY -m opentouch_train.regression_main \
        --train-data "$DATA" \
        --horizon-k "$k" --sequence-length 36 --causal --min-history 10 \
        --target-mode grip_aperture \
        --tactile-correction-input tactile_only \
        --grad-clip-scope per_branch \
        --split-group-by scene --split-seed "$ss" --seed 1 \
        --epochs 60 --val-frequency 2 --save-frequency 2 \
        --freeze-random-tactile-encoder --freeze-tactile-encoder \
        $extra --name "$name" > "/tmp/${name}.log" 2>&1
    log "  DONE  $name (exit $?)"
}

worker() {
    local offset="$1" gpu="${GPUS[$1]}"
    local i
    for (( i=offset; i<${#JOBS[@]}; i+=${#GPUS[@]} )); do
        run_one ${JOBS[$i]} "$gpu"
    done
    log "worker $offset finished"
}

worker 0 & worker 1 & worker 2 &
wait

log "training complete; selecting on val, scoring on TEST"
for k in $HORIZONS; do
    for ss in $PARTITIONS; do
        log "  collecting k=$k partition $ss"
        $PY scripts/aperture_select_and_test.py \
            --logs logs --data "$DATA" --prefix "arn${k}_${ss}_" \
            --out "results_aperture_randenc_k${k}_ss${ss}.json"
    done
done | tee /tmp/aperture_randenc_collect.txt

log "comparing against the pretrained arms already in results/"
$PY scripts/collect_aperture_randenc.py --out results_aperture_randenc_summary.json \
    | tee /tmp/aperture_randenc_summary.txt
log "complete"
