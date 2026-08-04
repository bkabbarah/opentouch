#!/bin/bash
# IS THE TACTILE BENEFIT JUST A CONTACT SCALAR?
#
# Three observations are all consistent with the tactile contribution being a
# single number per frame -- whether, and how hard, the hand is pressing --
# rather than a spatial representation:
#
#   - it is learnable within 2-4 epochs and then decays, in 25/25 runs (2.22)
#   - a RANDOM frozen encoder captures all of it (2.32: 8/16, mean -0.0020)
#   - deranging the temporal pairing destroys it entirely
#
# --tactile-reduce scalar replaces each tactile frame with its spatial mean,
# broadcast back. Per-frame TOTAL pressure is preserved exactly; every spatial
# pattern is destroyed; temporal structure is untouched; the encoder and its
# parameter count are identical. So this isolates exactly one thing: does it
# matter WHERE the hand is touching, or only WHETHER and HOW HARD.
#
# The random encoder is the vehicle, since 2.32 established it is equivalent
# to the pretrained one and it is the honest baseline.
#
# INFORMATIVE IN BOTH DIRECTIONS, which is why it is worth the GPU time:
#   scalar ~= full  -> the tactile benefit here is one number. Deflationary,
#                      and it unifies all three observations above.
#   scalar <<  full -> spatial structure DOES matter, which strengthens 2.32
#                      into "structure matters, learned structure does not".
#
#   bash scripts/tactile_scalar_ablation.sh
#
# 4 partitions x 2 horizons = 8 runs, ~18 min each, 3 streams -> ~1 hour.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PY=~/miniconda3/envs/opentouch/bin/python
DATA=../opentouch/preprocessed_data/train_dataset
HORIZONS="2 8"
PARTITIONS="42 1 2 3"
GPUS=(0 3 4)

log() { echo "[$(date "+%m-%d %H:%M:%S")] $*"; }

for s in randenc apranenc; do
    while tmux has-session -t "$s" 2>/dev/null; do
        log "waiting for $s to release the GPUs"; sleep 300
    done
done

JOBS=()
for k in $HORIZONS; do for ss in $PARTITIONS; do JOBS+=("$k $ss"); done; done
log "tactile scalar ablation: ${#JOBS[@]} runs over horizons [$HORIZONS] x partitions [$PARTITIONS]"

run_one() {
    local k="$1" ss="$2" gpu="$3"
    local name="tsc${k}_${ss}_rndscalar_k${k}_s1"
    if [ -f "logs/$name/out.log" ] && grep -q "Eval Epoch: 60" "logs/$name/out.log" 2>/dev/null; then
        log "  SKIP  $name"; return
    fi
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
        --tactile-reduce scalar \
        --name "$name" > "/tmp/${name}.log" 2>&1
    log "  DONE  $name (exit $?)"
}

worker() {
    local offset="$1" gpu="${GPUS[$1]}" i
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
            --logs logs --data "$DATA" --prefix "tsc${k}_${ss}_" \
            --out "results_aperture_scalar_k${k}_ss${ss}.json"
    done
done | tee /tmp/tactile_scalar_collect.txt
log "complete"
