# OpenTouch — working state, updated 2026-07-28

Written so a fresh session (or a fresh person) can pick this up cold. The
conversation is not the source of truth; this file, `AUDIT.md`,
`ABSTRACT_200.md`, the git log, and the result JSONs on the cluster are.

Branch: `audit/frames-subsets-scene-split` (pushed to `origin`, and checked
out on the cluster at `~/scratch/bashar/opentouch-gru`).

---

> **Updated 2026-08-02.** The forecasting story changed twice since the
> 07-28 version. Read §2.23 first, then §2.22, then §2.19f.
>
> **The live claim is §2.23, grip aperture.** Predicting whether the hand is
> about to open or close — a rotation-invariant scalar a policy can act on —
> frozen tactile features beat a capacity-matched control by **+0.11 AUC,
> positive on all four participant partitions**, and beat pose-only by +0.054
> AUC, out of sample with the stopping epoch chosen on val and the number
> reported on test. **It is the only forecasting result that survives
> redrawing the participant partition** — and it gets STRONGER at k=16
> (+0.143 vs shuffled, +0.069 vs pose-only, again 4/4; see §2.24).
>
> **§2.19's 15% is NOT that claim and should not be quoted as one.** Its
> capacity-matched contrast across redrawn partitions went −15.0%, −5.8%,
> −10.0%, **+5.0%** — a sign flip (§2.19c) — and a randomly initialised
> encoder recovers ~71% of it, beating the pretrained one outright (§2.19d).
> Whatever it measures is mostly correct temporal pairing, not the learned
> representation.
>
> The probe results (§2.2-§2.12) and retrieval (§2.1, §2.18, §2.20) are
> unaffected by any of this and remain the solid core.

## 1. Where things stand in one paragraph

The published direction-probe conclusion ("tactile is redundant with pose
kinematics") was an artifact of the prediction target. The codebase's
"articulation delta" removes wrist *translation* only, so whole-hand
*rotation* stays in, and rotation is ~95% of that target's energy at the
median sample. Isolating true finger articulation and expressing it in a
palm-anchored frame raises tactile's marginal contribution over a
matched-temporal pose baseline from +0.0017 to +0.0171 at k=8, and the effect
holds at all four horizons, under participant-disjoint splits, and with
clip-clustered confidence intervals excluding zero at k=8 (at k=16 only curl's
interval clears zero -- see §2.21). Separately, the retrieval
headline had been compared against the *paper's* avg-pool number rather than
this codebase's own re-run, so the correct figure is 2.7x (16.8 to 45.5), not
3.4x.

---

## 2. Results, with exact numbers

### 2.1 Retrieval (Phase 1)

| condition | T→P mAP | source |
|---|---|---|
| avg-pool, this codebase, test | **16.76** | `opentouch/logs/2026_06_22-20_49_48`, `tags.txt` |
| avg-pool, this codebase, final val | 14.42 | same run's `out.log` |
| biGRU, test | **45.46** | `opentouch-gru/logs/2026_06_22-21_01_54` |
| biGRU, final val | 40.97 | same run's `out.log` |
| biGRU, 3 seeds, test | 46.17 ± 0.54 | `experiments.md` |

**2.7x, not "triples".** The 13.43 in earlier drafts is the paper's number and
is not like-for-like. See `AUDIT.md` §1.1.

mAP here is `mean(1/rank)` against the entire eval split as gallery
(`metrics.py:75`), so it is a function of gallery size. Same weights score
14.42 on a 1572-window gallery and 16.76 on a 1399-window gallery. Always
print N next to a mAP.

### 2.2 Direction probe, corrected target (the main result)

Marginal AUC of touch added to `pose_emb_causal` (frozen biGRU pose encoder,
identical 20-frame causal window), best axis, val split, split-seed 42:

| k | ms | n eval | published target | corrected target |
|---|---|---|---|---|
| 2 | 67 | 15,196 | +0.0030 | **+0.0168** |
| 4 | 133 | 14,006 | +0.0036 | **+0.0164** |
| 8 | 267 | 11,425 | +0.0017 | **+0.0171** |
| 16 | 533 | 6,626 | +0.0001 | **+0.0132** |

Touch alone vs matched-temporal pose alone, corrected target, best axis:

| k | touch alone | pose alone | shuffled touch |
|---|---|---|---|
| 2 | 0.6481 | 0.6481 | 0.5013 |
| 4 | 0.6508 | 0.6482 | 0.5008 |
| 8 | 0.6605 | 0.6545 | 0.5009 |
| 16 | **0.6897** | 0.6863 | 0.4963 |

Per-axis marginal (axes renamed, see §3.2):

| k | radial | spread | curl |
|---|---|---|---|
| 2 | +0.0168 | +0.0125 | +0.0179 |
| 4 | +0.0164 | +0.0131 | +0.0169 |
| 8 | +0.0171 | +0.0126 | +0.0187 |
| 16 | +0.0132 | +0.0114 | +0.0198 |

Curl (flexion) is top at all four horizons, but the gap over radial is only
+0.0011 / +0.0005 / +0.0015 at k=2/4/8 and the intervals overlap. Consistent
ordering, **not** a separated effect. Do not defend it as significant.

### 2.3 Confidence intervals (k=8, clip-clustered, 1000 draws, 295 val clips)

`joints CI>0` is counted against matched pose (the vs-shuffled count is given
separately where it differs).

| axis | vs matched pose | vs shuffled twin | joints CI>0 (vs pose / vs shuf) |
|---|---|---|---|
| curl | +0.0187 [+0.0064, +0.0311] | +0.0215 [+0.0082, +0.0354] | 17/20 / 17/20 |
| radial | +0.0171 [+0.0040, +0.0314] | +0.0195 [+0.0055, +0.0346] | 17/20 / 17/20 |
| spread | +0.0126 [+0.0009, +0.0245] | +0.0161 [+0.0022, +0.0300] | **12/20** / 15/20 |

Clustering at the clip level is essential: 11,425 eval samples come from only
295 clips, and adjacent t share 19 of 20 causal frames. A per-sample bootstrap
reports roughly six times tighter and is wrong.

Earlier versions of this table reported spread as 15/20 against matched pose;
that was the vs-shuffled count. `results_rigid_ci_k8_T36.json` has
`flex.vs_pose.n_joints_ci_excludes_zero = 12`. Curl and radial happen to be 17
in both columns, which is why the transcription error survived.

### 2.4 Decomposition: rotation removal vs frame change (k=8)

| marginal | world axes | palm axes |
|---|---|---|
| rotation kept | +0.0017 | +0.0052 |
| rotation removed | +0.0025 | **+0.0171** |

**It is an interaction.** Frame alone buys +0.0035, rotation removal alone
+0.0008, together +0.0154. Both are required. `results_probe_rigid_k8_DECOMP.json`

### 2.5 Participant-disjoint (scene-disjoint) probe, k=8

| target | marginal | touch alone | shuffled |
|---|---|---|---|
| published | **−0.0051** | 0.5915 | 0.5041 |
| corrected | **+0.0165** | 0.6774 | 0.5023 |

The corrected effect survives participant hold-out at essentially full
strength; the published-target effect goes negative. `results_probe_rigid_k8_SCENE.json`

Caveat: the frozen encoder was itself trained with clip-level splits, so it
saw val participants' clips during retrieval training. The fully clean version
using the scene-disjoint-trained encoder is §2.12's row 2 (+0.0225), and the
forecasting equivalent is §2.19.

### 2.6 Encoder seed sensitivity (k=8, corrected target, best axis)

| checkpoint | marginal | touch alone |
|---|---|---|
| seed 42 (`2026_06_22-21_01_54`) | +0.0171 | 0.6605 |
| seed 0 (`2026_07_06-22_35_40`) | **+0.0124** | 0.6598 |
| seed 1 (`2026_07_06-22_36_12`) | **+0.0120** | 0.6614 |

**Mean +0.0138, std 0.0023.** Seed 42 is the outlier high, and it is the seed
every headline number came from. Quote the mean. Touch-alone is far more
stable (0.6605 / 0.6598 / 0.6614), so the abstract's AUC range is unaffected.

### 2.7 Rotation share of the "articulation" target (val)

| k | median rigid share | >80% rigid |
|---|---|---|
| 2 | 0.952 | 73.1% |
| 4 | 0.957 | 74.3% |
| 8 | 0.957 | 74.5% |

`scripts/rigid_diag.py`, `results` printed to stdout only.

### 2.8 Palm-axis label validation

`scripts/validate_handframe.py` on val, n=18,864:

- grip open/close correlates with **radial** at r = +0.991
- finger fanning correlates with **spread** at r = +0.695
- fingertip-minus-knuckle geometry is 61% along **curl**

This caught a real bug: the axes were originally named (long, flex, normal)
with `flex` on the abduction axis, because `curl × radial` lies *in* the palm
plane. Rows are unchanged; only labels moved. No number was affected.

**Read the JSON's verdict key with care.** `results_handframe_validation.json`
as committed contains `"labels_behave_anatomically": false`. That is not a
failure of the corrected labels. The script's verdict tested index 1 (`flex`)
for flexion behaviour — the pre-rename hypothesis this very script refuted —
and index 1 is the abduction axis, which of course fails a flexion test. The
verdict now tests index 2 (`normal` → curl) for flexion and checks that
fanning peaks on `flex` → spread; both hold, so it reports `true`. The three
correlations quoted above were always correct and are unchanged. The stored
JSON was refreshed on 2026-07-28 and now reads
`"labels_behave_anatomically": true`, alongside the two component checks
`curl_behaves_like_a_flexion_axis` and `spread_behaves_like_an_abduction_axis`.

### 2.9 Regression / forecasting (Phase 2B) — the OLD, superseded runs

| config | k=16 MSE (moving, fingertips, articulation) |
|---|---|
| copy-zero baseline | 0.006717 |
| pose-only | 0.005574 |
| tactile + pose | 0.008694 |
| shuffled tactile | 0.009901 |

**We still have no forecaster that uses tactile and beats pose-only.** But
these runs are pre-causal-fix (§3.4), used the rotation-dominated target, and
had a confounded ablation, and §2.11 now shows touch does carry magnitude
information. So this null is a design artifact, not evidence of absence.
Rebuilding is the main open work.

### 2.10 THE DECISIVE CONTROL: touch vs full raw kinematics (k=8)

Probe given 504 dims of wrist-centred lagged pose, not the 64-dim embedding.
All fits converged (an earlier attempt hit the iteration limit on 177 of 240
fits and was discarded; see `results_rawpose_k8_UNCONVERGED_DISCARD.json`).

| axis | raw kinematics | + touch | touch adds | vs shuffled |
|---|---|---|---|---|
| radial | 0.6469 | 0.6842 | **+0.0373** [+0.0197, +0.0567] | +0.0398 |
| spread | 0.6222 | 0.6425 | +0.0203 [+0.0072, +0.0337] | +0.0232 |
| curl | 0.6200 | 0.6586 | **+0.0386** [+0.0232, +0.0544] | +0.0414 |

**Touch adds on every axis against uncompressed kinematics.** The "touch is
recovering what the bottleneck discarded" explanation is dead. Note raw
kinematics is *worse* than the learned embedding on radial and curl, so the
contrastive embedding is not merely lossy: it extracts something a linear
readout of positions cannot.

### 2.11 Magnitude (k=8) — the gate OPENED

Ridge onto the delta vector rather than its sign. Error reduction from adding
touch, clip-clustered CIs:

| comparison | all joints | fingertips |
|---|---|---|
| vs pose embedding | +3.66% [CI excludes 0] | +3.15% |
| vs raw kinematics | **+7.38%** [CI excludes 0] | **+6.79%** |
| vs shuffled twin | +7.67% | +7.08% |

**Touch carries magnitude information, not only direction.** This reverses the
expectation and means the end-to-end forecasting rebuild is worth doing. Caveat
for any claim: this is frozen features plus ridge, an easier setting than
end-to-end MSE training. It shows the information is linearly accessible, not
that a trained network will find it.

### 2.12 Robustness — all four checks passed

| check | published target | corrected target | touch alone |
|---|---|---|---|
| Participant-disjoint, clip-split encoder | −0.0051 | **+0.0165** | 0.6774 |
| Participant-disjoint, scene-trained encoder (fully clean) | +0.0008 | **+0.0225** | 0.6861 |
| Held-out TEST split, k=4 | −0.0025 | **+0.0192** | 0.6468 |
| Held-out TEST split, k=8 | −0.0012 | **+0.0199** | 0.6566 |

Test-split numbers are *higher* than val, and the fully clean participant test
is the strongest in the set. The published target sits near zero throughout.

### 2.13 Retrieval under participant-disjoint splits — DO NOT QUOTE YET

Both arms trained AND evaluated with whole scenes held out:

| split | avg-pool | biGRU | ratio |
|---|---|---|---|
| val | 5.81 | 29.09 | 5.01x |
| test | 5.29 | **28.31** | **5.36x** |

The ratio improves (2.71x to 5.36x) but **both absolute numbers collapse**:
biGRU 45.46 to 28.31, avg-pool 16.76 to 5.29. The clip-disjoint figures were
inflated by participant memorization. Not a gallery artifact: 1411 windows
versus 1399, and a smaller gallery would have raised mAP.

> **These numbers are INVALID as reported and must not go on a poster.** The
> avg-pool arm was trained with a ReLU before the projection that the upstream
> avg-pool architecture never had (see §2.15), so the baseline was handicapped
> and 5.36x is measured against too weak an opponent. That arm is being
> retrained with the corrected encoder. The biGRU side and every probe result
> are unaffected.

**The architectural claim strengthens; the absolute performance claim weakens.**
Bring this to Paul rather than letting him find it, once the corrected number
is in.

### 2.14 Subset analysis — the PI's regime hypothesis, answered

**0 of 36 pre-registered cells survive multiplicity correction**, on both the
published and corrected targets. Touch's contribution is global, not
concentrated in any contact-level, contact-transition, pose-speed or
grip-aperture regime. A cleaner result than a subset finding would have been.

### 2.15 Same-codebase retrieval, verified 2026-07-27

The headline comparison is now reproduced end to end through one eval path in
this repository:

| | this codebase, today | originally reported |
|---|---|---|
| avg-pool test | **16.76** | 16.76 |
| avg-pool val | 14.41 | 14.42 |
| biGRU test | **45.46** | 45.46 |
| biGRU val | 40.98 | 40.97 |

**16.8 → 45.5, 2.71x, same codebase, same eval path.** The cross-repo
provenance gap is closed.

Reaching it required two fixes to `PoseEncoder(temporal_mode="mean")`, both
found by trying to load the upstream checkpoint:

1. **Projection width.** The first version zero-padded the pooled 128-dim
   vector to 240 so both modes shared a projection shape. Upstream projects
   from 128, so those checkpoints failed to load on a size mismatch.
2. **The ReLU.** Upstream avg-pool projects the pooled vector directly; the
   ReLU arrived with the GRU in commit 73dc799 and was never part of that
   architecture. Mean mode had inherited it. With the ReLU the upstream
   checkpoint scores **10.25** instead of 16.76.

The second one has consequences beyond checkpoint loading: it handicapped the
overnight scene-disjoint avg-pool arm, which is why §2.13 is quarantined.

### 2.16 Forecasting rebuild — COMPLETE, 18 runs

18 runs: pose-only / tactile+pose / shuffled-tactile, k ∈ {8,16}, 3 seeds.
All methodological fixes on: `--target-mode rigid_articulation`, `--causal`
with `--sequence-length 36 --min-history 10`,
`--tactile-correction-input tactile_only` (so gate=0 removes exactly tactile),
`--grad-clip-scope per_branch` (so the pose head is clipped identically in
both arms).

Final, epoch 300, rigid target, moving subset, fingertip MSE (lower better):

| | k=8 | k=16 |
|---|---|---|
| copy-zero | 0.000312 | 0.000609 |
| **pose-only** | **0.000287** | **0.000521** |
| tactile+pose | 0.000356 | 0.000692 |
| shuffled-tactile | 0.000479 | 0.001135 |

Seed std ≤ 0.000033 throughout (the max is `rr_shuf_k16`; every other
condition is ≤ 0.000019). Gates converge to a consistent sign
(−0.019 to −0.023) rather than flipping across seeds as they did in the old
runs, which is what an unconfounded ablation should look like.

**Two things are true at once.**

1. **Real touch beats its deranged twin by 26% (k=8) and 39% (k=16)**,
   consistently at every seed. The network is genuinely extracting tactile
   information.
2. **tactile+pose still loses to pose-only** by 24% and 33%, and now also
   loses to copy-zero.

So the failure is **capacity, not information**, which matches the
frozen-feature probes exactly. There is still no forecaster that uses touch
and beats the simpler model; that claim remains unavailable.

Note also that the corrected target is a harder problem: pose-only beats
copy-zero by 17% on the old articulation target and only 8-14% here, because
removing whole-hand rotation strips out the predictable inertial component.
Read tactile-vs-shuffled, not tactile-vs-pose-only: only the shuffled twin is
capacity-matched.

### 2.17 Frozen-encoder bridge — COMPLETE. Touch beats pose-only.

The regression trained its tactile encoder from RANDOM init (~500k parameters
against a 33k pose head, ~116k samples, MSE loss), while every positive result
in this project came from that encoder FROZEN from the retrieval checkpoint.
The forecaster had never been given the encoder that demonstrably contains the
signal. `--tactile-init-checkpoint` + `--freeze-tactile-encoder` close that.

Final, epoch 300, rigid target, moving subset, fingertip MSE, 3 seeds:

| condition | trainable | k=8 | k=16 |
|---|---|---|---|
| **frozen tactile + pose** | 66,045 | **0.000255** | **0.000461** |
| pose-only | 32,957 | 0.000287 | 0.000521 |
| frozen shuffled + pose | 66,045 | 0.000297 | 0.000547 |
| copy-zero | — | 0.000312 | 0.000609 |
| fine-tuned tactile | 574,205 | 0.000341 | 0.000638 |
| scratch tactile | 574,205 | 0.000356 | 0.000692 |
| scratch shuffled | 574,205 | 0.000479 | 0.001135 |

Seed std is ~1e-6 to 2e-5; the condition clusters do not overlap.

| comparison | k=8 | k=16 |
|---|---|---|
| frozen tactile vs pose-only | **−11.2%** | **−11.4%** |
| frozen tactile vs shuffled (capacity-matched) | **−14.3%** | **−15.7%** |
| frozen shuffled vs pose-only | +3.6% | +5.1% |
| frozen tactile vs copy-zero | −18.4% | −24.2% |

**Why this is a complete argument, not one number.** The seven conditions form
a monotone ordering that is identical at both horizons and explains itself:

- `scratch shuffled` is worst: capacity cost, no information.
- `scratch tactile` beats it: information helps, but capacity still sinks it.
- `fine-tuned` beats scratch: a good initialisation helps.
- `frozen shuffled` sits just above pose-only (+3.6%/+5.1%): that is the price
  of carrying the branch at all, with no usable content.
- `frozen tactile` is best: the information, without the capacity cost.

The shuffled twin has **identical trainable parameter count** (66,045), so the
14-16% gap is tactile content and cannot be capacity, regularisation or
architecture. And `frozen shuffled` being *worse* than pose-only shows real
touch first pays back the branch's cost and then profits.

Caveats worth stating: the encoder is frozen, so this is a linear-ish readout
over pretrained features rather than a jointly optimised model; and the
retrieval checkpoint was trained on the same clips (clip-disjoint split), so a
participant-disjoint version of this experiment has not been run.

### 2.18 Scene-disjoint retrieval — CORRECTED, §2.13 superseded

Re-trained with the no-ReLU encoder (§2.15), so the avg-pool baseline is no
longer handicapped:

| split | avg-pool | biGRU | ratio |
|---|---|---|---|
| val | 7.02 | 29.09 | 4.14x |
| test | **6.44** | **28.31** | **4.40x** |

Against clip-disjoint's 2.71x (16.76 → 45.46). So the biGRU advantage *widens*
under participant hold-out while both absolute numbers fall sharply. The
clip-disjoint figures were inflated by participant memorisation; the
architectural claim survives and strengthens.

The earlier 5.36x in §2.13 came from the handicapped baseline. **Use 4.40x.**

### 2.19 Participant-disjoint forecasting — COMPLETE. Open question #1 closed.

18 runs, `sf_{frz,frzshuf,pose}_k{8,16}_s{1,2,3}`. Both the tactile encoder
(`logs/p2t_scene_gru`) and the regression split are scene-disjoint, so no val
participant appears anywhere in either training stage. `results_scene_forecast.json`,
integrity checks passed (all runs scene-split, correct encoder, motion
threshold consistent within each horizon).

| condition | k=8 | k=16 |
|---|---|---|
| **frozen tactile + pose** | **0.000270** | **0.000500** |
| pose-only | 0.000290 | 0.000515 |
| frozen shuffled + pose | 0.000317 | 0.000588 |
| copy-zero | 0.000319 | 0.000622 |

Seed sd 0.000001-0.000005, 3 seeds each.

| comparison | k=8 | k=16 | (clip-split, §2.17) |
|---|---|---|---|
| frozen tactile vs **shuffled twin** | **−15.0%** | **−15.0%** | −14.3% / −15.7% |
| frozen tactile vs pose-only | −7.0% | −3.0% | −11.2% / −11.4% |
| frozen shuffled vs pose-only | +9.4% | +14.0% | +3.6% / +5.1% |
| frozen tactile vs copy-zero | −15.5% | −19.7% | −18.4% / −24.2% |

**The load-bearing comparison did not move.** Against the capacity-matched
shuffled twin — same frozen encoder, same 66,045 trainable parameters, only
the tactile-to-pose pairing scrambled — touch is worth 15.0% at *both*
horizons, versus 14.3%/15.7% when the encoder had seen these participants.
Participant familiarity was not what produced that gap.

**What did change is the branch cost.** Carrying a frozen tactile branch with
no usable content costs +9.4% / +14.0% under participant hold-out, against
+3.6% / +5.1% before. Real touch still pays that back and profits, but the net
margin over pose-only compresses from ~11% to 7.0% (k=8) and 3.0% (k=16).
At k=16 the frz-vs-pose gap is 0.000015 against seed sds of 0.000003-0.000005.
**A clip-clustered bootstrap has since shown that this margin includes zero**
(-2.70% [-7.50, +1.82], §2.19b) and it must not be quoted. The k=8 margin
(-6.87% [-11.06, -2.67]) does exclude zero.

**Quote the shuffled comparison, not the pose-only one.** It is both the
stronger claim and the one that survived participant hold-out unchanged.

Caveat that now matters most: the scene split leaves only **3 held-out
scenes** (train 20 / val 3 / test 3 of 26). Participant generalization is
being measured over three held-out participant-locations. The effect is
consistent across seeds and horizons, but the participant base is small.

Not comparable to §2.17's absolute MSEs: the motion threshold defining the
"moving" subset is the 25th percentile of the *train* split, so it shifts
(k=8: 0.014044 → 0.013977; k=16: 0.024003). Within-sweep comparisons are
valid; cross-sweep ones are not.

### 2.19b Forecasting CIs, and the gate-zero diagnostic

`results_forecast_ci_val.json`, `scripts/forecast_ci.py`. Paired
clip-clustered bootstrap, 1000 draws, seeds averaged per sample first so the
interval describes the condition rather than one training run. Reproduces
§2.19's MSEs exactly (0.000270 / 0.000318 / 0.000290 at k=8, moving
11,060/14,554), so it is bracketing the same quantity.

| comparison | k=8 | k=16 |
|---|---|---|
| frz vs **frzshuf** | **−14.96% [−18.91, −11.24]** | **−14.86% [−19.09, −10.57]** |
| frz vs pose-only | −6.87% [−11.06, −2.67] | **−2.70% [−7.50, +1.82]** |
| frzshuf vs pose-only | +9.52% [+7.36, +11.43] | +14.27% [+10.61, +17.94] |

**The k=16 pose-only margin includes zero and must be withdrawn.** Every other
comparison excludes it. The shuffled comparison is ~15% at both horizons with
intervals nowhere near zero, which is why it, and not the pose-only margin, is
the sentence to quote.

**Gate-zero diagnostic — the "optimiser failed to switch the branch off"
hypothesis is REFUTED.** The worry was that `frzshuf` lands above pose-only
only because the optimiser never found gate≈0 (which reproduces the pose-only
head exactly), making the frz-vs-frzshuf gap partly a measure of optimiser
failure rather than tactile content. Forcing gate=0 at inference:

| condition | k=8 vs pose-only | k=16 vs pose-only |
|---|---|---|
| frz @ gate=0 | **+42.99%** | +20.70% |
| frzshuf @ gate=0 | **+27.40%** | +11.03% |
| frz vs frz @ gate=0 | −34.87% | −19.39% |

Zeroing the gate makes both arms *dramatically worse*, not better. gate≈0 is
not a better solution the optimiser missed — it is a much worse one. The pose
head in these models co-adapts to having a branch, so deleting the branch
breaks it; the branch is load-bearing even when its content is scrambled.

So `frzshuf` is not a crippled model that failed to switch off. It is a
genuine capacity-matched control that uses its branch productively for
everything *except* tactile-specific information — which is exactly what the
control is supposed to be. **This strengthens the frz-vs-frzshuf comparison
rather than qualifying it.**

The +9.5%/+14.3% branch cost still stands, and is now correctly read as:
training *with* a useless branch is worse than training *without* one
(pose-only 0.000290 < frzshuf 0.000318), but once trained with it you cannot
recover pose-only by switching it off (frzshuf@gate0 0.000370).

Gates flip sign across model seeds (+0.026 / −0.027) with consistent
magnitude. That is the expected sign symmetry — gate and branch output can
co-flip for identical products — not instability.

### 2.19c Encoder-quality ladder — hypothesis REFUTED, and a new problem

`results_encoder_ladder.json`, `scripts/encoder_ladder.sh`. Partition held
fixed at split_seed 42; only the frozen encoder varies, using intermediate
`p2t_scene_gru` checkpoints as a quality ladder. k=8, one model seed.

| encoder epoch | encoder val t2p mAP | frz vs frzshuf | frz vs pose-only |
|---|---|---|---|
| 20 | **10.81** | **−14.56%** | −8.97% |
| 40 | 16.07 | −15.06% | −8.62% |
| 60 | 20.26 | −14.06% | −7.24% |
| 100 | 25.19 | −11.04% | −5.52% |
| 300 | **28.24** | **−14.83%** | −6.90% |

**The gain is flat across a 2.6x range of encoder quality.** Pearson
r = +0.388 (n=5) — weak, and the wrong sign for the hypothesis.

**So representation quality does NOT explain the split-seed instability.** The
ladder's prediction test fails everywhere except its own endpoint:

| partition | encoder mAP | observed | ladder predicts |
|---|---|---|---|
| seed 3 | 18.74 | **+5.02%** | −14.06% |
| seed 2 | 20.78 | −10.03% | −14.06% |
| seed 1 | 24.29 | −5.77% | −11.04% |
| seed 42 | 28.24 | −14.96% | −14.83% (same partition — not an independent test) |

**Conclusion: the participant draw is what drives the instability**, not
encoder quality. The forecasting result is genuinely fragile at 26 scenes with
3 held out, and no cheap reanalysis fixes that — k-fold over scenes is the
only honest path.

> ### The new problem this raises
>
> An encoder at **mAP 10.81** — epoch 20 of 300, barely trained — delivers the
> **same ~14% advantage over its shuffled twin** as one at 28.24. And it is
> marginally *better* on frz-vs-pose-only (−8.97% vs −6.90%).
>
> That is not what "frozen retrieval features contain predictive tactile
> information" predicts. If retrieval pretraining barely matters, the
> frz-vs-frzshuf gap may not be measuring learned tactile semantics at all —
> it may be measuring nothing more than **correct temporal pairing**, which
> almost any encoder would preserve and the derangement destroys.
>
> **The decisive control is a randomly-initialised frozen encoder.** If a
> random frozen tactile branch also scores ~−14% against its shuffled twin,
> then §2.19's headline is about pairing, not representation, and the
> "frozen retrieval features" framing has to go. Two runs, ~1 hour. **Run this
> before quoting §2.19 anywhere.**
>
> **RUN. See §2.19d. It confirmed the pairing explanation.**

### 2.19d Random-encoder control — §2.19's framing does not survive

`logs/rnd_{frz,frzshuf}_k8`, `--freeze-random-tactile-encoder`. Same
partition (split_seed 42, scene-disjoint), same horizon, same 66,045 trainable
parameters, seed 1. `tactile_init_checkpoint: None` — **no pretrained weights
at all**, a randomly initialised tactile encoder simply frozen.

| condition (k=8, seed 1) | frz | frzshuf | gap | vs pose-only (0.000290) |
|---|---|---|---|---|
| **random** frozen encoder | **0.000261** | 0.000292 | **−10.62%** | **−10.00%** |
| pretrained frozen encoder | 0.000270 | 0.000316 | −14.56% | −6.90% |

**Two things, both bad for the current framing.**

1. **A never-trained encoder recovers ~71% of the effect** (10.6 of 14.96
   percentage points). So most of the frozen-vs-shuffled gap is *not* learned
   tactile semantics — it is **correct temporal pairing**, which a random
   projection preserves and the derangement destroys.
2. **The random encoder is BETTER than the pretrained one**, both absolutely
   (0.000261 vs 0.000270) and against pose-only (−10.0% vs −6.9%). Retrieval
   pretraining is not merely unnecessary here; on this evidence it is mildly
   counterproductive.

**What must change.** §2.19 cannot be described as "frozen retrieval features
contain predictive tactile information". What it supports is: *a frozen random
projection of correctly-paired tactile input improves forecasting.* The
tactile signal helps; the learned representation contributes little.

**This breaks the causal chain between the retrieval result and the
forecasting result.** They are two separate findings, not one story. Any
claim of the form "the temporal encoder drives both alignment and prediction"
is currently unsupported on the prediction half.

Caveat: n=1 seed for this control. Seed spread on the pretrained arms was
~1e-6 on means of ~3e-4, so seed noise is very unlikely to account for a
3.9-point difference, but two more seeds would settle it cheaply.

> ### The control this now demands on the PROBE side
>
> The direction probe (§2.2-§2.12, and the whole poster) also uses the frozen
> **pretrained** tactile encoder, and the analogous control has **never been
> run**. If a randomly initialised frozen tactile encoder also yields ~+0.017
> marginal AUC, then the probe result is likewise about raw tactile signal
> rather than the learned representation.
>
> The probe's *numbers* would survive either way — "touch predicts finger
> motion beyond pose kinematics" is a claim about tactile information, not
> about the encoder. What would not survive is the **hypothesis**: that
> matching temporal structure in the encoder is what unlocks predictive
> tactile information. That claim is on the poster.
>
> The probe is minutes, not hours. **Run it before the poster session.**
>
> **RUN. See §2.19e. The hypothesis is not supported.**

### 2.19e Probe random-encoder control — the HYPOTHESIS is unsupported

> **SUPERSEDED BY §2.27. Do not quote this section's conclusion.** Everything
> below is k=8 only, and k=8 turns out to be the one horizon where the
> pretrained and random encoders coincide. At k=2 and k=4 the pretrained
> encoder gives roughly twice the marginal, 7–9 encoder-seed sd outside noise.
> The numbers below are correct; the generalisation from them is not.

`results_probe_rigid_k8_RANDENC.json`, `probe_rigid.py --random-tactile-encoder`.
Tactile encoder randomly initialised and frozen; **pose encoder still the
pretrained one**, so the only thing removed is the learned tactile
representation. k=8, corrected target, clip split, n=11,425 — identical
to §2.2 in every other respect.

| | touch alone | shuffled | marginal over matched pose |
|---|---|---|---|
| pretrained encoder, seed 42 | 0.6605 | 0.5009 | **+0.0171** |
| **random encoder** | **0.6392** | 0.4865 | **+0.0122** |

Per-axis marginal, random: radial +0.0122, spread +0.0087, curl +0.0115.

**Put this next to §2.6's encoder-seed variance and the result is stark.**
Across three *pretrained* encoder seeds the marginal is +0.0171 / +0.0124 /
+0.0120 (mean +0.0138, std 0.0023). A **random** encoder gives **+0.0122** —
inside that range, and all but identical to seeds 0 and 1.

**The probe's marginal AUC is not distinguishable from what a fixed random
projection of the tactile signal achieves.** Only seed 42, already flagged as
the outlier high, exceeds it.

> **Scope this claim carefully — see §2.19f.** It holds for the marginal over
> the *learned pose embedding*, which is what this probe measures. It does NOT
> generalise: against uncompressed 504-d kinematics, and on magnitude, the
> pretrained encoder is worth about 2x a random one.

So the project's stated hypothesis — *matching temporal structure across
modalities is what unlocks predictive tactile information; an encoder that
represents motion should align better and predict* — **is not supported by
this probe.** The encoder's learned representation is not what makes touch
predictive of motion direction. Raw tactile signal is.

**What survives, and it is most of the work:**

- **The target-correction finding.** It is about the *target*, not the
  encoder: rotation is ~95% of the conventional target's energy, and the
  conventional-vs-corrected inversion (§2.2, §2.4) is reproduced by the random
  encoder too (+0.0122 corrected vs +0.0067 conventional). This is the
  centrepiece and it is untouched.
- **Touch alone decodes direction** at 0.64 (random) to 0.66 (pretrained)
  against 0.50 chance. Note the pretrained encoder *is* better here (+0.021),
  so the learned representation does help decode direction from touch in
  absolute terms — it just does not add to a pose baseline any better than
  random does.
- **Retrieval.** 2.7x clip-disjoint, 4.4x scene-disjoint, with intervals.
  Directly measured, entirely unaffected.

**What must be withdrawn or reworded:** any claim that the temporal encoder is
what unlocks *prediction*. The alignment half stands on its own retrieval
evidence; the prediction half does not follow from it.

### 2.19f Where the learned representation DOES matter — §2.19e was too sweeping

The random-encoder control has now been run against the raw-kinematics
baseline and the magnitude probe. §2.19e's conclusion holds *for the
comparison it tested* but must not be generalised: on three of five
measurements the pretrained encoder is roughly twice a random one.

| measurement | pretrained | random | representation matters? |
|---|---|---|---|
| touch-alone AUC, k=8 | 0.6605 | 0.6392 | **yes** |
| marginal vs **raw kinematics**, curl | **+0.0386** | +0.0202 | **yes, ~2x** |
| marginal vs raw kinematics, radial | +0.0373 | +0.0191 | **yes, ~2x** |
| marginal vs raw kinematics, spread | +0.0203 | +0.0081 (CI incl. 0) | **yes** |
| magnitude vs raw kinematics | +7.38% | +3.45% | **yes, ~2x** |
| marginal vs **learned pose embedding** | +0.0171 | +0.0122 | **no** (inside seed range) |
| forecasting gain | −14.96% | −10.62% (and random wins outright) | **no** |

`results_rawpose_k8_RANDENC.json`, `results_magnitude_k8_RANDENC.json`.

**The single place the representation demonstrably does nothing is the
marginal over the *learned pose embedding*** — which is precisely what §2.2,
§2.3, §2.12, Fig 3 and the poster's CONTROLS table report. Any statement of
the form "the signal is in the touch input, not the representation" must be
scoped to that comparison, or it contradicts the +0.039 and magnitude results.

Honest verdict on the project hypothesis: **mixed, not refuted.** The
forecasting evidence contradicts the prediction half outright, and the
pose-embedding marginal shows nothing; but against uncompressed kinematics,
and for magnitude, the learned encoder is worth about 2x a random projection.

### 2.19g Magnitude under participant hold-out — STRONGER, open item closed

`results_magnitude_k8_SCENE.json`. Scene-disjoint split **and**
scene-disjoint encoder (`p2t_scene_gru`), n=11,060 from 342 clips.

| comparison | clip-split | **participant-disjoint** |
|---|---|---|
| vs raw kinematics, all joints | +7.38% | **+8.46%** |
| vs raw kinematics, fingertips | +6.79% | **+7.85%** |
| vs shuffled twin | +7.67% | **+8.54%** |
| vs pose embedding | +3.66% | +4.11% |

All CIs exclude zero. **The magnitude result gets stronger under participant
hold-out**, not weaker — the opposite of what happened to forecasting.

This closes the gap flagged when §2.11 went on the poster: it was the only
displayed number lacking participant-disjoint, held-out-test and multi-seed
controls. It now has the first, which is the one that mattered. Quote the
participant-disjoint figure; quoting +7.4% means quoting the less-controlled
number.

Caveat: split and encoder both change versus the clip-split run, so this is a
stricter test rather than a clean isolation of the split effect. Single
encoder seed.

### 2.22 Why the forecaster underperforms, and three attempts to fix it

**The diagnosis first.** The 63-d delta target is close to unpredictable:
pose-only beats copy-zero by only 8-9% at k=8, and the best linear model in
this project reaches **R² = 0.19** (`results_magnitude_k8.json`,
`pose_raw_plus_tactile`). ~81% of finger articulation at 267ms is not
predictable from pose or touch.

MSE-optimal prediction is the conditional mean. On a high-entropy,
near-symmetric target that mean sits near zero — which is *why* copy-zero is
so hard to beat, and why every tactile contribution has been fighting over an
8-9% sliver. Meanwhile the direction probe sees AUC 0.66 on the same data:
touch shifts the sign odds substantially while barely moving the mean. **The
two halves of this project were never in conflict; MSE is simply the wrong
instrument for the effect.**

Three fixes were tried. Two failed.

**(a) Better fusion — REFUTED.** `results_fusion_sweep.json`, k=8,
scene-disjoint, frozen scene encoder, 3 seeds.

| fusion | frz | frzshuf | frz vs frzshuf | frz vs pose-only | branch cost |
|---|---|---|---|---|---|
| gate | **0.000270** | 0.000318 | −14.90% | **−6.78%** | +9.54% |
| film | 0.000276 | 0.000344 | −19.69% | −4.83% | +18.51% |

FiLM (`h ← (1+Δγ(tactile))·h + β(tactile)`) is worse absolutely and worse
against pose-only. The gate arm reproduces §2.19 (−14.90% vs −14.96%), so the
sweep is sound.

> **A trap worth remembering.** FiLM's *capacity-matched* contrast looks
> better (−19.69% vs −14.90%) — the very number this file elsewhere says to
> lead with. But it got there by making the shuffled control **worse** (branch
> cost +18.51% vs +9.54%), not the real arm better. A more expressive fusion
> gives the model more ways to hurt itself with garbage input. Read
> frz-vs-frzshuf together with frzshuf-vs-pose, never alone.

**(b) Motion onset — NULL.** `results_onset_sweep.json`. Given a hand
currently still, will it move over the next k? n=3,077, base rate 0.463.

| arm | AUC | seed sd |
|---|---|---|
| frozen touch | 0.5234 | 0.0029 |
| frozen shuffled | 0.5121 | 0.0221 |
| pose-only | 0.5117 | 0.0124 |

All three at chance. Touch's +0.0117 is inside one seed sd of the controls.
**Pose-only is also at chance**, so this is not "touch failed to add" — it is
that nothing predicts motion initiation at this horizon in this data. The
hypothesis (contact forces precede visible motion) was reasonable and is
simply wrong here. The task was well-posed — symmetric k-frame windows, near
balanced base rate — so this is a real negative, not a measurement failure.

**(c) Grip aperture — REAL SIGNAL.** `results_aperture_sweep.json`. Target is
the change in mean fingertip-to-wrist distance: one scalar a policy acts on,
and **rotation invariant by construction**, so it needs none of the Kabsch
correction the delta targets require.

| arm | MSE | R² vs zero | AUC on sign |
|---|---|---|---|
| **frozen touch + pose** | **0.000115** | **0.155** | **0.645** |
| pose-only | 0.000120 | 0.116 | 0.589 |
| frozen shuffled | 0.000138 | **−0.012** | 0.545 |
| copy-zero | 0.000136 | 0.000 | — |

Touch beats pose-only: R² 0.116 → 0.155 (+33% relative), **AUC +0.056**. The
shuffled arm has *negative* R² — actively worse than predicting no change.

**But the 300-epoch protocol was hiding most of it.** Dense validation (every
2 epochs, 3 seeds) shows the arms have opposite dynamics:

| epoch | frozen touch | shuffled | pose-only |
|---|---|---|---|
| 2 | **0.213** / 0.692 | 0.007 / 0.517 | 0.002 / 0.520 |
| **4** | **0.248** / 0.678 | 0.055 / 0.541 | 0.054 / 0.544 |
| 60 | 0.163 / 0.654 | −0.006 / 0.558 | 0.107 / 0.577 |
| 300 | 0.155 / 0.645 | −0.012 / 0.545 | 0.116 / 0.589 |

Touch reaches R²=0.21 by epoch **2** and peaks at epoch 4, then decays —
consistent with the effective-sample-size problem (windows overlap 19 of 20
frames, so 116k samples are nowhere near 116k independent ones). Pose-only
starts at zero and is still climbing at 60. Comparing them at one shared epoch
mis-states both: the gap is 4.6x at epoch 4 and 1.3x by epoch 300.

**Do not quote epoch 4 from the table above.** That peak was selected on val
and those metrics are also val — selection on the eval set, the same error as
quoting seed 42. `scripts/aperture_earlystop.sh` selects the epoch on val and
reports on TEST; use its numbers.

### 2.23 Aperture under redrawn partitions — IT REPLICATES

`results_aperture_earlystop.json`, `results_aperture_splitseed_ss{1,2,3}.json`.
Four participant partitions, each with its OWN scene-disjoint encoder
(`p2t_scene_gru`, `_ss1`, `_ss2`, `_ss3`). Stopping epoch chosen on val by R²,
number reported on TEST. 2 model seeds per cell.

| partition | touch AUC | pose-only AUC | shuffled AUC |
|---|---|---|---|
| 42 (original) | 0.665 | 0.601 | 0.517 |
| 1 | 0.610 | 0.608 | 0.536 |
| 2 | 0.657 | 0.581 | 0.558 |
| 3 | 0.671 | 0.599 | 0.551 |

| contrast | mean | range | positive |
|---|---|---|---|
| touch − **shuffled** (capacity-matched), AUC | **+0.110** | [+0.074, +0.148] | **4/4** |
| touch − shuffled, R² | +0.132 | [+0.067, +0.190] | **4/4** |
| touch − pose-only, AUC | +0.054 | [+0.002, +0.076] | 4/4 |
| touch − pose-only, R² | +0.058 | [−0.010, +0.103] | 3/4 |

**This is the first result in the forecasting line that survives the test that
killed §2.19.** For contrast, the delta target's capacity-matched contrast
across redrawn partitions went −15.0%, −5.8%, −10.0%, **+5.0%** — a sign flip
— and its pose-only contrast turned negative in all six redrawn cells. The
aperture gap never flips, never approaches zero, and never drops below
+0.074 AUC.

Partition 1 is the weak one: touch beats pose-only by only +0.002 AUC and
−0.010 R². Pose-only had its best test performance of any partition there
(R² 0.119) while touch had its worst (0.109) — those three held-out scenes
suit the pose baseline. The capacity-matched contrast still holds cleanly
(+0.074 AUC).

**The learning-dynamics finding is fully robust.** Touch selected epoch 2-4 in
every run across all four partitions; pose-only selected 36-60 in every run.
That is a property of the data, not of one split.

**What can be said:** on predicting whether the hand is about to open or close
— a rotation-invariant, policy-relevant scalar — frozen tactile features beat
a capacity-matched control by +0.11 AUC on average across four participant
partitions, positive in all four, and beat a pose-only model by +0.054 AUC,
out of sample on participants held out of every training stage.

**Limits:** 3 held-out scenes per partition, 2 model seeds per cell.

### 2.24 Aperture at k=16 — the effect is present at 533ms too

> **SUPERSEDED IN PART BY §2.26.** The numbers below are correct. The
> *interpretation* — that the effect strengthens with horizon — was inferred
> from two horizons and does not hold once k=2 and k=4 are filled in: the
> pose-only margin is largest at k=2 and smallest at k=8. Read §2.26 before
> quoting anything directional from this section.

`results_aperture_k16_ss{42,1,2,3}.json`. Identical protocol to §2.23, so the
horizons are directly comparable.

| partition | k=8 touch/pose/shuf AUC | k=16 touch/pose/shuf AUC |
|---|---|---|
| 42 | 0.665 / 0.601 / 0.517 | **0.721** / 0.652 / 0.518 |
| 1 | 0.610 / 0.608 / 0.536 | 0.673 / 0.644 / 0.574 |
| 2 | 0.657 / 0.581 / 0.558 | 0.693 / 0.573 / 0.548 |
| 3 | 0.671 / 0.599 / 0.551 | 0.712 / 0.653 / 0.585 |

| contrast (AUC) | k=8 | k=16 |
|---|---|---|
| touch − shuffled (capacity-matched) | +0.110 [+0.074, +0.148] · 4/4 | **+0.143** [+0.099, +0.203] · **4/4** |
| touch − pose-only | +0.054 [+0.002, +0.076] · 4/4 | **+0.069** [+0.029, +0.120] · **4/4** |

Touch-alone AUC rises 0.651 → 0.700. **Every comparison is larger at 533ms
than at 267ms**, and both replicate on all four partitions.

Three points worth keeping:

- **The weak partition firms up.** Partition 1's pose-only margin was +0.002
  at k=8 (a wash); at k=16 it is +0.029. The one soft spot in §2.23 improves
  rather than degrades.
- **Epoch selection has now reproduced 16/16.** Touch selects epochs 2-8 in
  every run; pose-only selects 30-60 in every run, across 2 horizons x 4
  partitions x 2 seeds, no exceptions.
- **k=16 should be harder** — ~40% fewer samples (11 valid timesteps per
  window rather than 19) and a more distant target. Every arm improving, with
  touch improving most, points at longer-range structure rather than immediate
  contact dynamics.

**Interpretation:** touch is not reporting the instant of contact; it carries
information about where the grip is *heading*. That is the useful reading for
a policy, which needs lead time, and it is the opposite of the delta target's
behaviour, where the pose-only margin vanished at k=16 and the interval
crossed zero (§2.19b).

### 2.29 Is the rotation share an artifact of still frames? No — the opposite

`results/results_motion_sensitivity_{opentouch,dexycb}.json`,
`scripts/rotation_share_motion_sensitivity.py`.

**The objection.** `rotation_share.py` applies no minimum-motion threshold. For
a hand that barely moves between t and t+k, the reported fraction is a ratio of
two tiny numbers dominated by annotation noise. If those samples drove the
headline median, the claim would be an artifact of stillness. This is the first
thing a reviewer will ask.

**The answer: the rigid share RISES with motion magnitude on both datasets, so
the still frames are dragging the published number DOWN.** Every motion filter
makes the claim stronger, not weaker.

| filter | OpenTouch k=8 | DexYCB k=8 |
|---|---|---|
| all samples (**the published figure**) | 96.1% / mean 85.3% | 89.6% / mean 81.6% |
| above median motion | **98.0%** / mean 89.4% | **94.0%** / mean 88.4% |
| top decile of motion | **99.4%** / mean 96.4% | **97.1%** / mean 94.3% |

By motion decile, stillest to largest:

| dataset, k | stillest decile | largest-motion decile |
|---|---|---|
| OpenTouch, k=2 | 83.3% | 99.0% |
| OpenTouch, k=8 | 90.6% | 99.4% |
| DexYCB, k=2 | 52.9% | 93.2% |
| DexYCB, k=8 | 64.8% | 97.1% |

The trend is monotone across all ten deciles in three of the four cases (the
exception is DexYCB k=8, where deciles 3 and 4 invert by 2.3 points before the
climb resumes).

**This also explains the median-vs-mean gap.** The published means (85.3%,
81.6%) sit well below the medians because the distribution has a left tail —
and that tail *is* the near-still, noise-dominated samples. Filter to the top
motion decile and the mean rises to 96.4% and 94.3%, nearly meeting the median.
The skew was never evidence of a fragile effect; it was the stillness tail.

**Why still frames score low, verified rather than assumed.** The metric was
checked on synthetic input: a pure rigid rotation scores **100.0%**, and
isotropic per-joint noise with no rigid component scores **22.4%**. Annotation
noise is badly explained by a single whole-hand rotation, so noise-dominated
samples land near the bottom.

**What to say in the paper.** Report the all-samples figure as the headline —
it is the conservative one — and state that restricting to above-median motion
raises it to 98.0% (OpenTouch) and 94.0% (DexYCB). Pre-empting this beats
having it raised in review, and the answer runs in your favour.

### 2.31 The delta-MSE instrument cannot answer the random-encoder question

`results/results_randenc_horizons.json`, `scripts/randenc_horizons.sh`.
10 runs, 0 errors, completed 2026-08-03 17:12. Protocol copied from
`logs/rnd_frz_k8/params.txt`; the collector was validated against §2.19d's
three published figures (0.000261 / 0.000292 / 0.000270) before the sweep ran.

**The question.** §2.19d says a randomly-initialised frozen tactile encoder
*beats* the pretrained one on forecasting — the second leg of contribution 2.
It is k=8, one seed. Does it hold at other horizons?

**The answer is that this instrument cannot say.** The two contrasts disagree:

| k | capacity-matched (vs own shuffled) | vs pose-only |
|---|---|---|
| 2 | pre −15.22% / rnd −6.98% → **pretrained** | pre −9.30% / rnd −6.98% → **pretrained** |
| 4 | pre −16.67% / rnd −12.98% → **pretrained** | pre −12.88% / rnd −13.64% → *random* |
| 8 | pre −14.56% / rnd −10.62% → **pretrained** | pre −6.90% / rnd −10.00% → *random* |

§2.19d read the right-hand column. The left-hand column says the opposite at
every horizon, including k=8.

**Why they disagree — and it is the trap §2.22 already records.** The two
shuffled controls are not equivalent:

| k | pretrained shuffled vs pose-only | random shuffled vs pose-only |
|---|---|---|
| 2 | **+6.98%** | 0.00% |
| 4 | **+4.55%** | −0.76% |
| 8 | **+8.97%** | +0.69% |

Deranging a *pretrained* encoder's input costs 4.6–9.0% against pose-only;
deranging a *random* encoder's input costs nothing. So the pretrained arm's
capacity-matched gap is partly its own control being bad, not its real arm
being good. §2.22's rule is *"read frz-vs-frzshuf together with
frzshuf-vs-pose, never alone"* — that rule invalidates the comparison
§2.19d is built on.

**Precision compounds it.** The logs print 6 decimals. At k=2 the MSEs are
~4e-5, i.e. two significant figures:

| k | random vs pretrained, absolute |
|---|---|
| 2 | resolved — pretrained better |
| 4 | **inside rounding, not resolvable** |
| 8 | resolved — random better |

**Conclusion.** Neither §2.19d's claim nor its reversal is supported. This is
what a low-power instrument on a near-unpredictable target looks like:
contradictory contrasts, differences inside logging precision, and controls
that behave differently between arms. **Do not cite §2.19d, and do not cite
this section as a reversal of it.** The honest statement is that the
delta-forecasting line cannot resolve whether the learned representation
matters, and contribution 2 must not be founded on it.

> **This does not touch §2.27.** That is the *probe*, scored by AUC, three
> encoder seeds, one consistent contrast. Its finding — the learned
> representation contributes ~55% of the marginal at k=2/k=4 and washes out by
> k=8 — stands on its own evidence. It simply has no MSE corroboration,
> because there is nothing coherent in the MSE line to corroborate with.

**What replaces it.** `scripts/aperture_randenc.sh` runs the same control on
grip aperture — AUC and R² on a scalar with a real sign, four horizons, four
partitions, each encoder with its own shuffled control. That is the test that
can settle contribution 2.

### 2.30 HO-3D — third dataset, and it is the weakest of the three

`results/results_rotation_share_ho3d_clean.json`,
`results/results_motion_sensitivity_ho3d.json`. Read §2.28 first.

**HO-3D is far less data than its "55 training sequences" suggests.** After
two guards, both added because the first HO-3D run looked entirely reasonable
and was not:

| stage | sequences | takes |
|---|---|---|
| as loaded | 235 | 55 "recordings" |
| after dropping no-articulation annotations | — | 40 |
| after collapsing duplicate camera views | **44** | **9** |

1. **Five-way camera duplication.** HO-3D encodes the camera as the *last
   character of the sequence name* — `ABF10`–`ABF14` are five views of one
   take. Caught because the by-group table showed five entries at a time with
   identical n and identical median, which is what a rigid-invariant metric
   does to duplicate views. A name-based rule would not have been safe either:
   `MC1`–`MC6` share a prefix but are six genuinely different takes. Dedup is
   done on content.
2. **Six families have no articulation at all.** MC, ND, SM, SMu1, SS, SiS —
   15 of 55 recordings — have **exactly zero** variation in intra-hand joint
   distances across their entire length. The annotation is one frozen hand
   template re-posed rigidly. They score exactly 1.0 because 100% of their
   wrist-relative motion really is rigid rotation — a fact about the
   annotation pipeline, not about hands.

Including them inflated the result by ~5 points at every horizon:

| k | contaminated | **cleaned** |
|---|---|---|
| 2 | 80.1% | **74.3%** |
| 4 | 83.6% | **78.2%** |
| 8 | 86.8% | **82.1%** |
| 16 | 89.8% | **85.7%** |

**All three datasets, k=8, cleaned:**

| dataset | median rigid share | independent units |
|---|---|---|
| OpenTouch | **96.1%** | 26 scenes |
| DexYCB | **89.6%** | 300 takes, 3 subjects |
| HO-3D | **82.1%** | 9 takes |

**The ordering is exactly what §2.28's transport-vs-articulation reading
predicts.** OpenTouch is egocentric free-living, the hand carried around by a
walking body — highest. DexYCB is seated tabletop grasping — middle. HO-3D is
hand-object manipulation with a largely stationary arm and deliberate in-hand
articulation — lowest. The prediction was made before HO-3D was measured.

**Motion sensitivity holds here too** (§2.29): HO-3D k=8 goes 82.1% over all
samples → 91.5% above median motion → 93.8% in the top decile. Still frames
still drag the number down, so the published figure remains the conservative
one on all three datasets.

**Between-take spread is wide** — at k=2 the nine takes run 49.0% to 100.0%,
median across takes 69.2%. With only nine independent takes the pooled figure
is sensitive to take length, so quote the pooled number *and* the range.
`SMu4` sits at 1.0000: its shape does change (no frame pair is shape-frozen),
but articulation is negligible beside its rotation. It is real data at the
boundary, not an artifact, and it is one take of nine.

**How to state the claim across all three.** *The standard wrist-relative
target is majority whole-hand rotation on every dataset measured — 82%, 90%
and 96% of its energy at 267 ms — with the magnitude tracking how much the
hand is transported versus articulated in place.* That is weaker than "~95%"
and much harder to dismiss.

### 2.28 CROSS-DATASET: the rotation claim replicates on DexYCB

`results/results_rotation_share_dexycb3.json` (per subject),
`_dexycb3_bytake.json` (per take), `_dexycb.json` (subject-01 alone).
2026-08-03. DexYCB **subjects 01–03, 300 takes**, one camera per take, labels
only — no image was ever written to disk. 20,278 frames after dropping
unannotated ones and splitting sequences at every gap.

**This is the first evidence for the central claim from data this project did
not collect.** DexYCB is NVIDIA's, recorded with a fixed multi-camera RealSense
rig for tabletop grasping, annotated by a MANO fit — different lab, hardware,
task and annotation pipeline from OpenTouch's egocentric free-living capture.

| k | ms | DexYCB median rigid share | OpenTouch median |
|---|---|---|---|
| 2 | 67 | **81.5%** | 95.7% |
| 4 | 133 | 85.8% | 96.1% |
| 8 | 267 | **89.6%** | 96.1% |
| 16 | 533 | 93.2% | 95.6% |

**The claim holds: the standard wrist-relative "articulation" target is
dominated by whole-hand rotation on both datasets.** At k=8, 89.6% of DexYCB's
target energy is rotation the target was never meant to contain. Across all 300
takes at k=8, **261/300 exceed 80%** and **292/300 exceed 70%**, range
53.7%–98.5%.

**Adding two more people barely moved it**, which is the strongest single
argument that this is not a subject quirk:

| k | subject-01 alone | all three subjects |
|---|---|---|
| 2 | 81.4% | **81.5%** |
| 4 | 86.1% | 85.8% |
| 8 | 89.8% | **89.6%** |
| 16 | 92.6% | 93.2% |

Between-person variation is real but small against the effect — at k=8 the
three subjects sit at 89.8% / 92.0% / 85.7%, and at k=2 at 81.4% / 85.6% /
75.7%. **The lowest single person at the most articulation-favourable horizon
is still 75.7%.**

**But the magnitude is dataset-dependent, and the "~95%" figure is
OpenTouch-specific.** Two differences worth stating plainly rather than
burying:

1. **DexYCB is lower** — 89.8% vs 96.1% at k=8.
2. **DexYCB rises with horizon** (81.4% → 92.6%) where OpenTouch is flat
   (~96% at every horizon).

Both follow from what the hands are doing. DexYCB is seated tabletop grasping:
the hand is roughly stationary and deliberately articulating, so true finger
motion is a larger share of the target, and it dominates most at short
horizons. OpenTouch is egocentric free-living — the hand is being carried
around by a walking body through grocery aisles and hardware stores, so
whole-hand transport dominates at every horizon. **The rigid share tracks how
much the hand is being transported versus articulated in place.**

That is a better result than two identical numbers would have been. It says
the effect is not a quirk of one capture rig, it is a property of hand motion
whose size varies with activity — and even in the most articulation-favourable
case measured (tabletop grasping at 67 ms) **81% of the standard target is
still rotation**.

**How to state the claim now.** Not "the target is ~95% rotation" — that is
OpenTouch's number. Say: *the standard wrist-relative target is dominated by
whole-hand rotation, 81–96% of its energy depending on dataset and horizon,
measured on two independent datasets.* The correction argument is unchanged and
is now much harder to dismiss as a property of one capture setup.

**Limits.** Three of DexYCB's ten subjects. Adding subjects 02 and 03 changed
the pooled median by ≤0.6 points at every horizon, so the remaining seven are
unlikely to move it, but they are available if a reviewer asks (~12 GB of
download each, ~250 MB of labels kept; see `scripts/load_public_hands.py` and
the fetch scripts under `/scratch/bashar/datasets/dexycb/`). One camera per
take by design: all 8 views of a grasp are related by a rigid transform, so
their rigid shares are identical and including them would multiply apparent n
by 8 while adding no information.

> **The joint-order guard earned its keep here.** DexYCB's `joint_3d` arrives
> in MediaPipe order, not MANO, because `manopth` reorders internally before
> dex-ycb-toolkit ever sees it. Loading it under the MANO assumption scores
> **16.5%** on the anatomical check against **88.4%** for the correct one — and
> would have produced a plausible-looking rotation share computed over an
> arbitrary subset of samples. See `scripts/load_public_hands.py`.

### 2.27 Probe paper sweep — three gaps closed, and §2.19e was a k=8 artifact

12 runs, completed 2026-08-03 03:25, 0 errors.
`results/results_probe_rigid_k{2,4,16}_{seed0,seed1,RANDENC,SCENE_CLEANENC}.json`.
Every figure below re-derived from the JSONs.

**Gap 1 — the headline table is now a three-encoder mean at every horizon**,
not seed 42 alone. Marginal over matched pose, corrected target:

| k | seed 42 | seed 0 | seed 1 | **mean** | encoder-seed sd |
|---|---|---|---|---|---|
| 2 | +0.0168 | +0.0177 | +0.0151 | **+0.0165** | 0.0011 |
| 4 | +0.0164 | +0.0171 | +0.0143 | **+0.0159** | 0.0012 |
| 8 | +0.0171 | +0.0124 | +0.0120 | **+0.0138** | 0.0023 |
| 16 | +0.0132 | +0.0043 | +0.0051 | **+0.0075** | 0.0040 |

Two things fall out. **The marginal declines monotonically with horizon** —
touch adds most at 67 ms. And **seed 42 is not uniformly "the outlier high"**
as §2.6 has it: at k=2 and k=4 seed 0 is highest and seed 42 sits in the
middle. Seed 42 runs high only at k=8 and k=16. The encoder-seed sd also grows
4x with horizon (0.0011 → 0.0040), so at k=16 the spread is over half the
mean — independent support for §5's "do not publish the k=16 margin".

**Gap 2 — the random-encoder control, at every horizon. This is the important
one, and it overturns §2.19e's conclusion.**

| k | pretrained (3-enc mean) | random | gap, in encoder-seed sd | random recovers |
|---|---|---|---|---|
| 2 | +0.0165 | +0.0073 | **8.6 sd** | 44% |
| 4 | +0.0159 | +0.0074 | **7.0 sd** | 47% |
| 8 | +0.0138 | +0.0122 | 0.7 sd | 88% |
| 16 | +0.0075 | +0.0118 | −1.1 sd | 157% |

§2.19e tested **only k=8** and concluded "the probe's marginal AUC is not
distinguishable from what a fixed random projection achieves", and from that,
that the learned representation is not what makes touch predictive. **k=8 is
the single horizon where those two quantities happen to coincide.** At k=2 and
k=4 — where the probe is strongest — the pretrained encoder delivers roughly
**twice** the marginal of a random one, and the gap is 7–9 encoder-seed
standard deviations wide. The same ordering shows in touch-alone AUC:

| k | touch alone, pretrained | touch alone, random |
|---|---|---|
| 2 | 0.6490 (sd 0.0013) | 0.6083 |
| 4 | 0.6518 (sd 0.0010) | 0.6140 |
| 8 | 0.6606 (sd 0.0006) | 0.6392 |
| 16 | 0.6860 (sd 0.0028) | 0.6729 |

**What this means for the paper.** §5 of `SESSION_HANDOFF.md` says the learned
representation "is not what drives the headline probe marginal", and calls that
fatal at a venue expecting method novelty. **That assessment rests on §2.19e
and should be revised**: at the two shortest horizons the learned
representation accounts for about 55% of the marginal, by a margin far outside
encoder-seed noise. The honest statement is now horizon-dependent — the
learned representation matters at short horizons and washes out by 267 ms —
which is a more interesting finding than either "it matters" or "it doesn't".

> **Caveat, and it is the same one as everywhere else here.** The random
> encoder is **one seed per horizon**. The k=2 and k=4 results agree closely
> with each other (44% and 47%), which is reassuring, but neither is a tested
> difference. Two more random seeds at k=2 would settle it and cost ~30 GPU-min.

**Gap 3 — participant-disjoint holds at every horizon**, with a scene-disjoint
encoder and a scene-disjoint split (both harder than the clip split):

| k | marginal | touch alone | shuffled |
|---|---|---|---|
| 2 | +0.0168 | 0.6627 | 0.5011 |
| 4 | +0.0181 | 0.6724 | 0.4995 |
| 8 | **+0.0225** | 0.6861 | 0.4913 |
| 16 | +0.0171 | 0.7151 | 0.4923 |

Positive at all four, and **larger than the clip-split marginal** at k=8
(+0.0225 vs +0.0138) and k=16 (+0.0171 vs +0.0075). The effect does not depend
on participants being shared between train and test — it is, if anything,
cleaner when they are not. Shuffled controls sit on 0.50 throughout.

**The centrepiece survives as a three-encoder mean at four horizons.**
Conventional vs corrected target, marginal over matched pose:

| k | conventional | corrected |
|---|---|---|
| 2 | +0.0042 | **+0.0165** |
| 4 | +0.0048 | **+0.0159** |
| 8 | +0.0038 | **+0.0138** |
| 16 | +0.0027 | **+0.0075** |

Corrected beats conventional at every horizon, ~4x at k=2/4/8. This is the
paper's main claim and it is no longer a single-seed, single-horizon result.

### 2.26 The full horizon curve — §2.24's "strengthens with horizon" is WRONG

`results/results_aperture_k{2,4}_ss{42,1,2,3}.json`, completed 2026-08-03
01:54. Same protocol as §2.23/§2.24 (epoch chosen on val by R², scored on
test, four participant partitions), now at k=2 and k=4 so all four horizons
are comparable.

**§2.24 saw only k=8 and k=16, read the rise between them as a trend, and
titled itself "the effect STRENGTHENS with horizon". With k=2 and k=4 filled
in, that is not what the data does.**

| k | ms | touch − shuffled (capacity-matched) | touch − pose-only |
|---|---|---|---|
| 2 | 67 | +0.112 [+0.095, +0.129] · 4/4 | **+0.088** [+0.069, +0.109] · 4/4 |
| 4 | 133 | +0.114 [+0.062, +0.144] · 4/4 | +0.072 [+0.023, +0.096] · 4/4 |
| 8 | 267 | +0.110 [+0.074, +0.148] · 4/4 | **+0.054** [+0.002, +0.076] · 4/4 |
| 16 | 533 | **+0.143** [+0.099, +0.203] · 4/4 | +0.069 [+0.029, +0.120] · 4/4 |

**The pose-only margin is LARGEST at the shortest horizon and smallest at
k=8.** The k=8 → k=16 rise §2.24 leaned on is a partial recovery from a dip,
not the continuation of a trend. This is not a one-partition artifact: the k=2
margin exceeds the k=8 margin in **4/4 partitions** (+0.110 vs +0.064, +0.069
vs +0.003, +0.091 vs +0.076, +0.083 vs +0.072).

**Why the margin narrows while touch keeps improving.** Both arms get better
with horizon; pose-only just improves faster over the middle of the range:

| arm, mean over 4 partitions | k=2 | k=4 | k=8 | k=16 | Δ |
|---|---|---|---|---|---|
| touch | 0.631 | 0.641 | 0.651 | 0.700 | +0.069 |
| pose-only | 0.543 | 0.569 | 0.597 | 0.630 | **+0.087** |

So "touch carries information about where the grip is heading" (§2.24's
reading) survives in absolute terms — touch-alone AUC does rise monotonically
across all four horizons. What does not survive is the claim that touch's
*advantage* grows with lead time. Against a capacity-matched control the
advantage is flat (+0.110 to +0.114) from 67 ms to 267 ms and only rises at
533 ms; against pose-only it shrinks then partially recovers.

**The capacity-matched contrast is the robust one.** It is positive in 16/16
partition-horizon cells, never below +0.062 in any single cell, and varies
little across three of the four horizons.

**Epoch selection now reproduces 25/25.** Touch selects epochs 2–12, pose-only
selects 30–60, and touch selects strictly earlier than pose-only in **every one
of 25 runs across four horizons and four partitions**, no exceptions. This is
the most reproducible thing in the forecasting line.

> **Limit, and it matters for how hard to push this.** k=2 and k=4 are **one
> model seed per cell**; k=8 has 2–3 and k=16 has 2. The horizon *ordering*
> above rests on single-seed estimates at the two new horizons. The 4/4
> partition agreement makes a pure noise explanation unlikely, but a second
> seed at k=2 and k=4 would settle it and is ~2 GPU-hours. **Do not put the
> non-monotonic shape in a paper on one seed** — either run the seeds or report
> only the capacity-matched contrast, which is flat and does not depend on the
> shape.

**What to change in §2.24:** its table and numbers are correct; its title and
its closing interpretation are not. Reword to "the effect is present at every
horizon from 67 ms to 533 ms" and drop "every comparison is larger at 533 ms",
which correction 9 already flagged as false for R² and which this section now
shows is false for AUC as well once k=2 and k=4 exist.

### 2.25 Rotation share: the diagnostic validates, and it holds in all 26 scenes

`results/results_rotation_share_opentouch.json`,
`results/results_rigid_diag_val.json`. Both sides of this comparison now have
a JSON; §2.7's 0.957 previously existed only as stdout in a doc.

**Validation.** `scripts/rotation_share.py` is the dataset-agnostic path meant
to run on public datasets. It reproduces this project's own figure:

| path | split / seq len | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|
| `rigid_diag.py` (reference) | val, 20 | 0.9523 | 0.9569 | **0.9572** | 0.9555 |
| `rotation_share.py` (agnostic) | train+val+test, 36 | 0.9572 | 0.9612 | **0.9608** | 0.9559 |

0.004 apart at k=8. **The agnostic path is trustworthy; cross-dataset numbers
from it can be believed.**

> **Two caveats on what that validation actually buys.** First, the two runs
> are not apples-to-apples — different split and different window length (the
> table states both). They agree anyway. Second, and more limiting: both
> scripts import the *same* `rigid_fraction` from
> `opentouch/articulation_frames.py`, so they cannot disagree on the
> mathematics. This validates the data-loading path only. The maths is covered
> by `tests/test_articulation_frames.py`, not by this comparison.

**Generalization across scenes — the part worth putting in the paper.** The
share broken down over 26 recording sessions, k=2, 7,620 windows:

| | median rigid share |
|---|---|
| highest scene (`eat_ygf_p2`) | 99.3% |
| lowest scene (`fablab_ml_p1`) | **90.5%** |
| all 26 scenes | **every one above 90%** |

Spanning grocery aisles, an office, a fab lab, home kitchens, hardware stores,
sports retail and eating. This is the difference between "we measured this
once" and "this holds across every activity we have" — and it cost no GPU
time. It is not a substitute for a second dataset, but it is the strongest
form of the claim currently available.

**The share is also flat in horizon** (0.957 / 0.961 / 0.961 / 0.956 at
k=2/4/8/16), so this is not an artifact of one prediction distance.

### 2.20 Retrieval bootstrap CIs — open question #2 closed

Clip-clustered, 1000 draws, fixed gallery (queries resampled only). T→P mAP:

| checkpoint | split | mAP | 95% CI | n windows / clips |
|---|---|---|---|---|
| biGRU, clip-disjoint | test | 45.50 | [42.09, 48.65] | 1399 / 296 |
| biGRU, clip-disjoint | val | 41.02 | [37.91, 44.13] | 1572 / 296 |
| **biGRU, scene-disjoint** | test | **28.31** | [25.59, 31.01] | 1411 / 257 |
| **avg-pool, scene-disjoint** | test | **6.46** | [5.45, 7.59] | 1411 / 257 |
| biGRU, scene-disjoint | val | 29.09 | [26.84, 31.49] | 1530 / 345 |
| avg-pool, scene-disjoint | val | 7.03 | [6.12, 7.97] | 1530 / 345 |

These reproduce §2.1 and §2.18's point estimates, so the eval path is
consistent. **The scene-disjoint architectural claim is now interval-backed:
[25.59, 31.01] against [5.45, 7.59] on test — nowhere near overlapping.** The
4.40x ratio is safe to quote.

> **Trap, hit and fixed on 2026-07-28.** The first run of these produced
> `scene_gru` at **72.09** mAP. No retrieval checkpoint in this project
> records `split_group_by` in its metadata — the field postdates all of them —
> so `bootstrap_eval.py`'s "default to whatever the checkpoint recorded" path
> fell through to `clip` and handed the scene-trained model a gallery full of
> its own training participants. It produced a plausible-looking number that
> was wrong by 2.5x. The fallback now warns loudly and records
> `split_group_by_source` in the output JSON, and `scripts/post_sweep.sh`
> states the split explicitly per checkpoint. **If you see a scene checkpoint
> scoring above ~30, check that field first.**

### 2.21 Per-joint export at k=16 — open question #3 closed, with a caveat

`per_joint_k16.{json,csv}`. n=6626 moving samples from 288 clips.

| axis | k=8 marginal | k=16 marginal | k=16 CI | k=16 joints CI>0 |
|---|---|---|---|---|
| curl | +0.0187 | **+0.0198** | **[+0.0036, +0.0360]** | 14/20 |
| radial | +0.0171 | +0.0132 | [−0.0046, +0.0320] | 14/20 |
| spread | +0.0126 | +0.0114 | [−0.0049, +0.0274] | 6/20 |

**At k=16 only curl's interval excludes zero.** k=16 has 6626 samples from 288
clips against k=8's 11,425 from 295, so the intervals are wider. This is the
first horizon where the omnibus claim does not hold for every axis, and §2.3's
CIs were only ever computed at k=8.

This cuts both ways for the curl story. §2.2 says curl's *ordering* advantage
is not separated and should not be defended — that still stands at k=8. But
curl is the only axis that survives at the longest horizon, which is a real
asymmetry rather than a ranking artifact. Say "curl is the only axis whose
k=16 interval excludes zero", not "curl is the strongest axis".

---

## 3. Corrections to earlier claims (all verified)

1. **Retrieval baseline** was paper-vs-own. Correct figure 16.76 → 45.46.
2. **Axis labels** were wrong; `flex` was on the abduction axis. Fixed.
3. **"Redundant with pose kinematics"** withdrawn; artifact of the target.
4. **k=16 regression runs are noncausal.** `params.txt` shows
   `sequence_length: 20`, no `causal`/`min_history` field, at
   `git_commit: 69a5523`, which predates the causal fix `9bb8a6f`. Tactile
   saw frames including the target's own timeframe.
5. **"Co-training damage"** is not supported: `pose_regression.py:236` feeds
   the full 63-dim pose into the correction branch, so zeroing the gate
   ablates pose capacity, not tactile.
6. **"Active tactile harm"** is capacity cost; the shuffled control is *worse*
   than real tactile (0.009901 vs 0.008694).
7. **Anatomical-prior negative result is invalid.**
   `tactile_contact_encoder.py:281` mean-pools over the 21 joint queries after
   a shared projection, making the encoder provably invariant to joint
   relabeling (verified to 5e-07). Anatomy could not have helped by
   construction. Recommend removing this result entirely.
8. **"curl grows monotonically with horizon"** is false (dips at k=4).
9. **§2.24's "every comparison is larger at 533ms than at 267ms" is too broad.**
   True for both AUC contrasts. **False for touch − pose-only R²**, which goes
   the *other* way: +0.058 at k=8 → **+0.041** at k=16 (3/4 partitions positive
   at both horizons). §2.24's table only lists AUC, so the table is fine; the
   sentence is not. Scope it to "every AUC comparison" — a reviewer who
   computes R² will otherwise find this.
10. **§2.23's "2 model seeds per cell" is wrong for one cell.** Partition 42 at
    k=8 (`results_aperture_earlystop.json`) has **3** seeds; every other cell
    has 2. Consequently §2.24's "epoch selection has reproduced 16/16" is
    really **17/17** — the claim holds, the count was undercounted.
11. **§2.7's 0.957 had no JSON behind it** ("results printed to stdout only"),
    so it could not be re-derived. Regenerated as
    `results/results_rigid_diag_val.json`; it reproduces exactly. See §2.25.

Corrections 9–11 verified 2026-08-03 by re-deriving every §2.22–§2.24 cell
from the result JSONs. Everything else in those sections reproduced exactly.

Full detail with file:line in `AUDIT.md`.

---

## 4. Running / queued on the cluster

`ssh bashark@mib.media.mit.edu`, repo `~/scratch/bashar/opentouch-gru`,
env `~/miniconda3/envs/opentouch/bin/python`, `PYTHONPATH=src`.

**Nothing is running as of 2026-07-28 17:30.** The participant-disjoint
forecasting sweep (§2.19) and the whole post-sweep queue (§2.20, §2.21) are
complete. Left here because the launch recipe is the reusable part.

```
tmux attach -t scenefc          # live
cat /tmp/scenefc_master.log     # per-run START/DONE
```

18 runs, `sf_{frz,frzshuf,pose}_k{8,16}_s{1,2,3}`, launched by
`scripts/scene_forecast_sweep.sh` as three GPU-pinned streams (one per arm, on
GPUs 0/1/2). Both the tactile encoder (`logs/p2t_scene_gru`) and the
regression split are scene-disjoint. On completion it writes
`results_scene_forecast.json` via `scripts/summarize_scene_forecast.py`, which
refuses to bless the numbers if any run recorded a non-scene split, if either
frozen arm was initialised from something other than `p2t_scene_gru`, or if
the arms disagree on the auto-computed motion threshold.

Confirmed from the smoke run: scene split is **26 scenes → train 20 (2355
clips) / val 3 (345) / test 3 (258)**, frozen trainable parameters **66,045**
(identical to §2.17, so the arms stay capacity-matched), eval n = 11,060.

Note the motion threshold shifts from 0.014044 (clip) to 0.013977 (scene)
because it is the 25th percentile of the *train* split. **The scene-split MSEs
are therefore not directly comparable to §2.17's clip-split MSEs** — the
"moving" subset is a slightly different set of windows. Comparisons within the
sweep are valid; comparisons across the two sweeps are not.

Everything else has completed. All result files are on the cluster and
mirrored to `results/` in the repo (gitignored).

Completed run families, all in `~/scratch/bashar/opentouch-gru/logs/`:

| prefix | what |
|---|---|
| `rr_{pose,tac,shuf}_k{8,16}_s{1,2,3}` | forecasting rebuild, encoder from scratch (§2.16) |
| `rf_{frz,frzshuf,ft}_k{8,16}_s{1,2,3}` | frozen-encoder bridge (§2.17) |
| `p2t_scene_gru`, `p2t_scene_avgpool_norelu` | scene-disjoint retrieval pair (§2.18) |
| `p2t_scene_avgpool` | **superseded**, ReLU-handicapped baseline, do not use |

**Etiquette.** The box is shared and hit load 770 with three other users
active. Everything of mine is `nice 15`–`19`, capped at
`OMP_NUM_THREADS=4`, pinned with `CUDA_VISIBLE_DEVICES`, and queued rather
than launched concurrently. GPUs 5–7 belong to another user's VLLM. Check
`nvidia-smi` and `uptime` before adding anything.

---

## 5. What each new script does

| file | purpose |
|---|---|
| `src/opentouch/articulation_frames.py` | Kabsch rigid-rotation removal + palm-anchored basis. `all_target_variants()` returns all five target definitions; `rigid_fraction()` is the diagnostic. |
| `scripts/rigid_diag.py` | measures the rotation share of the target |
| `scripts/probe_rigid.py` | reruns the direction probe on old vs corrected target. `--all-variants` for the 2×2, `--split-group-by scene`, `--eval-split test` |
| `scripts/probe_rigid_ci.py` | clip-clustered paired bootstrap CIs |
| `scripts/probe_rawpose.py` | **the decisive control**: touch vs full raw kinematics (504-dim lagged pose), not a lossy embedding |
| `scripts/validate_handframe.py` | empirically checks the palm-axis labels |
| `scripts/tactile_subset_probe.py` | pre-registered subset families, causal subset definitions, BH correction, discover-on-val / confirm-on-test |
| `scripts/probe_magnitude.py` | ridge onto the delta vector rather than its sign; the gate that said the forecasting rebuild was worth doing |
| `scripts/export_per_joint.py` | per-(joint, axis) AUC table with clip-clustered CIs; `per_joint_k8.{json,csv}` |
| `scripts/hand_auc_figure.py` | hand diagram shaded by a per-joint quantity, plus a CSV with 2D layout |
| `scripts/collect_results.py` | regenerates `MORNING_REPORT.md` from every result JSON |
| `scripts/overnight.sh` | chains work that depends on a training run finishing |
| `src/opentouch/pose_encoder.py` | now has `temporal_mode={gru,mean}` so both ablation arms run in one repo |
| `src/opentouch_train/data.py` | now has `--split-group-by {clip,scene}` |

Tests: 190+, all passing. `python -m pytest tests/ -q`.

---

## 6. Open questions

Open questions #1, #2 and #3 are now CLOSED -- see §2.19, §2.20, §2.21. The
history below is kept because in all three cases the recorded description of
the work was wrong in a way that mattered, and that pattern is worth carrying
forward: **each was logged as "just run the existing script", and none of them
was.** #1 needed a CLI flag that did not exist, #2's script would have produced
a statistically invalid interval, and #2's first corrected run still silently
scored a scene model on a clip gallery. Re-derive before trusting a queued
task's description.

1. **[CLOSED — see §2.19]** Participant-disjoint frozen-encoder forecasting. §2.17 used the
   clip-disjoint retrieval checkpoint, so the encoder saw val participants'
   clips during retrieval training. The scene-disjoint biGRU checkpoint
   (`logs/p2t_scene_gru`, `split_group_by: scene` confirmed in its
   `params.txt`) now exists, so this is a rerun of §2.17 with
   `--tactile-init-checkpoint` pointed at it and `--split-group-by scene`.
   **This is the highest-value remaining experiment**: it is the only thing
   standing between the current result and "touch improves forecasting for a
   person the model has never seen".

   > **Corrected 2026-07-28.** Earlier versions of this file described this as
   > a pure rerun. It was not: `--split-group-by` did not exist in
   > `regression_params.py`, and `regression_data.py` called
   > `_load_and_split_dataset` without `group_by`, so the regression pipeline
   > could only ever do clip-level splits. Passing the flag would have been a
   > parser error; worse, had the flag existed but not been forwarded, the run
   > would have silently produced a clip-split result wearing a scene-split
   > name. Now wired, with the forwarding pinned by
   > `tests/test_pose_regression.py::test_regression_data_forwards_split_group_by_to_the_splitter`.

   Scope is 18 runs, not 12: `pose-only` must be rerun under the scene split
   too, since the existing `rr_pose_*` baselines are clip-split and are no
   longer a valid comparison. Arms are `frz` / `frzshuf` / `pose`, k ∈ {8,16},
   3 seeds.
2. **[CLOSED — see §2.20]** Retrieval bootstrap CIs. `bootstrap_eval.py` had never been
   run. Retrieval is the SOTA-relative-to-OpenTouch claim and currently has
   only a 3-seed std.

   > **Corrected 2026-07-28.** It should not have been run as it stood — it
   > would have produced an interval contradicting the methodology used
   > everywhere else here. Three defects, now fixed:
   >
   > - It **resampled individual windows**. Windows overlap by 19 of 20
   >   frames, so this is the exact error §2.3 documents for the probe, where
   >   a per-window bootstrap came out ~6x too tight. Now resamples whole
   >   clips; `--cluster window` reproduces the old behaviour for comparison
   >   only.
   > - It **resampled the gallery along with the queries**. mAP is
   >   gallery-size dependent (§2.1), so that blends a metric artifact into
   >   the interval, and duplicated gallery rows tie with the correct target
   >   under `sim >= correct_sims`, inflating ranks. The gallery is now fixed
   >   and only queries are resampled.
   > - It had **no `--split-group-by`**, so it could not bootstrap the
   >   scene-disjoint checkpoints — and §2.18's 4.40x is the number most in
   >   need of an interval. Now defaults to the checkpoint's recorded split.
   >
   > Both statistical properties are pinned in `tests/test_bootstrap_eval.py`,
   > including the converse guard that clustering widens the interval because
   > clips differ in difficulty rather than as blanket inflation.

   All six runs complete; results and the trap they exposed are in §2.20.
3. **[CLOSED — see §2.21]** Per-joint export at k=16. Run; `per_joint_k16.{json,csv}`.
   Curl is indeed largest at k=16 (+0.0198) and is the *only* axis whose k=16
   interval excludes zero.
4. **[IN PROGRESS] Split-seed robustness.** Every participant-disjoint number
   rests on one partition (split_seed=42, 26 scenes -> 20/3/3), so the claim
   rests on three held-out person-locations. A clip bootstrap cannot see this;
   it resamples clips *within* those three scenes. `scripts/split_seed_study.sh`
   redraws the partition at seeds 1/2/3 -- and retrains the retrieval encoder
   at each, because reusing the seed-42 encoder would hand the new val scenes
   to an encoder that trained on some of them, reintroducing the very leak
   §2.19 removes. Launched 2026-07-28 18:22 (tmux `splitseed`), ~12h.

5. **THE ONE GENUINELY OPEN ITEM. A jointly-optimised model.** §2.17 and §2.19 freeze the encoder. Whether joint
   optimisation can beat it, given more data or stronger regularisation, is
   unknown.

## 7. "Is this state of the art?"

Read relative to OpenTouch, which is what was actually asked:

- **Retrieval: yes, unambiguously.** 16.76 → 45.46 T→P on the same codebase,
  same split, same metric, 3 seeds. Nothing on this framework is close. The
  one remaining gap is that the avg-pool arm was historically run from the
  upstream `opentouch` repo; `temporal_mode=mean` now makes both arms run
  here, and the scene-disjoint pair currently training gives a same-codebase
  comparison for the first time.
- **Pose prediction: there is no OpenTouch baseline to beat.** OpenTouch never
  attempted it. So this is a *first* result, not a *better* one. Frame it that
  way; it is a stronger and more honest position than a SOTA claim.
- **Within our own work**, the best forecaster is `frozen tactile + pose`
  (§2.17): it beats pose-only by 11% at both horizons and beats its
  capacity-matched shuffled twin by 14-16%. The earlier "adding tactile makes
  it worse" (§2.9, §2.16) held only when the tactile encoder was trained from
  scratch inside the forecaster, which was a capacity failure, not an
  information one. The remaining gap is participant generalization, not
  performance: the frozen encoder came from a clip-disjoint retrieval run, so
  the claim is currently "for new clips of participants seen during encoder
  pretraining". See open question #1.
