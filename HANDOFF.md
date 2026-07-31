# OpenTouch — working state, updated 2026-07-28

Written so a fresh session (or a fresh person) can pick this up cold. The
conversation is not the source of truth; this file, `AUDIT.md`,
`ABSTRACT_200.md`, the git log, and the result JSONs on the cluster are.

Branch: `audit/frames-subsets-scene-split` (pushed to `origin`, and checked
out on the cluster at `~/scratch/bashar/opentouch-gru`).

---

> **Updated 2026-07-28 (evening).** Open questions #1-#3 are closed. The
> headline: with the tactile encoder frozen from a retrieval checkpoint,
> touch beats its capacity-matched shuffled twin by **15.0% at both horizons
> even when neither training stage ever saw the evaluation participants**
> (§2.19). That is the practical claim that was unavailable for the whole
> project, and it now survives participant hold-out.
>
> Read §2.19 and §2.19b first, then §2.17, §2.10 and §2.11. **Quote the
> shuffled comparison, not the pose-only one.** Under participant hold-out the
> shuffled margin holds at ~15% with clip-clustered intervals far from zero at
> both horizons, while the margin over pose-only compresses to −6.9% at k=8
> and −2.7% at k=16 — and **the k=16 one includes zero, so it is withdrawn.**

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

Not yet run: the same control against the 504-d raw-kinematics baseline
(§2.10, the +0.039). Given the pattern here, assume it is at risk until
measured.

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
