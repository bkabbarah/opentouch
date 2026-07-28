# OpenTouch — working state, updated 2026-07-27

Written so a fresh session (or a fresh person) can pick this up cold. The
conversation is not the source of truth; this file, `AUDIT.md`,
`ABSTRACT_200.md`, the git log, and the result JSONs on the cluster are.

Branch: `audit/frames-subsets-scene-split` (pushed to `origin`, and checked
out on the cluster at `~/scratch/bashar/opentouch-gru`).

---

> **Updated 2026-07-27 after the overnight run.** Every open question from the
> first version has been answered. See `MORNING_REPORT.md` on the cluster for
> the auto-generated current view; sections 2.10 to 2.14 below carry the new
> results.

## 1. Where things stand in one paragraph

The published direction-probe conclusion ("tactile is redundant with pose
kinematics") was an artifact of the prediction target. The codebase's
"articulation delta" removes wrist *translation* only, so whole-hand
*rotation* stays in, and rotation is ~95% of that target's energy at the
median sample. Isolating true finger articulation and expressing it in a
palm-anchored frame raises tactile's marginal contribution over a
matched-temporal pose baseline from +0.0017 to +0.0171 at k=8, and the effect
holds at all four horizons, under participant-disjoint splits, and with
clip-clustered confidence intervals excluding zero. Separately, the retrieval
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

### 2.3 Confidence intervals (k=8, clip-clustered, 1000 draws, 296 val clips)

| axis | vs matched pose | vs shuffled twin | joints CI>0 |
|---|---|---|---|
| curl | +0.0187 [+0.0064, +0.0311] | +0.0215 [+0.0082, +0.0354] | 17/20 |
| radial | +0.0171 [+0.0040, +0.0314] | +0.0195 [+0.0055, +0.0346] | 17/20 |
| spread | +0.0126 [+0.0009, +0.0245] | +0.0161 [+0.0022, +0.0300] | 15/20 |

Clustering at the clip level is essential: 11,425 eval samples come from only
296 clips, and adjacent t share 19 of 20 causal frames. A per-sample bootstrap
reports roughly six times tighter and is wrong.

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
needs the scene-disjoint-trained encoder (see §4).

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

Seed std ≤ 0.000027 throughout. Gates converge to a consistent sign
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

### 2.17 Frozen-encoder bridge — running as of 2026-07-27 21:00

The regression trains its tactile encoder from RANDOM init (~500k parameters
against a 33k pose head, ~116k samples, MSE loss). Every positive result in
this project comes from that encoder FROZEN from the retrieval checkpoint. The
forecaster had never been given the encoder that demonstrably contains the
signal.

`--tactile-init-checkpoint` + `--freeze-tactile-encoder` close that gap.
Verified at launch: 24 tensors loaded strict, trainable parameters drop from
574,205 to **66,045**, so the tactile arm is now comparable to pose-only's
32,957 rather than 17x larger.

Three arms x k in {8,16} x 3 seeds: frozen, frozen+shuffled (capacity-matched
control), fine-tuned. Logs `/tmp/rf_{frz,frzshuf,ft}_k*_s*.log`, runs
`logs/rf_*`.

**How to read it.** If frozen wins, the information was always there and the
forecaster just needed the right encoder -- that is the practical claim. If it
still loses, that is a specific and strong negative: the information is real
but not exploitable at this data scale. Neither answer is available today.

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

| tmux session | what | state |
|---|---|---|
| `scene_gru` | biGRU, scene-disjoint split, 300 ep, GPU 4 | ~epoch 175 |
| `scene_avg` | avg-pool, scene-disjoint split, GPU 2 | ~epoch 177 |
| `validity` | probe on seed0 then seed1 | seed0 done, seed1 running |
| `rawpose` | **the decisive control**, waits for probes to drain | queued |
| `testconf` | probe on held-out TEST split, k=4 and k=8 | queued |
| `subset_rigid` / `subset_wrist` | pre-registered subset families | ~family 3 of 4 |

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
| `src/opentouch/pose_encoder.py` | now has `temporal_mode={gru,mean}` so both ablation arms run in one repo |
| `src/opentouch_train/data.py` | now has `--split-group-by {clip,scene}` |

Tests: 190+, all passing. `python -m pytest tests/ -q`.

---

## 6. Open questions, in priority order

Items 1, 3, 4 and 5 from the first version are all **done** and reported in
§2.10 to §2.14. What remains:

1. **Rebuild the forecasting arm.** Now well motivated: §2.11 shows touch
   carries magnitude information, so the earlier null was a design problem
   (rotation-dominated target, confounded ablation, noncausal tactile) rather
   than absence of signal. Needs the corrected target plumbed into
   `regression_data`/`pose_regression`, an ablation where zeroing the gate
   removes only tactile, T=36 so k=16 is feasible with min-history, and
   three seeds per condition.
2. **Retrieval bootstrap CIs.** `bootstrap_eval.py` exists and has never run.
   Retrieval is the actual SOTA-relative-to-OpenTouch claim and currently has
   only a 3-seed std.
3. **Subset confirm stage** is moot: nothing survived discovery.

---

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
- **Within our own work**, the best forecaster is pose-only, and adding
  tactile makes it worse (§2.9). That is the gap between what we have and any
  performance claim.
