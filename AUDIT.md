# OpenTouch — pre-publication audit and next-step plan

Audit date 2026-07-26, against commit `0a338c1`. Scope: every result in the
results-state document. Method: full read of the retrieval, regression,
contact-structure and probe code paths, plus independent re-derivation of the
load-bearing claims. No data or checkpoints are on this machine, so run
artifacts (`params.txt`, results JSON) could not be inspected — several items
below can only be closed by producing them from the cluster.

Verdict labels: **INVALID** (claim cannot stand), **MUST RESTATE** (effect is
real, stated number or wording is wrong), **UNVERIFIED** (correctness depends
on a run artifact not in the repo), **SOUND**.

---

## 1. Findings that change what can be claimed

### 1.1 The retrieval headline compares against the wrong baseline — MUST RESTATE

The claim "13.43 → 45.46, triples T→P retrieval" pairs a **paper number** with
an **own-codebase number**. `experiments.md:33-38` labels that column
"Paper". The matched own-codebase avg-pool re-run exists and is recorded in
`README.md:276`:

```
| Paper baseline | opentouch | 2026_06_22-20_49_48 | p2t | No | T->P mAP 16.76, matches paper |
| GRU p2t seed 42 | opentouch-gru | 2026_06_22-21_01_54 | p2t | Yes | T->P mAP 45.46 |
```

The correct like-for-like comparison is **16.76 → 45.46, a 2.7× improvement
(+28.7 mAP)**. Still a large, real effect — but the published figure must
change, and the `+32.03` delta in `experiments.md:37` is not a like-for-like
delta.

Corroborating evidence that own-codebase numbers run above the paper's: V→T
and T→V improve by +2.01 and +1.47 between the same two rows, on tasks where
the pose encoder is *not instantiated at all* (`model.py:117` sets
`self.pose = None` for `v2t`). That +1.5–2.0 is a pure protocol offset, the
same size as the 13.43→16.76 gap.

Two further inconsistencies: `experiments.md:80` assigns 13.43 to **P→T** in a
different experiment, and `tactile_contact_encoder.py:116` cites the baseline
as **14.33** (a digit transposition of 13.43). The baseline is a hand-carried
literal, not a tracked measurement.

### 1.2 The anatomical-prior negative result is architecturally guaranteed — INVALID

Two independent problems, each individually sufficient.

**(a) The encoder is provably blind to joint identity.**
`tactile_contact_encoder.py:277-282`:

```python
out = attn @ v                                    # (B,H,21,head_dim)
out = out.transpose(1, 2).reshape(b, NUM_KEYPOINTS, self.d_model)
out = self.output_norm(self.out_proj(out))
pooled = out.mean(dim=1)                          # (B,d_model)
```

`out_proj` is shared across queries, `output_norm` is per-token, and
`mean(dim=1)` is permutation-invariant over the 21 joint slots. Nothing
downstream ever reads "this slot is the index DIP". Permuting `anat_bias`
rows together with `joint_queries` rows leaves the output bit-identical
(max difference 4.8e-07). The only thing an anatomical bias can supply is a
*partition* of taxels; which anatomical joint each group corresponds to is
unrecoverable by construction.

So "anatomy contributes nothing; attention sparsity does the work" is not an
empirical finding. It is a theorem about this architecture. The experiment
could not have detected an anatomy effect however the shuffle was built.

**(b) The shuffled-anatomy control is not in version control.** No code
anywhere shuffles, permutes or randomizes `B_skinning` / `B_region` /
`anat_bias`. `--tactile-encoder-type` accepts only `cnn_gru`,
`contact_skinning`, `contact_region`, `contact_plain`. `git log --all -S`
across every commit finds nothing. The numbers 13.32, "78.7% misassigned",
and the 12.68–14.46 seed range cannot be reproduced or audited.

**(c) Compounding:** the tested encoder peaks at 13.65 mAP against cnn_gru's
45.46 at matched parameters (527,901 vs 508,160) with an identical pose
branch. Any anatomy effect would be swamped 32 mAP below the working
baseline. The negative is about this encoder, not about anatomical priors.

**Also worth knowing** (does not rescue it): the bias is a frozen near-hard
mask, not a soft trainable one — `anat_bias` spans [-10, 0] with 70% pinned
at the floor, against a learned logit scale of std 0.024, and the residual is
gated at 0.1 from zero initialization under weight decay. It cannot be
trained away.

**Recommendation: remove this result from the paper.** Reporting a null from
an architecture that cannot express the alternative is a reviewer-fatal
finding if caught.

### 1.3 "Co-training damage" measures ensemble ablation, not tactile harm — INVALID as worded

`pose_regression.py:236`:

```python
correction_input = torch.cat([pose_flat, tactile_embed], dim=-1)
```

The "tactile correction" branch receives **the full 63-dim pose**. It is a
second *pose* model with tactile side information. Training optimizes
`F.mse_loss` on the **sum** `delta_pose + gate * delta_correction`, so
`self.head` is never fit as a standalone predictor — it is fit as one term of
an ensemble.

Zeroing the gate at eval therefore deletes genuine **pose** capacity, not
tactile contribution. A second branch fed pure noise would produce the same
+0.000574. The number 0.006148 measures "half an ensemble", not "a pose model
damaged by tactile".

The control that would separate these does not exist in the repo: a pose-only
model with a second, tactile-free head, or the two-branch model trained with
the gate hard-frozen at zero.

**Second, independent problem — the training recipe is not matched.**
`regression_train.py:127,132` applies `clip_grad_norm_(model.parameters(),
1.0)` as a **global** norm. The tactile model adds ~500k parameters to that
norm; whenever it exceeds 1.0 every gradient including the pose head's is
rescaled. The pose head sees a different effective learning rate in the two
conditions. This is exactly the mechanism that manufactures a small
"co-training damage" with no tactile involvement.

### 1.4 "Active tactile harm" is capacity cost, and the control says so — MUST RESTATE

The four numbers read: tactile+pose 0.008694, shuffled-tactile 0.009901,
pose-only 0.005574, copy-zero 0.006717. The shuffled control is **worse than
real tactile**, and both lose to predicting exactly zero.

A model that loses to the zero baseline is overfitting, not reporting on
information content. The supportable reading is: *adding a ~500k-parameter
second branch costs ~+0.003 val MSE, and tactile content is largely
irrelevant to that cost.* That supports the null on tactile **content**. It
does not support "tactile actively harms", which attributes to tactile an
effect its own shuffled control shows is not tactile's.

**Also:** `regression_main.py:219-221` constructs an extra reference model for
a parameter-parity assert **only under `--shuffle-tactile`**, which advances
the global torch RNG. Dropout masks and DataLoader shuffle order therefore
differ, so "shuffled seed 1" is not paired with "tactile+pose seed 1". The
0.0012 gap between them is ~1.3× the shuffled condition's own seed spread.
Fix by wrapping the parity check in `torch.random.fork_rng()`.

### 1.5 The target is not articulation — rigid rotation was never removed — MUST RESTATE

`pose_regression.py:130-132`:

```python
wrist_delta = world_delta[:, WRIST_INDEX, :]
articulation_delta = world_delta - wrist_delta.unsqueeze(1)
```

This removes whole-hand **translation** only. A rigid rotation of the hand
about the wrist — forearm pronation, wrist flexion, fingers perfectly still
relative to the palm — moves every non-wrist keypoint, and all of it lands in
what the codebase calls "articulation". Nothing anywhere in the repo performs
rotation canonicalization (`pose_encoder._normalize_pose` centres and scales
only).

**Why this is serious for the central claim.** The direction probe found its
entire above-chance signal on the world **y** axis, with x and z at chance at
every horizon. World axes carry no anatomical meaning. A signal confined to a
single lab axis is what a *global* effect looks like — a dominant vertical
motion direction, gravity, systematic wrist re-orientation — and is not what
finger flexion should look like, since flexion has no reason to prefer one
world axis. As it stands the probe cannot distinguish "tactile predicts
finger articulation direction" from "tactile predicts whole-hand
re-orientation".

Addressed by the new `src/opentouch/articulation_frames.py` (below).

### 1.6 The k=16 configuration cannot have run as described — UNVERIFIED

`regression_params.py:182-200` hard-raises for exactly the reported
configuration:

```
parse_regression_args(['--train-data','x','--horizon-k','16'])
-> ValueError: --causal with --min-history=10 leaves ZERO valid samples for
   --horizon-k=16 --sequence-length=20 --causal-window=20
```

At k=16 and T=20 the longest real causal history is `min(20-16, 20) = 4 < 10`.
So k=16 ran with one of `--noncausal` (the leak the causal fix exists to
close), `--min-history ≤ 4` (1–4 real frames plus 80–95% edge padding fed to a
biGRU trained on 20-frame dynamics — the regime the filter exists to exclude),
or a non-default `--sequence-length ≥ 26` (which changes windowing, window
count, motion threshold and copy baseline, so the k-sweep is no longer
controlled).

The probe's own T=36 runs satisfy `36 ≥ 16 + 10` and are fine. The
**regression** k=16 runs are the open question. This is resolvable in
minutes from `params.txt`.

### 1.7 No uncertainty is computed anywhere — MUST FIX

The probe reports point AUCs with no interval, and the redundancy verdict
(+0.000 to +0.004) is asserted as a null with no confidence interval at all.
Samples are far from independent: adjacent `t` within a window share 19 of 20
causal frames and most of their target, and windows within a clip share
participant, scene, glove calibration and activity. Nominal n overstates
information by roughly the sample-to-window ratio.

Additionally, "all 20 joints clear 0.55" is not 20 confirmations —
`regression_metrics.py:3-6` records that the 21 keypoints are retargeted from
**7 Rokoko sensors**, so most joints are kinematic interpolations of the
sensor-driven ones. Effective independent measurements ≈ 7 at most.

Addressed by the clip-clustered paired bootstrap in the new subset script.

### 1.8 The split is clip-disjoint but not participant-disjoint — RISK

`data.py:83-108` splits on `(scene, clip_id)` tuples. Clips from the same
scene land in train, val and test. `failure_analysis_100.csv` shows ~21
distinct scenes (location + participant ID) across the worst 100 test
queries, over ~1150 clips — essentially every test scene and participant also
appears in train.

The biGRU adds ~440k parameters to a pose encoder that previously had ~50k. A
high-capacity temporal model can memorize participant hand geometry, glove
calibration, and scene-conditioned pose-tactile coupling and reapply it to
held-out clips of the same participant; avg-pooling has far less capacity to
do so. This is the most plausible route by which the 2.7× is inflated rather
than fabricated. Nothing in the repo tests it.

Related, untested: both encoders read out window **endpoints**
(`pose_encoder.py:62`, `tactile_encoder.py:52`) and both modalities are cut at
identical frame boundaries (`data.py:221-226`), so endpoint-signature matching
is available to the GRU and structurally unavailable to avg-pooling.

---

## 2. What checks out

Worth stating plainly, because it is most of the pipeline.

- **Causality of the probe.** Verified independently by poisoning frames > t
  with 1e6 and asserting invariance, against the actual production functions
  (`encode_causal`, `encode_pose_causal` including `PoseEncoder`'s internal
  time-averaged scale statistic, and the deranged-window path). No leak.
- **The redundancy verdict survives a hard attempt to break it.** Per-feature
  `StandardScaler` is applied, so tactile columns enter at unit variance;
  L2 at `C=1.0` is negligible against n≈10⁴–10⁵ so the probes are effectively
  unregularized MLE; and the *same* 127-dim machinery does detect tactile's
  +0.015–0.033 over single-frame pose. The pipeline demonstrably can register
  a tactile increment at that dimensionality. Its failure to do so over
  `pose_emb_causal` is informative, not an artifact.
- **Split hygiene.** Clip-disjoint by construction; windows built after the
  split with `stride = sequence_length` so no cross-split window overlap.
  `--split-seed` is decoupled from `--seed` and correctly plumbed.
- **Normalization.** No dataset-level statistics exist. Tactile is a constant
  `/255.0`; pose normalization is strictly per-sample. No leak.
- **Derangement control.** True derangement (rejection-sampled, re-asserted at
  access), a bijection over windows so it preserves the tactile marginal
  exactly, composes correctly with causal slicing.
- **Target indexing.** No off-by-one: the target sits exactly k frames after
  input frame t and exactly k frames after the last causal tactile frame.
- **Copy-zero baseline.** Computed on the same masked tensors through the same
  summarizer as the model MSE. Normalization identical across conditions.
- **Motion threshold.** Computed once on train, stored in the checkpoint,
  reused at eval. No val leakage.
- **Gate=0 zeroes tactile exactly at inference.** Separate BatchNorm per
  branch, no shared trunk, no dropout path connecting them; the scalar
  multiply is the only coupling. (The *interpretation* of gate=0 is the
  problem — see 1.3 — not the mechanics.)
- **Parameter parity** across the three contact modes: exactly 527,901,
  enforced by assert and test.
- **No model selection on val.** Checkpoints saved by epoch only.
- **Test suite**: 150 passing, including 40 added by this audit.

---

## 3. What was added by this audit

### `src/opentouch/articulation_frames.py` + `tests/test_articulation_frames.py`

Separates the three levels the current target conflates, and provides a
palm-anchored basis so per-axis results mean something anatomical:

| variant | what it is |
|---|---|
| `world` | raw `p(t+k) - p(t)` |
| `wrist_translation_removed` | the codebase's current target, bit-compatible (asserted in test); rigid rotation still present |
| `wrist_translation_removed_handframe` | same quantity, palm axes |
| `rigid_removed` | best rigid rotation about the wrist fit by Kabsch and removed; the true non-rigid residual |
| `rigid_removed_handframe` | the above in palm axes — the strictest and most interpretable target |

`rigid_fraction()` returns the single diagnostic number: what share of
"articulation" energy the best rigid rotation explains. Tests verify that pure
rigid rotation produces large motion under the current target and ~zero under
the rigid-removed one, that Kabsch never returns a reflection, and that
hand-frame components are invariant to an arbitrary lab frame while
world-axis components are not.

Note on the PI constraint: defining the target's coordinate frame is
measurement, not an architectural prior. No model is given anatomical
structure and nothing here constrains any network's parameters or attention.

### `scripts/tactile_subset_probe.py` + `tests/test_tactile_subset_probe.py`

Tests the regime hypothesis with the guards that make a positive result
survivable, and supplies the missing statistics. Four pre-registered subset
families, each with a stated mechanism:

| family | bins | hypothesis |
|---|---|---|
| `contact_level` | no / light / firm | tactile informs motion only where the hand is loaded |
| `contact_change` | releasing / steady / onsetting | tactile informs motion at contact transitions, where kinematic continuation breaks |
| `pose_speed` | slow / medium / fast | tactile helps where pose velocity is uninformative |
| `grip_aperture` | pinch / medium / open | tactile's value depends on grasp configuration |

Enforced properties:

- **Causal subset definitions.** Every subset statistic is computed from the
  same causal slice the encoder saw; a poisoning test asserts no frame > t
  enters. A subset defined using the future would select on the answer.
- **Train-fitted bin edges**, applied unchanged to eval.
- **Matched control inside every cell.** The shuffled-tactile twin is refit
  per subset, so an easier subset raises pose, tactile and shuffled together
  and only tactile-minus-shuffled reads as signal.
- **Clip-clustered paired bootstrap CIs**, with `--cluster-unit scene` for the
  stricter participant-level question. A test verifies clustered intervals are
  wider than naive per-sample intervals under within-cluster correlation.
- **Benjamini–Hochberg across the whole pre-registered grid**, with the grid
  size written into the output so it cannot be quietly recomputed on a
  smaller grid later.
- **Discover on val, confirm on TEST.** The test split is untouched by the
  original probe (it asserts train/val only), so it is a genuine held-out
  confirmation set. `--stage confirm` re-runs only the cells that survived BH.

---

## 4. Ordered next steps

### Tier 0 — hours, no training. Do these before writing anything.

1. **Pull `params.txt`** for the five regression runs and the probe runs.
   Record `causal`, `min_history`, `sequence_length`, `motion_threshold`,
   `seed`, `git_commit`, `git_dirty`. Resolves 1.6. If k=16 shows
   `causal: False` or `min_history: 1`, withdraw that row.
2. **Sample-set parity canary.** `moving_copy_baseline_mse_fingertips` is a
   pure statistic of the eval targets — it must be identical to the last digit
   across all four regression configs. If any differs from 0.006717, that
   config was scored on a different sample set and every comparison involving
   it is invalid.
3. **Run `rigid_fraction`** over the val split. One number: how much of
   "articulation" was whole-hand rotation. If it is high, 1.5 reframes the
   central claim and everything downstream changes.
4. **Report the full-set (`all_*`) metric orderings** alongside the
   moving/fingertip subset. If the orderings disagree, the subset choice needs
   justification in the text.
5. **Fix the RNG divergence** (`fork_rng` around the parity check) and re-pair
   the shuffled seeds.

### Tier 1 — about a day, frozen encoders, no retraining.

6. **Re-run the direction probe on `rigid_removed_handframe`** with clustered
   CIs. This is the decisive test of the central claim. If the signal survives,
   it is genuine articulation prediction and the claim strengthens. If it
   vanishes, the "real but weak directional signal" was whole-hand motion and
   Section 6 must be rewritten.
7. **Run `tactile_subset_probe.py --stage discover`** across horizons. Answers
   the PI's question under conditions where a positive would hold up.

### Tier 2 — only if Tier 1 produces something.

8. **`--stage confirm`** on the held-out test split for surviving cells.
9. **Scene-disjoint retrieval re-run** (group the split on `scene` instead of
   `(scene, clip_id)` — one line at `data.py:146`) and an **offset-gallery
   check** (gallery windows at a half-window offset). Together these bound how
   much of the 2.7× is participant memorization and boundary alignment.
10. **Pose-only-with-second-head control**, or the two-branch model trained
    with the gate frozen at zero, before the phrase "co-training damage" is
    used anywhere.

### Housekeeping

- Commit a leak test for the probe's own production path
  (`encode_causal` / `encode_pose_causal`); the existing test covers
  `PoseTransitionDataset.__getitem__`, which the probe never calls. The
  docstring at `tactile_direction_probe.py:370-373` cites a "module report"
  that is not in the repo — commit it or delete the reference.
- Store `split_seed` in retrieval checkpoints; `eval.py:113` currently
  overloads `--seed` for it, and a mismatch silently evaluates on training
  clips.
- Sort clip keys before shuffling in `data.py:147,287`. The split currently
  depends on dataset row order, so rebuilding the HF dataset silently
  repartitions at the same `--split-seed`.
- `metrics.py:75-77` computes MRR (`1/rank`), which equals mAP only with
  exactly one relevant item per query. Rename, or confirm the paper's
  definition matches — this rides on top of 1.1.
- The trimodal-overfitting explanation in `experiments.md:137` is contradicted
  by the data code: `_build_sliding_windows` never filters on modality
  availability, so the trimodal training set is exactly the bimodal one. The
  drop is real; the stated mechanism is not.
- Windows stride over list positions, not `frame_idx` values
  (`data.py:190`), and dropped frames upstream can make a T-frame window span a
  temporal gap — which corrupts a k-frame delta target. Add a contiguity
  assert.
