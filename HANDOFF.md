# OpenTouch — working state, 2026-07-26

Written so a fresh session (or a fresh person) can pick this up cold. The
conversation is not the source of truth; this file, `AUDIT.md`,
`ABSTRACT_200.md`, the git log, and the result JSONs on the cluster are.

Branch: `audit/frames-subsets-scene-split` (pushed to `origin`, and checked
out on the cluster at `~/scratch/bashar/opentouch-gru`).

---

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
| seed 1 (`2026_07_06-22_36_12`) | running | — |

Positive at both seeds so far, but the magnitude varies by ~27%. Report as a
range, not a point estimate, once seed 1 lands.

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

### 2.9 Regression / forecasting (Phase 2B) — still negative

| config | k=16 MSE (moving, fingertips, articulation) |
|---|---|
| copy-zero baseline | 0.006717 |
| pose-only | 0.005574 |
| tactile + pose | 0.008694 |
| shuffled tactile | 0.009901 |

**We have no forecaster that uses tactile and beats pose-only.** The probe
shows the information is present; nothing yet shows a model exploiting it.
These runs are also pre-causal-fix (see §3.4), used the rotation-dominated
target, and had a confounded ablation. Rebuilding this is the main open work.

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

1. **Does touch add over FULL raw kinematics?** (`probe_rawpose.py`, queued.)
   This is the difference between "touch carries contact information
   kinematics lack" and "touch recovers what a 64-dim bottleneck discarded".
   If it fails, soften the claim to "beyond a learned pose encoder".
2. **Rebuild the forecasting arm** against the corrected target with causal
   tactile and an unconfounded ablation. Until a model beats pose-only, this
   is a representation-analysis result, not a prediction result.
3. **Scene-disjoint encoder + scene-disjoint probe** once training finishes.
4. **Test-split confirmation** (queued).
5. **Seed 1**, then report the marginal as a range.
6. Retrieval bootstrap CIs. `bootstrap_eval.py` exists and has never been run.

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
