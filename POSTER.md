# Poster: what to put up, and what not to

Title: **Piezoresistive Palm Reading: Tactile Prediction for Dexterous Manipulation**

Every number below is traceable to a result file. Section 5 lists what is
unsafe to display and why. Read that before laying anything out.

Updated 2026-07-28: the forecasting panel (1b) and the participant-disjoint
retrieval figure are both new and both safe. Nothing is quarantined any more.

---

## 1. Safe numbers, ready to display

### Retrieval (Panel: "Better alignment")

| pose encoder | T→P mAP (test) |
|---|---|
| average pooling | **16.8** |
| bidirectional GRU | **45.5** |

**2.7x.** Same codebase, same eval path, same clip-disjoint split, verified
2026-07-27. Three GRU seeds give 46.17 ± 0.54.

Put the gallery size on the slide: **N = 1399 windows**. mAP here is
mean(1/rank) against the whole eval split, so it is not comparable across
papers with different gallery sizes.

### Direction decoding (Panel: "Does touch predict motion?")

Touch alone, decoding the sign of future finger articulation, corrected
target, val:

| k | ms ahead | touch | shuffled touch |
|---|---|---|---|
| 2 | 67 | 0.648 | 0.501 |
| 4 | 133 | 0.651 | 0.501 |
| 8 | 267 | 0.661 | 0.501 |
| 16 | 533 | **0.690** | 0.496 |

Chance is 0.50. Across all three anatomical axes the range is **0.60 to 0.69**.

### The contribution over pose (Panel: "Is it redundant?")

Marginal AUC of adding touch to a pose baseline with **identical temporal
access**, k=8, clip-clustered bootstrap over 296 val clips:

| baseline | touch adds | 95% CI |
|---|---|---|
| learned pose encoder | +0.019 | [+0.006, +0.031] |
| **full raw kinematics (504 dims)** | **+0.039** | [+0.023, +0.054] |

Both exclude zero. The raw-kinematics row is the strong one: it rules out
"touch is just recovering what a 64-dim bottleneck discarded."

Across seeds the marginal over the learned encoder is **+0.014 ± 0.002**
(seeds 42 / 0 / 1 give +0.0171 / +0.0124 / +0.0120). **Quote the mean, not
seed 42.**

### 1b. Forecasting (Panel: "Does it actually help?")

Predicting the future articulation vector, 3 seeds, moving subset, fingertip
MSE. The tactile encoder is taken frozen from the retrieval checkpoint.

| condition | trainable | k=8 | k=16 |
|---|---|---|---|
| **frozen touch + pose** | 66,045 | **0.000255** | **0.000461** |
| pose-only | 32,957 | 0.000287 | 0.000521 |
| frozen **shuffled** + pose | 66,045 | 0.000297 | 0.000547 |
| copy-zero | — | 0.000312 | 0.000609 |

**Touch beats pose-only by 11% at both horizons, and beats its
capacity-matched shuffled twin by 14-16%** at identical trainable parameter
count. The shuffled arm being slightly *worse* than pose-only is the point:
real touch first pays back the cost of carrying the branch, then profits.

Say that the encoder is frozen. A trained-from-scratch tactile branch loses to
pose-only (0.000356), and that contrast is worth a sentence: the information
was always there, the model was drowning in its own parameters.

### Participant-disjoint retrieval (Panel or footnote)

| split | avg-pool | biGRU | ratio |
|---|---|---|---|
| test | 6.44 | 28.31 | **4.40x** |

Against 2.71x clip-disjoint. The architectural advantage *widens* when
participants are held out, while both absolute numbers fall — so the
clip-disjoint figures were inflated by participant memorisation. Presenting
both is more honest and more interesting than presenting either alone.

### The methodological result (Panel: "Why it was missed")

Rotation is **~95%** of the conventional target's energy at the median sample
(73–75% of samples are >80% rigid, stable across horizons).

Marginal AUC at k=8 under the 2×2:

| | world axes | palm axes |
|---|---|---|
| rotation kept | +0.002 | +0.005 |
| rotation removed | +0.003 | **+0.017** |

It is an **interaction**: neither change alone recovers the signal. This is
probably your most transferable finding and deserves a panel of its own.

### Robustness (Panel or footnote)

| check | corrected target | published target |
|---|---|---|
| held-out TEST, k=8 | **+0.020** | −0.001 |
| held-out TEST, k=4 | **+0.019** | −0.003 |
| participants held out (clean encoder) | **+0.023** | +0.001 |

Test numbers are higher than val, and the participant-disjoint number is the
strongest of the set.

### The PI's regime hypothesis (one line)

**0 of 36** pre-registered subset cells survive multiplicity correction, on
both targets. Touch's contribution is global, not concentrated in any
contact-level, contact-transition, pose-speed or grip-aperture regime.

---

## 2. The centrepiece figure

Grouped bars, marginal AUC at k = 2/4/8/16, two series:

- **conventional target** (near zero, decaying to +0.0001 at k=16)
- **corrected target** (flat at +0.013 to +0.017)

One picture carries the whole methodological argument: the same data, the
same model, the same controls, and the conclusion inverts on the definition
of what is being predicted.

Second figure if you have room: the 2×2 interaction table above, as a heatmap.

---

## 3. Suggested panel order

1. Motivation — video sees appearance, not contact
2. Method — trimodal InfoNCE, the three encoders
3. Better alignment — 2.7x retrieval
4. The question — does touch predict what the hand does next?
5. **Centrepiece** — conventional vs corrected target across horizons
6. What touch predicts — 0.60 to 0.69, shuffles at chance, strongest for curl
7. Not redundant — +0.039 over full raw kinematics, CI excludes zero
8. **It actually helps** — frozen touch beats pose-only by 11%, beats its
   capacity-matched shuffle by 14-16%
9. Controls — participant-disjoint, held-out test, three seeds, no regime
10. Conclusion — touch reveals how the hand will reshape, not just what it holds

---

## 4. Honest caveats to keep visible

Reviewers respect these and they cost you nothing.

- **The forecasting result uses a FROZEN encoder.** It is a readout over
  pretrained features, not a jointly optimised model.
- **The frozen encoder was pretrained on clip-disjoint data**, so the
  forecasting panel is not participant-disjoint. The probe panels are.
- **Linear probes on frozen encoders.** The claim is about linearly
  accessible information, not about a model that exploits it.
- **"Strongest for curl" is a ranking, not a separated effect.** Curl tops
  all four horizons but the gap over radial is +0.0011/+0.0005/+0.0015 at
  k=2/4/8 and the intervals overlap. Do not defend it as significant.
- **20 joints is not 20 independent measurements.** The 21 keypoints are
  retargeted from 7 Rokoko sensors.
- **Clip-clustered intervals are wide on purpose.** 11,425 eval samples come
  from 296 clips; a per-sample bootstrap would be ~6x tighter and wrong.

---

## 5. DO NOT PUT THESE ON THE POSTER

**The old forecasting numbers** (tactile+pose 0.0087 vs pose-only 0.0056).
Those runs were noncausal, used the rotation-dominated target, and had a
confounded ablation. Superseded by the panel in section 1b.

**The 5.36x participant-disjoint retrieval ratio.** Superseded: that avg-pool
baseline was handicapped by a ReLU the upstream architecture never had. The
corrected figure is **4.40x**, and it is safe to display.

**The anatomical-prior negative result.** Invalid: the encoder mean-pools over
its 21 joint queries after a shared projection, making it provably invariant
to joint relabeling, so anatomy could not have helped by construction. Cut it
entirely.

**"Redundant with pose kinematics."** Withdrawn. It was an artifact of the
rotation-dominated target.

**Anything sourced to 13.43 mAP.** That is the paper's avg-pool number, not
this codebase's. The matched figure is 16.76.

---

## 6. Where each number lives

| claim | file, on the cluster at `~/scratch/bashar/opentouch-gru` |
|---|---|
| retrieval, both arms | `results_retrieval_clip_{gru,avgpool}_{val,test}.json` |
| horizons | `results_probe_rigid_k{2,4,8,16}_T36.json` |
| confidence intervals | `results_rigid_ci_k8_T36.json` |
| raw-kinematics control | `results_rawpose_k8.json` |
| 2×2 decomposition | `results_probe_rigid_k8_DECOMP.json` |
| participant-disjoint | `results_probe_rigid_k8_SCENE{,_CLEANENC}.json` |
| held-out test | `results_probe_rigid_k{4,8}_TEST.json` |
| seeds | `results_probe_rigid_k8_seed{0,1}.json` |
| rotation share | `scripts/rigid_diag.py` (stdout) |
| axis-label validation | `results_handframe_validation.json` |
| subsets | `results_subset_discover_*.json` |
| magnitude | `results_magnitude_k8.json` |
| forecasting | `logs/rf_{frz,frzshuf,ft}_k{8,16}_s{1,2,3}/out.log` |
| participant-disjoint retrieval | `results_retrieval_scene_avgpool_norelu_*.json`, `results_retrieval_scene_p2t_scene_gru_*.json` |
| per-joint AUC | `per_joint_k8.{json,csv}` (also copied to `results/`) |

`MORNING_REPORT.md` regenerates all of it:

```bash
ssh bashark@mib.media.mit.edu 'cd ~/scratch/bashar/opentouch-gru && PYTHONPATH=src ~/miniconda3/envs/opentouch/bin/python scripts/collect_results.py --output MORNING_REPORT.md && cat MORNING_REPORT.md'
```
