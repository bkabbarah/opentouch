# New findings for the poster — handoff brief

> ## ⚠ REVISED 2026-07-29. READ THIS FIRST.
>
> An earlier version of this brief recommended a **−15% forecasting headline
> stat**. A split-seed robustness study has since shown that number is **not
> robust**, and the recommendation is withdrawn. Section 1 below has been
> rewritten. If you already acted on the earlier version, remove the −15%
> stat and the "touch beats pose-only" claim from the poster.
>
> What survives: the direction of the effect (touch beats its
> capacity-matched control) is consistent, but the magnitude ranges roughly
> −8% to −15% depending on which participants are held out, and **touch does
> not reliably beat a plain pose-only model at all.**

Self-contained. Everything below was read back from result JSONs and cluster
logs, not from prior write-ups. Numbers are exact as stated.

**Current poster status:** the poster covers retrieval + the direction probe
only. Its three headline stats are 2.7× retrieval mAP, 0.69 direction AUC,
+0.039 beyond raw pose. There is no forecasting content on it anywhere.

---

## 1. Forecasting on unseen participants — real, but partition-dependent

A trained forecaster now exists and was run fully participant-disjoint. The
effect is real and its direction is consistent, but its magnitude depends
substantially on which participants are held out, so it does not support a
headline number. Read the whole section before using any figure from it.

**Setup.** Predict the next *k* frames of finger articulation. The tactile
encoder is taken **frozen** from the retrieval checkpoint. Both the encoder's
retrieval pretraining and the forecaster's own training use a
**scene-disjoint** split, so no evaluation participant appears in either
training stage. 18 runs: 3 conditions × 2 horizons × 3 seeds, 300 epochs.

**Metric:** fingertip MSE on the moving subset, rigid (rotation-removed)
target. Lower is better.

| condition | trainable params | k=8 | k=16 |
|---|---|---|---|
| **frozen touch + pose** | 66,045 | **0.000270** | **0.000500** |
| pose-only | 32,957 | 0.000290 | 0.000515 |
| frozen **shuffled** touch + pose | 66,045 | 0.000318 | 0.000588 |
| copy-zero baseline | — | 0.000319 | 0.000622 |

**Paired clip-clustered bootstrap, 1000 draws** (val: 11,060 windows from 342
clips at k=8; 6,540 from 333 clips at k=16). Negative = touch helps.

| comparison | k=8 | k=16 |
|---|---|---|
| frozen touch vs **shuffled twin** | **−14.96% [−18.91, −11.24]** | **−14.86% [−19.09, −10.57]** |
| frozen touch vs pose-only | **−6.87% [−11.06, −2.67]** | −2.70% [−7.50, +1.82] |
| frozen shuffled vs pose-only | +9.52% [+7.36, +11.43] | +14.27% [+10.61, +17.94] |

### What is actually safe to say

**Robust:** on held-out participants, frozen tactile features beat a
**capacity-matched control** (identical architecture, identical 66,045
trainable parameters, tactile stream deterministically mis-paired). This was
negative — touch helping — in **every one of the 6 individually held-out
scenes** and on both the val and test splits, with CIs excluding zero:

| split | k=8 | k=16 |
|---|---|---|
| val (3 held-out people) | −14.96% [−18.91, −11.24] | −14.86% [−19.09, −10.57] |
| test (3 *different* held-out people) | −7.83% [−11.21, −4.17] | −8.91% [−13.25, −4.89] |

**Not robust: the magnitude.** Redrawing the participant partition three more
times (each with its own retrieval encoder retrained on that partition) gives:

| k | partition draws | mean | range |
|---|---|---|---|
| 8 | −5.8%, −10.0%, **+5.0%** | −3.6% | +5.0 to −10.0 |
| 16 | −9.0%, −11.5%, −2.7% | −7.7% | −2.7 to −11.5 |

So the original split was the most favourable of four draws, and at k=8 one
draw **reverses sign**. Across all four partitions the effect averages roughly
**−6% (k=8) and −9.5% (k=16)**, not −15%.

**NOT SAFE AT ALL: "touch beats pose-only."** It holds only on the original
partition's val split at k=8. On that same partition's test split it is −0.40%
(CI includes zero) and +2.72% (includes zero). On the three redrawn partitions
it is *positive* — touch is worse than pose-only — in all six cells. **Do not
put any version of this on the poster.**

### If you want a forecasting line on the poster

Use a qualitative claim with the range, not a headline stat:

> Frozen tactile features reduce forecasting error below a capacity-matched
> control on held-out participants (−8% to −15%, CIs exclude zero on both
> splits), though the margin depends on which participants are held out.

**Recommended:** given a poster's space and the need to defend every number at
close range, consider leaving the forecaster off entirely. The probe and
retrieval results are strong and fully robust; the forecasting result is real
but currently needs three sentences of qualification to state honestly.

## 2. DELETE FROM "FUTURE DIRECTIONS" — both items are done

The poster currently lists as future work:

> *"From linear probes to a trained tactile forecaster, and from direction to
> magnitude"*

**Both are completed results.** The forecaster is section 1 above. Magnitude:

Regressing onto the delta **vector** rather than its sign (k=8, clip-clustered
CIs, all excluding zero), touch reduces error by:

- **+7.4%** against full raw kinematics (504-d)
- +3.7% against the learned pose embedding
- +7.7% against the shuffled control

So touch carries **magnitude information, not only direction**. Worth one line
in RESULTS or CONCLUSIONS.

**Replacement future-directions bullets** (all genuinely open):

- A **jointly optimised** model — the tactile encoder is frozen throughout;
  whether joint training beats it is unknown
- **Scale participants** — the participant-disjoint result rests on 3 held-out
  scenes
- Condition a manipulation policy on tactile embeddings (keep this one, it's
  already there and still open)

---

## 3. Participant-disjoint retrieval, now with confidence intervals

Fits in the CONTROLS column beside the existing "Participants held out +0.023".

Clip-clustered bootstrap, 1000 draws, fixed gallery (queries resampled only,
because mAP depends on gallery size). T→P mAP:

| checkpoint | split | mAP | 95% CI | N windows / clips |
|---|---|---|---|---|
| biGRU | clip-disjoint, test | 45.50 | [42.09, 48.65] | 1399 / 296 |
| **biGRU** | **scene-disjoint, test** | **28.31** | **[25.59, 31.01]** | 1411 / 257 |
| **avg-pool** | **scene-disjoint, test** | **6.46** | **[5.45, 7.59]** | 1411 / 257 |

**Intervals do not overlap.** Ratio ≈ **4.4×** under participant hold-out,
against 2.7× clip-disjoint.

The interesting framing: **the architectural advantage widens when
participants are held out (2.7× → 4.4×) while both absolute numbers fall
sharply.** The clip-disjoint figures were inflated by participant
memorization; the architecture claim survives and strengthens. Presenting both
is more honest and more interesting than either alone.

---

## 4. Corrections to what is already on the poster

**"clip clustered bootstrap over 296 clips" → 295.** The result file records
`n_clusters: 295`. This appears in the controls/seeds line.

**Fig 2's caption is CORRECT — do not change it.** "17 of 20 joints whose
intervals exclude zero on at least two axes" was verified directly against the
per-joint file: radial 17, curl 17, spread 12, and exactly 17 joints clear
zero on ≥2 axes. (If anyone raises "spread is only 12/20," that is the
single-axis count for one axis, a different statistic, and does not affect
this caption.)

---

## 5. New caveat that must go on the poster

**The participant-disjoint split leaves only 3 held-out scenes.** The scene
split deals 26 scenes into train 20 / val 3 / test 3. "Participants the model
has never seen" therefore rests on three held-out person-locations. State it —
a reviewer will ask, and pre-empting it reads as rigor.

Keep the existing caveats too, and add one:

- **The forecasting result uses a FROZEN tactile encoder.** It is a readout
  over pretrained features, not a jointly optimised model.
- (Existing caveat *"the frozen encoder was pretrained on clip-disjoint data,
  so the forecasting panel is not participant-disjoint"* — **this can now be
  DELETED.** It no longer applies; the encoder is scene-disjoint.)

---

## 6. Do NOT put these on the poster

- **The k=16 "touch beats pose-only" margin.** −2.70% [−7.50, +1.82] includes
  zero. The k=8 version (−6.87%) is fine.
- **The older clip-split forecasting numbers** (frozen touch 0.000255 /
  pose-only 0.000287 / "beats pose-only by 11% at both horizons"). Superseded
  by the participant-disjoint runs in section 1 — same experiment without the
  participant confound. Do not mix the two tables; their motion thresholds
  differ, so the absolute MSEs are not comparable across them.
- **The gate-zero diagnostic.** It is a strong answer to "how do you know the
  shuffled control is fair?" but costs too much space to explain. Keep it in
  your pocket for questions. (Short version: forcing the gate to zero makes
  both arms *much worse*, so the shuffled arm is not a crippled model that
  failed to switch itself off — it is a genuine capacity-matched control.)

---

## 7. Where every number lives

On the cluster at `~/scratch/bashar/opentouch-gru`, mirrored to `results/`:

| claim | file |
|---|---|
| participant-disjoint forecasting | `results_scene_forecast.json`; `logs/sf_*` |
| forecasting CIs + gate-zero | `results_forecast_ci_val.json` |
| retrieval CIs | `results_bootstrap_{clip_gru,scene_gru,scene_avgpool}_{val,test}.json` |
| magnitude | `results_magnitude_k8.json` |
| per-joint k=8 (Fig 2) | `per_joint_k8.{json,csv}` |
| per-joint k=16 | `per_joint_k16.{json,csv}` |

---

## 8. Why the forecasting magnitude moves — a likely mechanism

Each redrawn partition needed its own retrieval encoder, and those encoders
came out at noticeably different quality. Validation T→P mAP of the frozen
encoder used by each partition:

| partition | encoder val mAP | frz vs shuffled, k=8 |
|---|---|---|
| original (seed 42) | **29.09** | −14.96% |
| seed 1 | 24.29 | −5.77% |
| seed 2 | 20.78 | −10.03% |
| seed 3 | **18.74** | **+5.02%** |

The best encoder gives the largest tactile benefit and the worst gives the
only reversal. That is a coherent explanation — the forecasting gain depends
on how good the frozen tactile representation is, and representation quality
itself varies with the partition — but it rests on four points and should be
treated as a hypothesis, not a finding.

Practical consequence: the original partition was favourable **twice over** —
a favourable participant draw *and* the best-trained encoder of the four.
