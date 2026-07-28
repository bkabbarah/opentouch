# Does touch tell us anything about hand motion that pose alone doesn't?

Two experiments, one question. Every number below was read back from the result
JSONs and cluster logs, not from previous write-ups.

**Bottom line.** The earlier conclusion that tactile is redundant with pose
kinematics was an artifact of what we were asking the model to predict. With a
corrected prediction target, touch carries real information about finger motion
— direction *and* magnitude — and it survives every control we've thrown at it.
As of the latest runs, touch also improves an actual trained forecaster,
beating pose-only by 11% and beating its capacity-matched control by 14–16%.
The one claim we cannot yet make is that this holds for a **person the model
has never seen**; that experiment is specified and ready to run.

---

## Part 1 — Direction prediction

### The bug that was driving the old result

We were predicting "articulation delta," which was supposed to mean *finger
motion with whole-hand motion removed*. It only subtracted wrist **translation**.
Whole-hand **rotation** stayed in — and rotation is **95.7%** of that signal's
energy at the median sample (k=8; 74.5% of samples are >80% rigid).

So the target was mostly "predict how the wrist rotates." Pose kinematics
predicts that almost perfectly, and touch has nothing to add. The old null was
real, but it was a null about the wrong question.

The fix removes rotation via Kabsch alignment *and* re-expresses the residual in
a palm-anchored frame. Both are required — this is an interaction, not two
independent improvements:

| marginal gain from touch (k=8) | world axes | palm axes |
|---|---|---|
| rotation kept | +0.0017 | +0.0052 |
| rotation removed | +0.0025 | **+0.0171** |

Frame alone buys +0.0035. Rotation removal alone buys +0.0008. Together, +0.0154.

### The corrected result

Marginal AUC from adding touch to a pose model that already sees an identical
20-frame causal window:

| horizon k | ms | n eval | old target | corrected target |
|---|---|---|---|---|
| 2 | 67 | 15,196 | +0.0030 | **+0.0168** |
| 4 | 133 | 14,006 | +0.0036 | **+0.0164** |
| 8 | 267 | 11,425 | +0.0017 | **+0.0171** |
| 16 | 533 | 6,626 | +0.0001 | **+0.0132** |

An order of magnitude larger, and stable across all four horizons.

### Four controls, all passed

**1. Confidence intervals, clustered at the clip.** k=8, 1000 draws, 295 clips.
Clustering matters: 11,425 samples come from only 295 clips and adjacent
timesteps share 19 of 20 frames, so a per-sample bootstrap is ~6× too tight.

| axis | vs matched pose | vs shuffled control |
|---|---|---|
| curl | +0.0187 [+0.0064, +0.0311] | +0.0215 [+0.0082, +0.0354] |
| radial | +0.0171 [+0.0040, +0.0314] | +0.0195 [+0.0055, +0.0346] |
| spread | +0.0126 [+0.0009, +0.0245] | +0.0161 [+0.0022, +0.0300] |

All exclude zero. Per-joint, 17/20 joints exclude zero on curl and radial;
spread is weaker at 12/20.

**2. Held-out test split.** Test numbers are *higher* than validation
(+0.0192 at k=4, +0.0199 at k=8), while the old target sits at −0.0025 / −0.0012.

**3. Participant hold-out.** With whole scenes held out so no participant
appears in both train and eval, the effect survives at **+0.0165**, and at
**+0.0225** using an encoder that was itself trained participant-disjoint. The
old target goes *negative* (−0.0051) under the same test.

**4. Against uncompressed kinematics — the control that matters most.** The
obvious objection is that touch is just recovering what the 64-dim pose
embedding threw away. So we gave the baseline 504 dimensions of raw lagged
joint positions instead:

| axis | raw kinematics | + touch | touch adds |
|---|---|---|---|
| radial | 0.6469 | 0.6842 | **+0.0373** [+0.0197, +0.0567] |
| spread | 0.6222 | 0.6425 | +0.0203 [+0.0072, +0.0337] |
| curl | 0.6200 | 0.6586 | **+0.0386** [+0.0232, +0.0544] |

Touch adds on every axis against uncompressed positions. The bottleneck
explanation is dead. Worth noting: raw kinematics is *worse* than the learned
embedding on two of three axes, so the contrastive encoder is extracting
something a linear readout of positions cannot.

### Direction is not the whole story — magnitude too

Regressing onto the delta *vector* rather than its sign, touch reduces error by
**7.4%** against raw kinematics and **7.7%** against the shuffled control
(CIs exclude zero). This was the gate: it said touch carries usable magnitude
information, which is what justified rebuilding the forecaster.

### What we deliberately did not find

A pre-registered subset analysis asked whether touch helps specifically in
certain regimes — high contact, contact transitions, fast motion, wide grip.
**0 of 36 cells survive multiplicity correction.** Touch's contribution is
global, not concentrated in any regime. This is a cleaner result than a
subset finding would have been, and we are reporting it as such rather than
mining it.

### Honest caveats

- **Encoder seed sensitivity.** Across three retrieval checkpoints the k=8
  marginal is +0.0171 / +0.0124 / +0.0120 — **mean +0.0138, std 0.0023**. The
  seed every headline came from is the outlier high. We quote the mean. Touch
  *alone* is far more stable (0.6605 / 0.6598 / 0.6614), so the absolute AUC
  is unaffected.
- **Axis ordering is not significant.** Curl leads at all four horizons, but by
  +0.0005 to +0.0015 with overlapping intervals. Consistent ordering, not a
  separated effect. We do not defend it.

---

## Part 2 — Raw regression (forecasting)

Direction probes freeze the encoders and fit a cheap linear model on top. They
answer "is the information present and readable." They do **not** show that a
trained network will find and use it. That is what this half tests: train a
network end-to-end to predict the next *k* frames of finger motion, and see
whether adding touch beats not adding it.

All runs: 300 epochs, corrected target, causal, 3 seeds. The metric is fingertip
MSE on the moving subset — lower is better.

### Attempt 1: train the tactile encoder from scratch

| | k=8 | k=16 |
|---|---|---|
| copy-zero baseline | 0.000312 | 0.000609 |
| **pose-only** | **0.000287** | **0.000521** |
| tactile + pose | 0.000356 | 0.000692 |
| shuffled tactile | 0.000479 | 0.001135 |

Two things are true here at once. Real touch **beats its scrambled twin by 26%
and 39%** — the network is genuinely extracting tactile signal. But
tactile+pose still **loses to pose-only**, and even to copy-zero.

The diagnosis: a ~500k-parameter tactile encoder was being trained from random
initialization against a 33k pose head on ~116k samples. Too much machinery,
not enough data. Not an information problem — a capacity problem.

### Attempt 2: reuse the encoder we already know contains the signal

Every positive result in this project came from a tactile encoder taken
**frozen** from the retrieval checkpoint. The forecaster had never been given
it. Doing that closes the gap:

| condition | trainable params | k=8 | k=16 |
|---|---|---|---|
| **frozen tactile + pose** | 66,045 | **0.000255** | **0.000461** |
| pose-only | 32,957 | 0.000287 | 0.000521 |
| frozen shuffled + pose | 66,045 | 0.000297 | 0.000547 |
| copy-zero | — | 0.000312 | 0.000609 |
| fine-tuned tactile | 574,205 | 0.000341 | 0.000638 |
| scratch tactile | 574,205 | 0.000356 | 0.000692 |
| scratch shuffled | 574,205 | 0.000479 | 0.001135 |

| comparison | k=8 | k=16 |
|---|---|---|
| frozen tactile vs pose-only | **−11.2%** | **−11.4%** |
| frozen tactile vs shuffled (capacity-matched) | **−14.3%** | **−15.7%** |
| frozen shuffled vs pose-only | +3.6% | +5.1% |
| frozen tactile vs copy-zero | −18.4% | −24.2% |

Seed std is 1e-6 to 3e-5; the condition clusters do not overlap.

**Why this is an argument and not just one number.** The seven conditions form a
single monotone ordering, identical at both horizons, and each step explains
itself:

- `scratch shuffled` is worst — capacity cost, no information.
- `scratch tactile` beats it — information helps, but capacity still sinks it.
- `fine-tuned` beats scratch — a good starting point helps.
- `frozen shuffled` sits *just above* pose-only (+3.6%/+5.1%) — that is the
  price of carrying the extra branch at all, with nothing useful in it.
- `frozen tactile` is best — the information, without the capacity cost.

The shuffled control has an **identical trainable parameter count** (66,045), so
the 14–16% gap cannot be capacity, regularization, or architecture. And the fact
that `frozen shuffled` is *worse* than pose-only shows real touch first has to
pay back the cost of the branch and only then turns a profit.

### What this does and does not license

**Can say:** touch improves finger-motion forecasting over a pose-only model of
the same task, and the gain is tactile content rather than added capacity.

**Cannot say yet:** that this holds for a new person. The frozen encoder came
from a retrieval run split at the **clip** level, meaning it trained on other
clips from the same participants it is evaluated on. It may have learned
participant-specific glove calibration.

How much this matters is worth being concrete about. The core comparison —
frozen tactile vs frozen shuffled, same encoder, same parameter count, only the
pairing scrambled — is not obviously manufactured by participant familiarity.
But we know from the retrieval side that participant leakage is *not* a small
effect: holding out whole participants drops retrieval from 45.5 to 28.3 mAP.
A third of that apparent performance was memorization. We should assume the
same risk applies here until measured.

The encoder trained participant-disjoint now exists, so this is a rerun, not new
work. It is the single highest-value experiment remaining.

---

## Where the two halves agree

The probes and the forecaster tell the same story, which is the strongest thing
about the current state:

- Probes said the information is **present** and linearly readable.
- Probes said the effect is **global**, not regime-specific.
- Probes said touch carries **magnitude**, not only direction.
- The forecaster's failure mode was **capacity, not information** — exactly what
  the probes predicted — and removing the capacity cost recovered the gain.

That is a mechanism, not a coincidence of numbers.

---

## Open items, in priority order

1. **Participant-disjoint forecasting — RUNNING.** The frozen-encoder
   experiment rerun with both the encoder and the split scene-disjoint. 18
   runs (pose-only is rerun too; the existing baseline is clip-split and no
   longer a valid comparison). This converts "touch helps" into "touch helps
   for someone we've never seen."

   Worth flagging: this was recorded as a one-flag rerun, and it was not. The
   regression pipeline had no scene-split option at all — the flag did not
   exist and the split function was called without it. Had the flag existed
   without being forwarded, the run would have quietly produced a clip-split
   result labelled as scene-split, which is precisely the confound the
   experiment exists to remove. It is now wired, tested, and in flight.

   One caveat for reading the results when they land: the motion threshold
   that defines the "moving" subset is the 25th percentile of the *train*
   split, so it shifts under a scene split (0.014044 → 0.013977). Comparisons
   *within* the new sweep are valid; comparing its MSEs directly against the
   clip-split table above is not.

2. **Retrieval confidence intervals.** Retrieval is our strongest claim and
   currently rests on a 3-seed standard deviation. The bootstrap script
   existed but should not have been run as written: it resampled individual
   windows rather than clips — the same error that would have made the probe
   intervals ~6x too tight — and resampled the gallery along with the queries,
   which matters because mAP here depends on gallery size. Both are fixed and
   pinned by tests; the runs are queued behind the sweep.
3. **Per-joint export at k=16**, where the curl effect is largest. Currently
   k=8 only.
4. **Joint optimization.** Freezing the encoder works, but whether joint
   training can beat it given more data or stronger regularization is unknown.

---

## Provenance

| claim | source |
|---|---|
| direction probe, all horizons | `results/results_probe_rigid_k{2,4,8,16}_T36.json` |
| decomposition | `results_probe_rigid_k8_DECOMP.json` |
| clip-clustered CIs | `results_rigid_ci_k8_T36.json` |
| participant hold-out | `results_probe_rigid_k8_SCENE{,_CLEANENC}.json` |
| held-out test | `results_probe_rigid_k{4,8}_TEST.json` |
| encoder seeds | `results_probe_rigid_k8_seed{0,1}.json` |
| raw-kinematics control | `results_rawpose_k8.json` |
| magnitude | `results_magnitude_k8.json` |
| subset analysis | `results_subset_discover_k8_*.json` |
| forecasting, scratch | cluster `logs/rr_{pose,tac,shuf}_k{8,16}_s{1,2,3}` |
| forecasting, frozen | cluster `logs/rf_{frz,frzshuf,ft}_k{8,16}_s{1,2,3}` |

A note on the AUC scale: these are AUCs over per-joint directional
classification, where 0.5 is chance. Baselines sit near 0.62–0.69, so a
marginal of +0.017 is a ~2-point move on a scale where the total available
headroom above the pose baseline is roughly 0.31. The shuffled controls land at
0.496–0.505 throughout, which is the sanity check that the pipeline is not
leaking.
