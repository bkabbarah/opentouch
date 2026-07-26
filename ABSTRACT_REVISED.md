# Abstract — required changes and revised text

Every number below was pulled from the cluster today and verified against the
actual run artifacts, not from the results-state document.

---

## The four changes

### 1. The retrieval number (you already knew)

`13.4 → 45.5, "triples"` becomes **`16.8 → 45.5, 2.7×`**.

13.4 is the *paper's* avg-pool number; 45.5 is *your codebase's* GRU number.
Your own matched avg-pool re-run is `logs/2026_06_22-20_49_48`, whose
`tags.txt` reads "original paper avg pool pose encoder, matches paper T->P mAP
16.76". Final-epoch val for the matched pair confirms the ratio
independently: **14.42 → 40.97 = 2.84×**. Test: 16.76 → 45.46 = 2.71×.

### 2. "the direction of future finger motion" → "future wrist-relative hand motion"

**This is the one that would have cost you the paper.** `decompose_world_delta`
subtracts the wrist's *translation* and nothing else. Whole-hand *rotation* —
forearm pronation, wrist flexion, fingers perfectly still relative to the palm
— is still in the target. Nothing in the codebase canonicalizes rotation.

Measured on the val split today (`scripts/rigid_diag.py`, N=18,864–28,296 per
horizon):

| horizon | median rigid share | mean | >50% rigid | >80% rigid |
|---|---|---|---|---|
| k=2  | **0.952** | 0.830 | 87.9% | 73.1% |
| k=4  | **0.957** | 0.845 | 89.5% | 74.3% |
| k=8  | **0.957** | 0.848 | 90.1% | 74.5% |

A single rigid rotation about the wrist explains ~95% of the "articulation"
energy for the median sample. Calling it finger motion is not supportable. The
phrase "wrist-relative hand motion" is exactly what was measured and is true
regardless of how the follow-up lands.

### 3. Report that the signal is one-axis

The AUC 0.58–0.60 figure is the **world y axis alone**. Verified from
`results_probe_uniqfair_k{2,4,8,16}_T36.json`:

| k | tactile y | tactile x | tactile z | shuffled y |
|---|---|---|---|---|
| 2  | 0.575 | 0.504 | 0.499 | 0.505 |
| 4  | 0.583 | 0.507 | 0.501 | 0.505 |
| 8  | 0.598 | 0.516 | 0.503 | 0.505 |
| 16 | 0.605 | 0.517 | 0.508 | 0.508 |

x and z are at chance everywhere. Presenting 0.58–0.60 without saying it is a
single axis overstates the signal — and given (2), the axis concentration is
itself evidence the probe is reading whole-hand re-orientation. Note the range
is 0.575–0.605, so **0.57 to 0.60** is the accurate rounding.

### 4. Drop "encodes no magnitude"

The probe decodes **sign only**. The magnitude evidence is the regression MSE
result, and that result has two confounds that make it unusable as stated:
the correction branch receives the full 63-dim pose (`pose_regression.py:236`),
so zeroing the gate ablates real pose capacity rather than tactile; and the
shuffled control is *worse* than real tactile (0.009958 vs 0.008496), which
means the cost is branch capacity, not tactile content. Replace the claim with
a scope statement.

---

## Two things that came back CLEAN — no change needed

**The "leaked view" sentence is verified true.** I read the `params.txt` for
the k=16 runs: `sequence_length: 20`, `horizon_k: 16`, no `causal` or
`min_history` field, at `git_commit: 69a5523` — which predates the causal fix
(`9bb8a6f`). Those runs *are* noncausal. The tactile encoder saw the full
20-frame window including frames (t, t+16], i.e. the target's own timeframe,
and still failed to improve prediction. Keep the sentence.

**Sample-set parity holds.** The copy baseline is byte-identical across all
three k=16 configs (0.006717, moving subset 4699/6288, threshold
0.02397303096950054), so pose-only / tactile / shuffled were scored on exactly
the same population and the comparison is valid.

---

## Recommended addition

The `+0.033 vs +0.003` contrast is your strongest and most general result and
it is currently invisible — only `+0.003` appears. Verified marginal y-AUC:

| k | vs single-frame pose | vs matched-temporal pose | shuffled control |
|---|---|---|---|
| 2  | +0.0323 | +0.0030 | −0.0013 |
| 4  | +0.0261 | +0.0036 | −0.0014 |
| 8  | +0.0227 | +0.0017 | −0.0024 |
| 16 | +0.0148 | +0.0001 | −0.0038 |

An order-of-magnitude difference produced purely by which baseline you pick.
That generalizes to any multimodal ablation claim, and it is the contribution
that clears the "fundamental and widely usable" bar without being an
off-the-shelf sequence model or an anatomical prior.

---

## Revised abstract

> A wide range of sensory signals helps dexterous robots manipulate objects
> reliably. First-person video captures the appearance of objects and
> interaction but not contact, force, or grip, leaving a gap that tactile
> information can fill. The OpenTouch framework aligns three modalities of
> manual interaction (touch, egocentric video, and hand pose) in a shared
> representation using contrastive learning. We first improve alignment, then
> investigate whether touch predicts where a hand will move next. Replacing
> average pooling with a temporal pose encoder improves tactile-to-pose
> retrieval 2.7-fold (16.8 to 45.5 mAP). To test prediction, we decode the
> direction of future wrist-relative hand motion from touch available only up
> to the present moment, against shuffled controls. Touch carries a real,
> above-chance signal, but it is confined to a single motion axis (AUC 0.57 to
> 0.60, chance 0.50; the other two axes sit at chance). The apparent size of
> that signal depends almost entirely on the baseline it is measured against:
> compared to a single pose frame, touch appears to add up to +0.033 AUC, but
> compared to a pose encoder given the same temporal history it adds +0.003 or
> less. The signal is genuine yet redundant with pose kinematics; we decode
> direction only, not magnitude. Even when touch is handed a leaked view of
> future frames it should not see, it still fails to improve transition
> prediction, making the conclusion conservative. This suggests touch is more
> suitable for established, present-tense manipulation applications like
> contact sensing than for future prediction.

**Optional, if you want to foreground the methodology** — replace the last
sentence with:

> Both artifacts we identify — a temporal leak and an unmatched baseline —
> would each have produced a false positive, suggesting that matched-temporal
> controls are necessary before attributing predictive value to any added
> modality.

Word count: ~245, close to your original ~230.

---

## Minor, your call

"Robotic Manipulation" in the title and "dexterous robots" in the opening
frame the work around robots, but every experiment is on human hand data with
no robot in the loop. It reads as motivation rather than a claim, so it is
defensible — but a reviewer may flag it. "Dexterous Manipulation" would be
safer.
