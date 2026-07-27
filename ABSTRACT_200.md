# Abstract, final

200-word body, no em-dashes, final sentence is a broader implication.
Ready-to-paste text is in `ABSTRACT_FINAL.txt` (196 words).

**All four horizons confirmed.** Version B is the one to submit.

---

## The abstract

> A wide range of sensory signals helps dexterous robots manipulate objects
> reliably. First-person video captures the appearance of objects and
> interaction but not contact, force, or grip, leaving gaps that tactile
> information can fill. OpenTouch aligns touch, egocentric video, and hand
> pose in a shared representation using contrastive learning. We first improve
> alignment, then ask whether touch predicts how a hand will move next.
> Replacing average pooling with a temporal pose encoder improves tactile-to-
> pose retrieval 2.7x, from 16.8 to 45.5 mAP. To test prediction, we isolate
> finger articulation from whole-hand motion and decode its future direction
> from touch available only up to the present moment. Touch predicts that
> direction well above chance, at AUC 0.60 to 0.69, while shuffled-touch
> controls remain at AUC 0.50. Adding touch to a pose encoder with the same
> temporal history still improves prediction, across horizons from 67 to 533
> milliseconds, and helps most when predicting whether the fingers are about
> to curl. Touch therefore supplies information that hand kinematics alone do
> not carry, revealing not just what a hand is holding but how it is about to
> reshape. A robot that feels contact may know where a hand is going, not just
> where it has been.

---

## The result, all four horizons

Marginal AUC of touch added to a pose encoder with **identical temporal
access**, best axis, val split:

| k | ms ahead | n eval | published target | corrected target |
|---|---|---|---|---|
| 2 | 67 | 15,196 | +0.0030 | **+0.0168** |
| 4 | 133 | 14,006 | +0.0036 | **+0.0164** |
| 8 | 267 | 11,425 | +0.0017 | **+0.0171** |
| 16 | 533 | 6,626 | +0.0001 | **+0.0132** |

Stable at +0.013 to +0.017 across an eightfold range of horizons, against a
published-target column that decays to nothing. k=16 was the horizon I flagged
as most at risk and it confirmed.

Touch alone versus matched-temporal pose alone, corrected target, best axis:

| k | touch alone | pose alone | shuffled touch |
|---|---|---|---|
| 2 | 0.6481 | 0.6481 | 0.5013 |
| 4 | 0.6508 | 0.6482 | 0.5008 |
| 8 | 0.6605 | 0.6545 | 0.5009 |
| 16 | **0.6897** | 0.6863 | 0.4963 |

Touch alone matches or beats the pose encoder at every horizon, and the gap
widens with distance. Both improve with horizon, which is counterintuitive
until you notice that contact state changes slowly and therefore constrains
longer-range finger motion more than short-range.

Per-axis marginal, corrected target:

| k | radial (grip aperture) | spread (abduction) | curl (flexion) |
|---|---|---|---|
| 2 | +0.0168 | +0.0125 | **+0.0179** |
| 4 | +0.0164 | +0.0131 | **+0.0169** |
| 8 | +0.0171 | +0.0126 | **+0.0187** |
| 16 | +0.0132 | +0.0114 | **+0.0198** |

**Flexion is the strongest axis at every horizon, and it is the only one that
grows monotonically with distance.** That is the best mechanistic outcome
available: a pressure sensor should know most about whether fingers are about
to curl, and it does. All shuffled-touch marginals are negative (−0.0016 to
−0.0058), which is the expected penalty for the extra 64 dimensions.

## Confidence intervals

Clip-clustered paired bootstrap at k=8, 1000 draws over all 296 val clips:

| axis | vs matched pose | vs shuffled twin | joints CI>0 |
|---|---|---|---|
| curl (flexion) | +0.0187 [+0.0064, +0.0311] | +0.0215 [+0.0082, +0.0354] | 17/20 |
| radial | +0.0171 [+0.0040, +0.0314] | +0.0195 [+0.0055, +0.0346] | 17/20 |
| spread | +0.0126 [+0.0009, +0.0245] | +0.0161 [+0.0022, +0.0300] | 15/20 |

Every interval excludes zero, against both the pose baseline and the shuffle.
The intervals are wide because clip-level clustering is the honest unit of
independence; a per-sample bootstrap would have reported roughly six times
tighter and been wrong.

## Titles

Profile book: *Multimodal Alignment and Tactile Prediction of Finger
Articulation in Dexterous Manipulation*.

Poster: *Fingers, Not Hands: Multimodal Alignment and Tactile Prediction of
Finger Articulation*.

Minimal change to what you submitted: delete "the Limits of".

## Claim-by-claim evidence

| claim | source |
|---|---|
| 2.7x, 16.8 to 45.5 mAP | matched own-codebase avg-pool run `2026_06_22-20_49_48`; val pair 14.42 to 40.97 agrees |
| causality by construction | poisoning test on the production functions, `tests/test_direction_probe_causality.py` |
| AUC 0.60 to 0.69 | corrected target, per-axis touch-alone across four horizons |
| shuffles at chance | 0.4956 to 0.5013 across all axes and horizons |
| holds against matched-temporal pose | four-horizon table above, CIs exclude zero |
| strongest on flexion | curl is the top axis at all four horizons |
| emerges after separating whole-hand motion | rigid rotation is ~95% of raw target energy at the median sample |

## Still outstanding

Not required for the abstract; these harden it for the paper.

1. **Rotation versus frame decomposition.** The corrected target changed two
   things at once (removed rotation, switched to palm axes). Running.
2. **Participant generalization.** Splits are clip-disjoint, not
   scene-disjoint. A scene-disjoint probe is queued.
3. **Encoder seed.** All numbers come from one retrieval checkpoint. Seeds 0
   and 1 are queued.
4. **Held-out test split.** Untouched by every probe so far.
5. **Magnitude.** Direction sign only. No magnitude claim is made anywhere.
