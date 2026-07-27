# Abstract — required changes, verified on the cluster 2026-07-26

Every number below was pulled from actual run artifacts today, not from the
results-state document.

**READ SECTION 0 FIRST.** A result landed this afternoon that contradicts the
abstract's central claim.

---

## 0. The redundancy claim is an artifact of the target definition

The abstract currently says the tactile signal is "genuine yet redundant with
pose kinematics". That conclusion does not survive a corrected prediction
target, and the correction is not a matter of taste — the current target is
mislabelled.

**What the target actually is.** `decompose_world_delta` subtracts the wrist's
*translation* and nothing else. Whole-hand *rotation* — forearm pronation,
wrist flexion, fingers perfectly still relative to the palm — stays in it.
Nothing in the codebase canonicalizes rotation. Measured on val today
(`scripts/rigid_diag.py`, N = 18,864–28,296 per horizon):

| horizon | median rigid share | mean | >50% rigid | >80% rigid |
|---|---|---|---|---|
| k=2 | **0.952** | 0.830 | 87.9% | 73.1% |
| k=4 | 0.957 | 0.845 | 89.5% | 74.3% |
| k=8 | 0.957 | 0.848 | 90.1% | 74.5% |

A single rigid rotation about the wrist explains **~95% of the "articulation"
energy** at the median sample. The published probe was therefore mostly
decoding whole-hand re-orientation, not finger articulation.

**What happens when the rotation is removed.** Re-running the identical probe
— same frozen encoders, same causal machinery, same shuffle control, same
sample set — against the rigid-rotation-removed target expressed in
palm-anchored axes (`results_probe_rigid_k8_T36.json`, k=8):

| condition | long | flex | normal |
|---|---|---|---|
| tactile alone | 0.661 | 0.610 | 0.638 |
| shuffled tactile | 0.501 | 0.498 | 0.501 |
| matched-temporal pose | 0.655 | 0.609 | 0.639 |
| **pose + tactile** | **0.672** | **0.622** | **0.658** |
| pose + shuffled tactile | 0.652 | 0.606 | 0.636 |

Tactile's marginal contribution over the *matched-temporal* pose baseline:

| target | marginal AUC |
|---|---|
| current (`wrist_translation_removed`, world y) | **+0.0017** |
| rigid-removed, palm frame (`long`) | **+0.0171** |

**A tenfold difference, produced entirely by how the target is defined.** Three
further points:

- The signal is no longer confined to one axis. It is present on all three
  anatomical axes (+0.017 / +0.013 / +0.019), with the shuffle at chance
  (0.498–0.501) on all three.
- Tactile *alone* (0.661 on `long`) now **beats matched-temporal pose alone**
  (0.655). On the true articulation target tactile is competitive with pose,
  not a weak add-on.
- The mechanism is clear: pose kinematics extrapolate rigid rotation very well
  by continuation, so on a rotation-dominated target pose explains away
  tactile's contribution. Remove the component tactile was never about, and
  the contribution is real.

The harness is validated — it reproduces the published numbers on the old
target exactly (tactile-y 0.5984 vs published 0.598; pose_emb 0.6329 vs 0.633;
shuffled 0.5048 vs 0.505).

**Caveat, and it is the reason for the recommendation below.** This is one
horizon (k=8), one seed, no confidence intervals, and not confirmed on the
held-out test split. k=2, 4 and 16 are running now.

### Recommendation for today

**Do not submit the redundancy claim.** It is the one sentence I can now show
is an artifact. But do not rewrite the abstract around the new positive
either — it is hours old and single-horizon.

Submit the version in §5 below, which is true under *both* outcomes: it
reports the signal, reports that the conclusion is target-dependent, and
commits to neither "redundant" nor "uniquely predictive". If the remaining
horizons confirm, the full paper gets the stronger claim with proper controls.

---

## 1. The retrieval number

`13.4 → 45.5, "triples"` becomes **`16.8 → 45.5, 2.7×`**.

13.4 is the *paper's* avg-pool number; 45.5 is *your codebase's* GRU number.
Your matched avg-pool re-run is `logs/2026_06_22-20_49_48`, whose `tags.txt`
reads "original paper avg pool pose encoder, matches paper T->P mAP 16.76".
Final-epoch val confirms the ratio independently: **14.42 → 40.97 = 2.84×**.

A same-codebase replication is now running (both arms from one repo via the
new `--model ...-AvgPool` flag), together with the scene-disjoint pair below.

## 2. "future finger motion" → "future wrist-relative hand motion"

Forced by §0 for the *current* target. If you adopt the rigid-removed target
in the paper, "finger articulation" becomes accurate — but on the target the
submitted numbers come from, it is not.

## 3. Say the signal is one axis (on the current target)

AUC 0.58–0.60 is world *y* alone. Verified from your four result JSONs:

| k | tactile y | tactile x | tactile z | shuffled y |
|---|---|---|---|---|
| 2 | 0.575 | 0.504 | 0.499 | 0.505 |
| 4 | 0.583 | 0.507 | 0.501 | 0.505 |
| 8 | 0.598 | 0.516 | 0.503 | 0.505 |
| 16 | 0.605 | 0.517 | 0.508 | 0.508 |

True range is 0.575–0.605, so **0.57 to 0.60**. Per §0, the single-axis
concentration is itself a symptom of the rotation-dominated target.

## 4. Drop "encodes no magnitude"

The probe decodes sign only. The magnitude evidence is the regression, which
has two confounds: the correction branch receives the full 63-dim pose
(`pose_regression.py:236`), so zeroing the gate ablates *pose* capacity, not
tactile; and the shuffled control is *worse* than real tactile (0.009958 vs
0.008496), so the cost is branch capacity. Make it a scope statement.

## Verified clean — no change needed

**The "leaked view" sentence is true.** The k=16 `params.txt` shows
`sequence_length: 20`, no `causal` or `min_history` field, at
`git_commit: 69a5523` — which predates the causal fix. Those runs are
noncausal: tactile saw frames including the target's own timeframe and still
failed to help.

**Sample-set parity holds.** Copy baseline byte-identical across all three
k=16 configs (0.006717, moving 4699/6288, same threshold).

---

## 5. Revised abstract — submit this

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
> above-chance directional signal (AUC 0.57 to 0.67 depending on the motion
> component, chance 0.50, shuffled controls at chance throughout). We then
> show that how much *unique* value touch adds is governed by two choices
> that are easy to get wrong. Against a single pose frame touch appears to add
> up to +0.033 AUC, but against a pose encoder given the same temporal history
> the gain shrinks by an order of magnitude — most of the apparent benefit was
> temporal access, not contact. And because whole-hand rotation accounts for
> roughly 95% of the motion target as conventionally defined, the measured
> contribution depends heavily on whether that rotation is removed before
> asking what touch predicts. We quantify both effects and a temporal leak
> that would separately have faked a positive. Our results suggest that claims
> about a modality's predictive value are only interpretable alongside matched
> temporal baselines and an explicitly decomposed prediction target.

~250 words. Commits to: the retrieval win, a real above-chance signal, the
baseline effect, the target-definition effect, and the methodological
conclusion. Commits to neither "redundant" nor "uniquely predictive" — so it
stays true whichever way the remaining horizons land.

### If you would rather stay closer to your original

Keep your text and make three edits: `16.8 to 45.5` and "2.7-fold";
"finger motion" → "wrist-relative hand motion"; and replace

> "once pose is given the same temporal history, touch adds almost nothing (a
> marginal gain near +0.003) and encodes no magnitude: the signal is genuine
> yet redundant with pose kinematics."

with

> "once pose is given the same temporal history, the marginal gain drops by an
> order of magnitude — and it depends further on how the motion target is
> decomposed, since whole-hand rotation accounts for roughly 95% of that
> target as conventionally defined. We decode direction only, not magnitude."

That removes the claim I can show is an artifact while changing the least.

---

## Minor

"Robotic Manipulation" in the title and "dexterous robots" in the opening
frame the work around robots, but every experiment is on human hand data.
Reads as motivation, so defensible — "Dexterous Manipulation" is safer.
