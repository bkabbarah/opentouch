# Session handoff — 2026-08-02

`HANDOFF.md` is the results-of-record and is current through §2.24. This file
covers only what a *fresh session* needs: what is in flight, what to do next,
and the how-to for tasks not yet started.

---

## 1. Situation

- MSRP ends in ~5 days; GPU access ends with it. **The scarce resource is GPU
  time, not calendar time** — the remaining high-value work (cross-dataset
  rotation diagnostic) is CPU/data work and survives losing the box.
- Target is a **workshop paper**, not a conference paper. Assessment in §5.
- The PI is not currently engaged. Several findings are corrections to his
  group's published direction, so he should hear them from Bashar directly
  rather than from a draft.

## 2. In flight (started 2026-08-02 evening)

| tmux session | what | expect |
|---|---|---|
| `aphz` | aperture k=2, k=4 × 4 partitions × 3 arms | completes the horizon curve |
| `probesweep` | 12 probe runs (queued behind `aphz`) | see below |
| `rotshare` | rotation-share validation + per-scene breakdown | see below |

**`probesweep` closes three gaps a reviewer will find:**
1. The headline probe table (§2.2) is encoder **seed 42 only** at k=2/4/16 —
   and §2.6 shows seed 42 is the outlier high. Seeds 0 and 1 at those horizons
   make every cell a three-encoder mean.
2. The **random-encoder control** exists only at k=8 (§2.19e).
3. **Participant-disjoint** exists only at k=8.

**`rotshare`** validates `scripts/rotation_share.py` against this project's own
95.7% at k=8. **If it disagrees, nothing that script says about another dataset
can be trusted** — check this before building on it. It also breaks the share
down by scene across 26 sessions, which is a free generalization check.

Check with:
```bash
ssh bashark@mib.media.mit.edu 'cd ~/scratch/bashar/opentouch-gru && tail -5 /tmp/aphz_master.log /tmp/probesweep_master.log /tmp/rotshare.log'
```

## 3. The next task: cross-dataset rotation diagnostic

**Why it matters.** The core claim — *the standard wrist-relative target is
~95% whole-hand rotation, and correcting it reverses a published conclusion* —
is currently one number on one dataset. The diagnostic needs **only hand-joint
annotations**: no tactile, no encoder, no images. Running it on 2-3 public
datasets converts "in our data" into "in hand-motion data generally", which is
the single biggest upgrade available to this paper.

**Access is already granted** (account registration only, no forms).

### Disk — read before downloading

| location | reality |
|---|---|
| `~` (`/home/bashark`) | root filesystem, **50 GB quota, 32 GB used, ~19 GB free** |
| `/scratch/bashar/` | the data volume; 191 GB already used; 782 GB free of 28 TB (98% full) |

**Use `/scratch/bashar/datasets/`. Never bulk-download into `~`** — it is the
machine's system disk and the quota is nearly full. Annotations-only downloads
are small enough that this is not a problem.

### Order of attack

1. **HO-3D** — fastest path to a number. Its annotations contain
   `handJoints3D` directly (21 joints, 3D positions), so **no MANO forward
   pass is needed**. 55 temporal training sequences.
   <https://github.com/shreyashampali/ho3d>
2. **DexYCB** — ships labels separately from RGB-D. <https://dex-ycb.github.io>
3. **ARCTIC** — granular download script in the repo; request annotation
   splits only. <https://github.com/zc-alexfan/arctic>
4. **GRAB** — cleanest data but MANO params need a forward pass through the
   model; lowest priority. <https://grab.is.tue.mpg.de>

**Filter for all of them: annotations only, never images.** Bashar has not
done downloads like this before — walk through it rather than handing over
commands.

### What the loader must produce

`scripts/rotation_share.py` takes `(N, T, 21, 3)`, joint 0 = wrist. Non-wrist
joint ordering only has to be internally consistent. Then:

```bash
python scripts/rotation_share.py --npy poses.npy --fps 30 --name HO3D --out report.json
```

**Sequences must be temporal.** Single-frame datasets (FreiHAND) are unusable —
there is no delta to decompose.

## 4. Traps this project has already hit

Each cost real time. Do not re-learn them.

- **Never trust a claim in a doc without re-deriving it from the result JSON.**
  Three "just rerun the script" tasks each turned out to need code that did not
  exist, or would have silently produced a wrong answer.
- **`--split-group-by` silently defaulting to `clip`.** A scene-trained
  retrieval model scored **72.09 mAP instead of 28.31** because no checkpoint
  records the field. Always state it explicitly.
- **Selecting on the eval set.** Picking the best epoch on val and reporting
  val metrics inflated a control's R² from −0.003 to +0.05. Select on val,
  report on test.
- **Reading the capacity-matched contrast alone.** FiLM's frz-vs-frzshuf gap
  looked *better* (−19.7% vs −14.9%) purely because its shuffled control got
  worse. Always read it together with shuffled-vs-pose-only.
- **Single-partition results.** §2.19 looked clean until the participant split
  was redrawn and it flipped sign. Nothing is believed until it survives
  redrawn partitions.

## 5. Paper assessment — rewritten 2026-08-03

Everything below was re-derived from the JSONs in `results/`. Section numbers
refer to `HANDOFF.md`.

**It is still a measurement paper**, but the earlier line that it is "fatal at
a venue expecting method novelty" rested on §2.19e and **no longer holds as
stated** — see risk 1 below.

### The lead contribution is now cross-dataset

**The standard wrist-relative target is majority whole-hand rotation, on every
dataset measured.** Not a property of one capture rig:

| dataset | k=8 median rigid share | independent units |
|---|---|---|
| OpenTouch | 96.1% | 26 scenes |
| DexYCB | 89.6% | 300 takes, 3 subjects |
| HO-3D | 82.1% | 9 takes |

Three things make this much stronger than the single 95.7% it replaces:

- **It replicates on data we did not collect** (§2.28, §2.30), which was §3's
  stated single biggest available upgrade. It is done.
- **The magnitude tracks transport-vs-articulation**, and that ordering was
  *predicted in writing before HO-3D was measured*. OpenTouch is a walking
  body, DexYCB is seated grasping, HO-3D is a stationary arm. A prediction
  that held on new data is worth more than any single number.
- **It survives the obvious attack.** The rigid share *rises* with motion
  magnitude on all three datasets (§2.29), so near-still frames drag the
  published figure DOWN and every motion filter strengthens it. Above-median
  motion at k=8: 98.0% / 94.0% / 91.5%.

**Stop saying "~95%".** That is OpenTouch's number. Say: *majority whole-hand
rotation on every dataset measured — 82%, 90%, 96% of target energy at 267 ms
— with the magnitude tracking how much the hand is transported versus
articulated in place.* Weaker, and far harder to dismiss.

The target-correction consequence is unchanged and now rests on a
three-encoder mean at four horizons (§2.27): corrected beats conventional
~4x at k=2/4/8.

### Contribution 3 (NEW, 2026-08-04) — the sharpest thing in the project

**~88% of what a 256-taxel array buys is available from its sum.** §2.34.

Destroy every spatial pattern in the tactile map, keep only per-frame total
pressure, and the benefit on grip aperture barely moves: +0.0700 vs +0.0776 at
k=2, +0.0544 vs +0.0632 at k=8, scalar-only positive in **4/4 partitions at
both horizons**. The spatial residual is +0.008 AUC, positive in 3/4 — below
this project's own bar.

State it as: *the tactile contribution is a one-dimensional contact-pressure
time series; where on the hand the pressure falls does not measurably matter.*
Not "a single number" — the encoder still sees 36 frames of it.

**Why this is the strongest contribution.** It explains three results that
previously had no common cause: the 2–4 epoch peak reproducing 25/25, a random
encoder matching a pretrained one, and derangement destroying the effect. All
three follow from a 1-D temporal signal and none require spatial structure.
And it is a claim about tactile sensing for manipulation, not about this
pipeline — with a direct implication for sensor design and for what tactile
pretraining could be expected to learn at all.

It also **subsumes contribution 2**: retrieval pretraining buys nothing (§2.32)
because there is almost nothing spatial for it to learn.

Limits: two horizons, one seed per cell, OpenTouch only, and the aperture
target is rotation-invariant by construction.

### Contribution 2, and the risk attached to it

**Contrastive retrieval quality is a poor proxy for downstream tactile
utility.** Verified against the JSONs and *stronger* than previously written:
the encoder ladder does not merely fail to improve, it **degrades** — mAP
improves 2.6x (10.81 → 28.24) while the forecasting gain goes −8.97% → −6.90%
(`results_encoder_ladder.json`).

> **RISK 1 — FULLY RESOLVED 2026-08-03 22:07. Contribution 2 SURVIVES, in a
> better form. See §2.32.** On grip aperture — AUC, 4 horizons x 4 partitions,
> each encoder with its own control — a random frozen encoder captures the
> *entire* downstream benefit. Pretrained minus random is **−0.0020 pooled,
> positive in 8/16 cells**, a coin flip and 54x smaller than the +0.110
> headline tactile effect. Meanwhile **both** encoders beat pose-only in
> **16/16** cells. So: the tactile signal is worth ~+0.07 AUC, the learned
> representation is worth ~0.00.
>
> **State it as:** *a randomly initialised frozen encoder captures the entire
> downstream benefit; contrastive retrieval pretraining adds nothing measurable
> on the task where tactile demonstrably helps.* Do NOT state it as "random
> beats pretrained" — §2.31 shows that was an instrument artifact.
>
> §2.32 also found a methodological problem that reaches back into §2.19,
> §2.22, §2.23 and §2.26: **the shuffled control is not encoder-neutral.**
> Deranging a pretrained encoder's input costs up to 0.074 AUC against
> pose-only; deranging a random encoder's costs ~0. So capacity-matched gaps
> are inflated by however bad each arm's own control is, and comparing them
> across encoders is invalid. **The advice below to lead with the
> capacity-matched contrast is withdrawn — lead with touch − pose-only.**
>
> The superseded detail, kept because the failure mode is instructive: the
> second leg of this contribution was "a random frozen encoder beats the
> pretrained one on forecasting" (§2.19d), k=8, one seed. It was run at k=2
> and k=4. **The delta-MSE instrument cannot answer the question at all**: its
> two contrasts disagree at k=4 and k=8 (capacity-matched favours pretrained
> at every horizon; vs-pose-only favours random at k=4 and k=8), the two
> shuffled controls are not equivalent — deranging a pretrained encoder's
> input costs 4.6–9.0% against pose-only while deranging a random one's costs
> nothing, which is exactly §2.22's FiLM trap — and at k=4 the difference sits
> inside the logs' 6-decimal rounding.
>
> **Neither §2.19d nor a reversal of it is supported.** Do not cite either.
> Contribution 2's second leg currently has no usable evidence, and its first
> leg (the encoder ladder, §2.19c) is the same instrument on the same target.
>
> **This is a live threat to contribution 2 as a whole.** What can settle it
> is `scripts/aperture_randenc.sh` — the same control on grip aperture, scored
> by AUC on a scalar with a real sign, 4 horizons x 4 partitions, each encoder
> with its own shuffled control. Running as of 2026-08-03 17:11. If that shows
> no gap either, **drop contribution 2** and stand on contribution 1 plus the
> aperture result, both of which are unaffected.

**Correspondingly, the "no method novelty" verdict is now wrong as stated.**
§2.19e concluded the learned representation is not what makes touch
predictive; §2.27 shows that conclusion came from the one horizon where
pretrained and random coincide. At k=2 and k=4 the learned representation
accounts for **~55%** of the marginal. The honest claim is horizon-dependent:
*the learned representation matters at short horizons and washes out by 267
ms* — which is more interesting than either absolute version.

### Downstream result: grip aperture

Restated from §2.26's full horizon curve. **§2.24's "the effect strengthens
with horizon" is wrong** and §2.24 now carries a superseded banner.

| contrast | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|
| touch − shuffled (capacity-matched) | +0.112 | +0.114 | +0.110 | +0.143 |
| touch − pose-only | **+0.088** | +0.072 | +0.054 | +0.069 |

All 4/4 partitions at every horizon.

**Lead with touch − pose-only, not the capacity-matched contrast** (revised
2026-08-03; see §2.32). The capacity-matched number is larger (+0.110 to
+0.143) but sits on a shuffled control that is itself **0.024–0.074 AUC worse
than pose-only** — at k=16 roughly half the gap is the control failing rather
than the arm succeeding. Report it alongside its branch cost, never alone.
The pose-only margin is largest at the *shortest* horizon and dips at k=8 — do
not describe it as growing with lead time.

Also solid and cheap to state: **epoch selection reproduces 25/25.** Touch
selects epochs 2–12, pose-only 30–60, touch strictly earlier in every run
across four horizons and four partitions.

### Do not put in a paper

- §2.19's 15% — partition-dependent, sign-flips, mostly temporal pairing.
- **The k=16 probe marginal** (+0.0075) — encoder-seed sd is 0.0040, over half
  the mean. §2.27 gives this independent support.
- **The aperture horizon *shape***, unless the seeds get run — k=2 and k=4 are
  one model seed per cell against 2–3 at k=8/k=16 (risk 2).
- "Seed 42 is the outlier high" without qualification — true at k=8 and k=16,
  false at k=2 and k=4 where seed 0 is highest (§2.27).
- HO-3D's pooled figure without its range — 9 takes only, spread 49%–100%.

### Open risks, in priority order

1. **Forecasting random-encoder control is k=8-only, one seed.** Directly
   threatens contribution 2. ~1 GPU-hour. Do this first.
2. **Aperture k=2/k=4 are single-seed.** Threatens the horizon shape, not the
   headline. ~2 GPU-hours.
3. **Probe random-encoder is one seed per horizon.** k=2 and k=4 agree closely
   (44%, 47% recovery) so this is the least likely to move. ~30 GPU-min.

GPU access ends ~2026-08-07. All three fit. Nothing else needs the box.

**Venues** — verify current CFPs, these shift: NeurIPS workshops (~Sept/Oct
deadlines) are the best fit; ICRA (~Sept) plausible; ICLR workshops are
~February, not September.

## 6. State

- Branch `audit/frames-subsets-scene-split`, pushed. Cluster is on it.
- **All 115 result JSONs are now tracked in `results/`** — the provenance
  chain survives losing the cluster. Root-level `results_*.json` stays ignored
  so `git_dirty` keeps its meaning.
- 243 tests passing (`python -m pytest tests/ -q`, `PYTHONPATH=src`).
- Poster (`Kabbarah_Bashar_2026_MSRP_poster_MASTER (2).pptx`) is **final and
  verified** — do not change it. The forecaster is correctly absent from it.

**Before access ends:** decide whether to preserve `p2t_scene_gru` and
`p2t_scene_gru_ss{1,2,3}` checkpoints. Each took ~6.5h and cannot be
regenerated without the box.
