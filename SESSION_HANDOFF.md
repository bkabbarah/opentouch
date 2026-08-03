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

## 5. Paper assessment

**It is a measurement paper.** No new architecture, and §2.19f shows the
learned representation is not what drives the headline probe marginal. That is
fine for a workshop and fatal at a venue expecting method novelty.

**Two contributions, both actionable:**
1. The standard evaluation target is ~95% the wrong thing, and correcting it
   reverses a published conclusion. (§2.2–§2.12, §2.7)
2. **Contrastive retrieval quality is a poor proxy for downstream tactile
   utility** — the encoder-quality ladder is flat across mAP 10.8→28.2
   (§2.19c), and a random frozen encoder beats the pretrained one on
   forecasting (§2.19d). **This is already measured and has never been framed
   as a finding.** Zero new compute required.

Plus the downstream result: grip aperture, +0.11 AUC at k=8 and +0.14 at k=16
over a capacity-matched control, 4/4 partitions (§2.23, §2.24).

**Do not put on a poster or in a paper:** §2.19's 15% (partition-dependent,
and mostly measuring temporal pairing), the k=16 pose-only margin (CI includes
zero).

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
