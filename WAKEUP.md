# Wake-up checklist — morning of 2026-08-03

Everything below is copy-paste. Total hands-on time if all goes well: **~1 minute.**

---

## 0. Do this first, it is not optional

**Change your MIT password.** You pasted `bashark@mit.edu` and its password into
the chat last night. I did not use it — entering an account password is
something I won't do — but it is now sitting in a transcript, and if that is
your Kerberos credential it also covers the cluster and your email.

<https://atlas.mit.edu> → Password Reset. Takes a minute.

Nothing else in this file depends on it.

---

## 1. Check whether DexYCB downloaded itself (0 seconds if lucky)

DexYCB needs **no login at all** — the only thing blocking it last night was a
Google Drive anonymous-download quota, which resets on its own. A poller is
armed and retries every 30 minutes for 48 hours.

```bash
ssh bashark@mib.media.mit.edu 'tail -3 /tmp/dexycb_auto.log'
```

- Says **`SUCCESS: extracted N label files`** → go to step 2.
- Says **`quota still exceeded`** → nothing to do, it keeps trying. Skip to step 3.

## 2. If DexYCB landed, get the number (one command)

```bash
ssh bashark@mib.media.mit.edu 'bash /scratch/bashar/datasets/run_dexycb.sh'
```

This converts the labels and runs the diagnostic end to end. Read the `k=8`
median off the table it prints and compare to OpenTouch's **0.961**.

---

## 3. HO-3D — the one thing that genuinely needs you (~30 seconds)

HO-3D is the surest path to a second dataset, because its annotations contain
`handJoints3D` directly — no MANO forward pass. The catch is that it ships as
one 31.9 GB zip on OneDrive, and OneDrive's SharePoint migration blocks
headless downloads. A browser has to start it.

**Do this:**

1. Open <https://1drv.ms/f/s!AsG9HA3ULXQRlFy5tCZXahAe3bEV?e=BevrKO> in Chrome.
   (No sign-in. If it prompts you to, close the prompt — the folder is public.)
2. Click **`HO3D_v3.zip`** once to select it, then **Download** in the top toolbar.
3. Chrome starts a 31.9 GB download. Open `chrome://downloads`, **right-click
   the HO3D_v3 item → Copy link address**.
4. **Cancel** the Chrome download — you do not want it on your laptop.
5. Paste the URL into this, keeping the quotes:

```bash
ssh bashark@mib.media.mit.edu 'bash /scratch/bashar/datasets/run_ho3d.sh "PASTE_URL_HERE"'
```

That script downloads the zip to `/scratch`, extracts **only** the `meta/*.pkl`
annotations, **deletes the 31.9 GB zip**, converts to `(N,T,21,3)`, and runs the
diagnostic. Takes ~30–60 min unattended. Peak disk ~34 GB on `/scratch`, which
has 782 GB free — it never touches your home quota.

> If the URL has expired by the time you paste it (they last a few hours),
> just redo steps 1–4.

---

## 4. What I could not do, and why

| dataset | status |
|---|---|
| **DexYCB** | Fully automated, poller armed. No login needed. |
| **HO-3D** | Needs step 3. SharePoint blocks headless; no train-annotations-only archive exists. |
| **ARCTIC** | **Deprioritised.** Its cheap 215 MB drop is MANO *parameters*, not joints — needs a MANO forward pass, same problem the handoff flags for GRAB. The 18 GB "processed splits" have joints if you want it later. |

**The handoff's dataset ranking was wrong** and I'd revise it: HO-3D was listed
as "fastest path to a number" on the basis of an annotations-only download that
does not exist. Corrected order is **DexYCB → HO-3D → ARCTIC**.

---

## 4a. READ THIS: `aphz` finished and it overturns a §2.24 claim

Done 01:54, all 8 JSONs collected into `results/` and committed. I re-derived
every cell rather than reading its summary. **§2.24's "the effect STRENGTHENS
with horizon" does not survive the two new horizons.**

| k | ms | touch − shuffled | touch − pose-only |
|---|---|---|---|
| 2 | 67 | +0.112 · 4/4 | **+0.088** · 4/4 |
| 4 | 133 | +0.114 · 4/4 | +0.072 · 4/4 |
| 8 | 267 | +0.110 · 4/4 | **+0.054** · 4/4 |
| 16 | 533 | **+0.143** · 4/4 | +0.069 · 4/4 |

The pose-only margin is **largest at the shortest horizon**, dips at k=8, and
only partly recovers at k=16 — so the k=8→k=16 rise §2.24 leaned on is a
recovery, not a trend. It holds in 4/4 partitions, not one. Full analysis and
the reason (pose-only improves *faster* with horizon than touch does) is in
`HANDOFF.md` §2.26; §2.24 now carries a SUPERSEDED-IN-PART banner.

**Two things you should decide, not me:**
1. **k=2 and k=4 are one seed per cell** (k=8 has 2–3, k=16 has 2). The shape
   rests on single-seed estimates at the new horizons. A second seed is ~2
   GPU-hours and you have until ~Aug 7. I did **not** queue it — `probesweep`
   is using the GPUs and it closes reviewer gaps you already committed to.
2. The **capacity-matched contrast is flat and robust** (+0.110 to +0.114
   across 67–267 ms, positive in 16/16 cells, never below +0.062). If you don't
   want to spend the seeds, report that one and drop the horizon-shape claim.

Good news in the same data: **epoch selection now reproduces 25/25** — touch
selects epochs 2–12, pose-only 30–60, touch strictly earlier in every run
across four horizons and four partitions. That is the most reproducible result
in the forecasting line.

`probesweep` picked up the GPUs at 01:57 and is running its first three probes.

## 4a-bis. `probesweep` also finished — and it is good news for the paper

Done 03:25, 12/12, zero errors. All JSONs in `results/` and committed. Full
analysis in `HANDOFF.md` §2.27. Three headlines:

**1. Your headline probe table is now a three-encoder mean at four horizons**,
not seed 42 alone. The centrepiece — corrected target beats conventional —
holds at every horizon, ~4× at k=2/4/8. That is the main claim of the paper and
it is no longer single-seed, single-horizon.

**2. §2.19e was a k=8 artifact, and this is worth your attention.** It
concluded the learned representation isn't what makes touch predictive — but it
only tested k=8, which turns out to be the *one* horizon where pretrained and
random encoders coincide:

| k | pretrained | random | gap |
|---|---|---|---|
| 2 | +0.0165 | +0.0073 | **8.6 encoder-seed sd** |
| 4 | +0.0159 | +0.0074 | **7.0 sd** |
| 8 | +0.0138 | +0.0122 | 0.7 sd ← the only one tested |
| 16 | +0.0075 | +0.0118 | −1.1 sd |

At the horizons where the probe is strongest the learned representation
accounts for **~55% of the marginal**. **`SESSION_HANDOFF.md` §5 says this
finding is "fatal at a venue expecting method novelty" — that judgement rests
on §2.19e and should be revisited.** (Caveat: random encoder is 1 seed per
horizon; k=2 and k=4 agree closely with each other, but two more seeds at k=2
would settle it for ~30 GPU-min.)

**3. Participant-disjoint holds at all four horizons** — and is *larger* than
the clip-split marginal at k=8 (+0.0225 vs +0.0138) and k=16. The effect is
cleaner when participants are held out, not weaker.

One incidental correction: §2.6 calls seed 42 "the outlier high". True at k=8
and k=16 — but at k=2 and k=4 seed 0 is highest and seed 42 sits mid-pack.

## 4b. Preserve anything else the cluster produces

Both sweeps write to the **repo root**, where `results_*.json` is gitignored on
purpose. Nothing is tracked until you copy it in — and cluster access ends
~2026-08-07, so do this before then.

```bash
ssh bashark@mib.media.mit.edu 'cd ~/scratch/bashar/opentouch-gru && grep -l . results_aperture_k[24]_ss*.json results_probe_rigid_k*_SCENE_CLEANENC.json results_probe_rigid_k*_seed*.json results_probe_rigid_k*_RANDENC.json 2>/dev/null'
```

then, for whatever that lists:

```bash
ssh bashark@mib.media.mit.edu 'cd ~/scratch/bashar/opentouch-gru && cp results_aperture_k[24]_ss*.json results_probe_*.json results/ 2>/dev/null; git add results/ && git commit -m "results: aperture horizons k=2,4 and the probe paper sweep" && git push'
```

Do **not** trust any summary of these without re-deriving the contrasts from
the JSONs — that is what turned up corrections 9–11 last night.

---

## 5. What ran while you slept

- **`rotshare` finished and PASSED.** The agnostic diagnostic gives 0.961 at
  k=8 against the 0.957 reference — it agrees, so cross-dataset runs are
  trustworthy. Details and a caveat in `HANDOFF.md` §2.25.
- **The by-scene breakdown is a better result than the single number.** All
  **26 scenes above 90%** rigid share, range 90.5%–99.3%. That upgrades "we
  measured this once" to "this holds across every activity we have."
- **§2.22–§2.24 re-derived from the JSONs.** They hold. Three small doc
  corrections applied — see §3 of `HANDOFF.md`.
- **`aphz` still running**, ~8/24 runs done as of 23:43, on k=4 now.
  `probesweep` correctly queued behind it.
- **New: `scripts/load_public_hands.py`** + 17 tests. Suite is **260 passing**
  (was 243).
- **Rehearsed both runner scripts end to end against synthetic data, and it
  found three bugs that would each have hit you in the morning.** I built fake
  HO-3D and DexYCB trees and ran the real scripts on them rather than assuming
  the pieces composed.
  Because fix 1 touches the file that produced the 0.961 validation, I re-ran
  that validation on the patched code: the output JSON is **byte-identical** to
  the pre-patch run. The fix changes nothing on real data.
  1. **`rotation_share.py` rejected its own documented input.** Gap-splitting
     makes sequences ragged; `np.save` stores those as an *object* array, and
     `_as_tensor` only tested for `list`/`tuple`. The loader worked, the
     diagnostic worked, and the two-command pipeline between them died on
     command two — the only way the CLI is ever used. Fixed, with a
     round-trip regression test.
  2. **`run_ho3d.sh` derived the wrong dataset root** (`extracted/train/train`).
     Now derived from the `train/` directory's parent, verified against both
     plausible zip layouts.
  3. **`run_ho3d.sh` deleted the 31.9 GB zip before checking extraction
     worked.** If the unzip pattern had missed, you'd have re-downloaded 31.9 GB.
     It now verifies the layout and the `.pkl` count first, and *keeps* the zip
     with a diagnostic message on failure.
- **Caught one live bug in that loader before it could bite you.** I had it
  defaulting both datasets to MANO joint order. HO-3D is MANO — confirmed
  against its own `jointsMapManoToSimple`, which is byte-identical to my remap.
  **DexYCB is not.** Its joints come out of `manopth`, whose `ManoLayer.forward()`
  already applies that remap internally, and dex-ycb-toolkit passes the result
  through untouched — so DexYCB ships MediaPipe order and remapping it again
  would have scrambled it. Order is now per-dataset (`NATIVE_ORDER`), with a
  regression test. You don't need to pass any flag; the runner scripts are
  correct as written.

```bash
ssh bashark@mib.media.mit.edu 'tail -3 /tmp/aphz_master.log; tail -2 /tmp/probesweep_master.log'
```

### Contribution #2 is verified and ready to write — I did not write it

`SESSION_HANDOFF.md` §5 says the "retrieval quality is a poor proxy for
downstream tactile utility" finding is already measured and never framed. I
re-derived it from the JSONs; **it holds**, and it is stronger than §5 claims.
I stopped there rather than draft paper text you didn't ask for.

From `results_encoder_ladder.json` — retrieval mAP improves 2.6× and the
forecasting gain gets *worse*, not flat:

| encoder mAP | 10.81 | 16.07 | 20.26 | 25.19 | 28.24 |
|---|---|---|---|---|---|
| gain vs pose-only | **−8.97%** | −8.62% | −7.24% | −5.52% | **−6.90%** |

From the probe JSONs — a **random** frozen encoder's marginal AUC sits inside
the pretrained seed spread, not below it:

| encoder | marginal over matched pose |
|---|---|
| pretrained seed 42 (the outlier high) | +0.0171 |
| pretrained seed 0 | +0.0124 |
| pretrained seed 1 | +0.0120 |
| **random** | **+0.0122** |

Every number above came out of `results/`, not out of `HANDOFF.md`.

### One housekeeping note

To pull the new code onto the cluster I ran `git stash -u`, and the matching
`git stash drop` was blocked by a permission guard, so **`stash@{0}` is still
sitting in the cluster repo**. It is redundant — every file in it is either
restored in the working tree or now tracked in git, and `git stash show` on it
reports zero tracked changes. Clear it whenever you like:

```bash
ssh bashark@mib.media.mit.edu 'cd ~/scratch/bashar/opentouch-gru && git stash drop stash@{0}'
```

`stash@{1}` is older than this session and is not mine — leave it alone unless
you know what it is.
