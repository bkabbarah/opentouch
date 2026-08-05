# VALIDATION — independent adversarial re-derivation of the results of record

Date: 2026-08-04. Branch `audit/frames-subsets-scene-split`, clean tree.
Method: every headline statistic below was **recomputed from the committed
JSONs in `results/`**, never read off `HANDOFF.md` or `SESSION_HANDOFF.md`;
core method code was then reviewed line by line. Claimed numbers are shown
next to computed numbers everywhere. Nothing was edited except the creation of
this file.

Verdict in one sentence: **the results of record reproduce from the JSONs
essentially exactly — every headline number checked out — with a short list
of minor doc-vs-JSON discrepancies (§5), several unstated caveats and code
fragilities worth fixing before review (§4), and no error that moves any
conclusion.**

## 1. Executive summary

The project's own rule ("never trust a doc claim without re-deriving it from
the JSON") was applied to the project. All twelve findings listed in the audit
scope re-derive correctly: rotation share on three datasets, motion
sensitivity, the four-horizon aperture result, the probe paper sweep, the
delta-MSE random-encoder analysis, the aperture random-encoder sweep, the
scalar ablation, the STAG positive control, the paired probe seeds, and the
retrieval/GRU numbers with their bootstrap CIs. The corrections already
recorded in HANDOFF §3 (9–11) were independently confirmed — including the
spread 12/20 CI count and the 3-seed cell at partition 42. The discrepancies
found by this audit are small: a mis-stated decile-monotonicity count in
§2.29, an internally inconsistent "resolved at k=2" cell in §2.31's precision
table, a generous "~4x" at k=4, one unpinned synthetic figure (22.4%), and two
caveats that are true but stated nowhere (the rotation share is an upper bound
by construction; the clip-split avg-pool retrieval baseline is a single run).

| finding | status | evidence file(s) | caveat |
|---|---|---|---|
| Rotation share OpenTouch 96.1% k=8 | **VERIFIED** (0.9608) | `results_rotation_share_opentouch.json` | upper-bound framing not stated anywhere (§5-D6) |
| Rotation share DexYCB 89.6% (3 subj) | **VERIFIED** (0.8961) | `results_rotation_share_dexycb3.json`, `_bytake`, `_dexycb` | 3 of 10 subjects |
| Rotation share HO-3D 82.1% clean / 86.8% contaminated | **VERIFIED** (0.8215 / 0.8677) | `results_rotation_share_ho3d_clean.json`, `_ho3d.json` | 9 takes; k=2 take range 49.0–100% confirmed |
| rigid_diag vs rotation_share validation pair | **VERIFIED** (0.9572 vs 0.9608 k=8) | `results_rigid_diag_val.json` | both import the same `rigid_fraction`; validates loading only (doc says so) |
| Motion sensitivity (3 datasets) | **VERIFIED-WITH-CAVEATS** | `results_motion_sensitivity_{opentouch,dexycb,ho3d}.json` | monotonicity sentence overstated — see §5-D1 |
| Aperture k=2..16, 4 partitions | **VERIFIED** (all 16 cells) | `results_aperture_{earlystop,splitseed_ss*,k2_ss*,k4_ss*,k16_ss*}.json` | k=2/k=4 one seed per cell (stated) |
| Probe sweep 3-encoder means, k=8 artifact, participant-disjoint, inversion | **VERIFIED** | `results_probe_rigid_k{2,4,8,16}_{T36,seed0,seed1,RANDENC,SCENE_CLEANENC}.json` | "~4x" is 3.3x at k=4; RANDENC runs unregenerable (pre-seed) |
| Delta-MSE randenc: instrument cannot answer | **VERIFIED-WITH-CAVEATS** | `results_randenc_horizons.json` | k=2 "resolved" cell wrong by its own criterion (§5-D2); k=8 row cluster-only |
| Aperture randenc: pretrained−random ≈ −0.0020, 8/16 | **VERIFIED** | `results_aperture_randenc_*.json` + summary + pretrained aperture files | — |
| Scalar ablation 90%/86% retention | **VERIFIED** | `results_aperture_scalar_k{2,8}_ss*.json` | residual 3/4, below project's own 4/4 bar (stated) |
| STAG control 27.8%→5.0% vs 3.7% | **VERIFIED** (5.4% retained) | `results_stag_scalar_control.json` | floor, not reproduction (stated loudly) |
| Paired probe seeds +0.0091 sd 0.0012, 3/3 | **VERIFIED** | `results_probe_k2_RND_es{0,1,2}_{none,scalar}.json`, `..._k8_RANDENC_SCALAR.json` | k=8 pair unseeded (stated) |
| Shortcut analysis (24–42% vs ~96%) | **NOT LOCALLY VERIFIABLE** | no `results_shortcut_*` in repo; script + commit `da09239` only | withdrawn metric confirmed removed in code |
| Retrieval 16.76→45.46 (2.71x), scene 6.44→28.31 (4.40x), CIs disjoint | **VERIFIED** | `results_retrieval_*.json`, `results_bootstrap_*.json`, `results_p2t_*.json` | avg-pool clip baseline is a single run (§5-D6) |

## 2. Test suite

```
PYTHONPATH=src python -m pytest tests/ -q
282 passed, 9 warnings in 42.92s
```

No failures, no errors, no skips. (Docs cite historical counts 243/260; the
suite has since grown.) Warnings are third-party deprecations (`chumpy`/scipy).

## 3. Per-finding validation detail (computed vs claimed)

### 3.1 Rotation share

Computed medians (claimed in parentheses where they differ in precision):

| k | OpenTouch | DexYCB (3 subj) | HO-3D clean | HO-3D contaminated |
|---|---|---|---|---|
| 2 | 0.9572 (95.7) | 0.8148 (81.5) | 0.7430 (74.3) | 0.8009 (80.1) |
| 4 | 0.9612 | 0.8578 | 0.7825 | 0.8364 |
| 8 | **0.9608 (96.1)** | **0.8961 (89.6)** | **0.8215 (82.1)** | 0.8677 (86.8) |
| 16 | 0.9559 | 0.9323 | 0.8570 | 0.8981 |

All match. Supporting detail all re-derived:
- OpenTouch by-scene, k=2: 26 scenes, all above 0.90; lowest `fablab_ml_p1`
  0.9046, highest `eat_ygf_p2` 0.9935 (claimed 90.5% / 99.3%). ✓
- DexYCB per-subject k=8: 0.8981 / 0.9201 / 0.8572 (claimed 89.8/92.0/85.7);
  subject-01 alone 0.8142/0.8613/0.8981/0.9258 (claimed 81.4/86.1/89.8/92.6). ✓
- DexYCB per-take k=8: 300 takes, 261 > 80%, 292 > 70%, range 0.537–0.985
  (claimed 261/300, 292/300, 53.7–98.5%). ✓
- HO-3D clean: 44 sequences / 9 takes; k=2 per-take range 0.4895 (`ShSu10`) to
  1.0000 (`SMu40`), median across takes 0.6924 (claimed 49.0–100.0%, 69.2%). ✓
- Validation pair: `rigid_diag` val 0.9523/0.9569/0.9572/0.9555 vs agnostic
  path 0.9572/0.9612/0.9608/0.9559 — 0.004 apart at k=8 as claimed. ✓
  (§2.25's caveat that both share the same `rigid_fraction` and this validates
  loading only is accurate and stated.)

### 3.2 Motion sensitivity

Computed (median / mean), matching §2.29–§2.30 exactly:

| filter | OpenTouch k=8 | DexYCB k=8 | HO-3D k=8 |
|---|---|---|---|
| all samples | 0.9608 / 0.8532 (96.1/85.3) | 0.8961 / 0.8162 (89.6/81.6) | 0.8215 (82.1) |
| above median (p50 floor) | 0.9799 / 0.8935 (98.0/89.4) | 0.9400 / 0.8835 (94.0/88.4) | 0.9149 (91.5) |
| top decile (p90 floor) | 0.9939 / 0.9640 (99.4/96.4) | 0.9709 / 0.9426 (97.1/94.3) | 0.9380 (93.8) |

Decile extremes verified: OT k=2 0.833→0.990, OT k=8 0.906→0.994, DexYCB k=2
0.529→0.932, DexYCB k=8 0.648→0.971. The trend rises with motion everywhere.
**However §2.29's monotonicity sentence is wrong in detail — see §5-D1.**
The synthetic sanity claim was re-run against the repo code: a pure rigid
rotation scores exactly 1.0000 ✓; isotropic per-joint noise scores far below
the headline shares ✓ qualitatively, but the specific 22.4% has no committed
artifact (§5-D5).

### 3.3 Aperture, k=2..16 × 4 partitions (epoch on val by R², scored on TEST)

Computed means over 4 partitions, with ranges — identical to §2.26's table:

| k | touch − shuffled (AUC) | pos | touch − pose-only (AUC) | pos |
|---|---|---|---|---|
| 2 | +0.112 [+0.095, +0.129] | 4/4 | +0.088 [+0.069, +0.109] | 4/4 |
| 4 | +0.114 [+0.062, +0.144] | 4/4 | +0.072 [+0.023, +0.096] | 4/4 |
| 8 | +0.110 [+0.074, +0.148] | 4/4 | +0.054 [+0.002, +0.076] | 4/4 |
| 16 | +0.143 [+0.099, +0.203] | 4/4 | +0.069 [+0.029, +0.120] | 4/4 |

- R² contrasts: touch−pose k=8 +0.058 [−0.010, +0.103] 3/4, k=16 +0.041 3/4 —
  confirms HANDOFF correction 9 (R² does NOT rise k=8→k=16). ✓
- Per-partition arm AUCs at k=8 and k=16 match §2.23/§2.24 cell-for-cell
  (e.g. k=8 partition 42: 0.665/0.601/0.517; k=16: 0.721/0.652/0.518). ✓
- Epoch selection: touch selects 2–12 in all 25 runs; pose-only 30–60 in all
  25; every touch epoch strictly earlier than every pose epoch within all 16
  partition-horizon cells. **25/25 confirmed.** ✓
- Seed counts: k=2/k=4 = 1 per cell; k=16 = 2; k=8 = 2 except partition 42
  which has **3** — confirms correction 10. ✓

### 3.4 Probe paper sweep

Marginal over matched pose, corrected target, best axis (computed):

| k | s42 / s0 / s1 | mean (pop sd) | RANDENC | recovery, gap | SCENE_CLEANENC |
|---|---|---|---|---|---|
| 2 | +0.0168/+0.0177/+0.0151 | **+0.0165** (0.0011) | +0.0073 | 44%, 8.5 sd | +0.0168 |
| 4 | +0.0164/+0.0171/+0.0143 | **+0.0159** (0.0012) | +0.0074 | 47%, 7.1 sd | +0.0181 |
| 8 | +0.0171/+0.0124/+0.0120 | **+0.0138** (0.0023) | +0.0122 | 88%, 0.6 sd | +0.0225 |
| 16 | +0.0132/+0.0043/+0.0051 | **+0.0075** (0.0040) | +0.0118 | 157%, −0.9 sd | +0.0171 |

All match §2.27, including the k=8-artifact finding (k=8 is the one horizon
where pretrained and random coincide; at k=2/k=4 the gap is 7–8.5 encoder-seed
sd) and the observation that seed 42 is the outlier high only at k=8/k=16
(seed 0 is highest at k=2/k=4). Conventional-vs-corrected inversion (3-encoder
means): +0.0042/+0.0048/+0.0038/+0.0027 vs +0.0165/+0.0159/+0.0138/+0.0075 —
computed ratios **3.9x / 3.3x / 3.6x / 2.8x** (see §5-D4 on "~4x").
Participant-disjoint shuffled controls sit at 0.4913–0.5011 ✓; touch-alone
pretrained 0.6490/0.6518/0.6606/0.6860 vs random 0.6083/0.6140/0.6392/0.6729 ✓.

Also re-derived from the same family:
- §2.2 seed-42 k=8: marginal +0.0171, touch 0.6605, shuffled 0.5009; per-axis
  +0.0171/+0.0126/+0.0187 (radial/spread/curl). ✓
- §2.3 CIs (`results_rigid_ci_k8_T36.json`): curl +0.0187 [+0.0064, +0.0311]
  17/20; radial +0.0171 [+0.0040, +0.0314] 17/20; spread +0.0126 [+0.0009,
  +0.0245] **12/20** vs pose (15/20 vs shuffled) — confirms the corrected
  12/20 count. ✓
- §2.4 decomposition 2×2: +0.0017 / +0.0052 / +0.0025 / +0.0171; interaction
  arithmetic (+0.0035, +0.0008, +0.0154) checks. ✓
- §2.5 SCENE (clip-trained encoder): published −0.0051, corrected +0.0165,
  touch 0.6774, shuffled 0.5023. ✓
- §2.12 TEST split: k=4 +0.0192 (touch 0.6468), k=8 +0.0199 (0.6566). ✓
- §2.21 per-joint k=16: curl +0.0198 [+0.0036, +0.0360] is the only axis whose
  interval excludes zero; radial +0.0132 [−0.0046, +0.0320], spread +0.0114
  [−0.0049, +0.0274]. ✓

### 3.5 Delta-MSE random-encoder (§2.31)

`results_randenc_horizons.json` contains **only k=2 and k=4**; §2.31's k=8 row
is carried over from §2.19d cluster logs and is not locally re-derivable.
Computed from the JSON:

| k | capacity-matched pre / rnd | vs pose pre / rnd | agree? | branch cost pre / rnd |
|---|---|---|---|---|
| 2 | −15.22% / −6.98% | −9.30% / −6.98% | yes (pretrained) | +6.98% / 0.00% |
| 4 | −16.67% / −12.98% | −12.88% / −13.64% | **NO** | +4.55% / −0.76% |

All figures match §2.31. The k=4 absolute random-minus-pretrained difference
is −1e-06 — exactly one quantum of the logs' 6-decimal precision — so "inside
logging precision" is correct. The doc's overall reading — the two contrasts
disagree, the two shuffled controls are not equivalent, and **this instrument
cannot answer the random-encoder question** — is supported by the JSON. One
internal inconsistency in its precision table: §5-D2.

### 3.6 Aperture random-encoder sweep (§2.32)

Computed from `results_aperture_randenc_summary.json` + the pretrained
aperture files (pose-only arms):

| k | random − pose | pre − pose | pre − random (pos) | pre-shuf − pose | rnd-shuf − pose |
|---|---|---|---|---|---|
| 2 | +0.0776 (4/4) | +0.0880 (4/4) | +0.0104 (3/4) | −0.0236 | +0.0017 |
| 4 | +0.0691 (4/4) | +0.0715 (4/4) | +0.0024 (3/4) | −0.0427 | −0.0023 |
| 8 | +0.0632 (4/4) | +0.0538 (4/4) | −0.0095 (1/4) | −0.0564 | −0.0021 |
| 16 | +0.0805 (4/4) | +0.0691 (4/4) | −0.0115 (1/4) | −0.0741 | −0.0000 |

Pooled pretrained−random: **mean −0.0020, sd 0.0164 (population), range
[−0.0332, +0.0287], positive 8/16** — matches §2.32 exactly. Both arms beat
pose-only 16/16 ✓. The pretrained capacity-matched gap exceeds the random one
in 16/16 cells ✓, and the branch-cost table shows why that comparison is
invalid (deranging a pretrained encoder's input costs up to 0.074 AUC vs
~0 for a random one) ✓. §2.32's methodological conclusion — the shuffled
control is not encoder-neutral; lead with touch − pose-only — follows from
the numbers.

### 3.7 Scalar ablation (§2.34)

Computed (mean over 4 partitions; full = the random-frz arms, scalar = the
`rndscalar` arms, both minus the same pose-only):

| k | full | scalar | scalar recovers | residual | scalar>pose | residual>0 |
|---|---|---|---|---|---|---|
| 2 | +0.0776 | +0.0700 | **90%** | +0.0076 [−0.0014, +0.0158] | 4/4 | 3/4 |
| 8 | +0.0632 | +0.0544 | **86%** | +0.0089 [−0.0025, +0.0190] | 4/4 | 3/4 |

Matches §2.34 (the "~88%" summary figure is the average of 90/86). ✓

### 3.8 STAG positive control (§2.35)

Computed from `results_stag_scalar_control.json`: full 27.81% (28.42/27.73/
27.28), scalar 5.00% (5.08/4.72/5.21), majority baseline 3.70% (=1/27),
above-baseline retained **5.4%**; n = 36,531 train + 16,119 test = 52,650
valid+balanced frames. All match. The "floor, not a reproduction" caveat
(vs STAG's published ~76%) is stated prominently in §2.35 and SESSION_HANDOFF
§5. ✓

### 3.9 Paired probe seeds (§2.36)

Computed from `results_probe_k2_RND_es{0,1,2}_{none,scalar}.json`:

| es | full | scalar | diff |
|---|---|---|---|
| 0 | +0.0101 | +0.0176 | +0.0075 |
| 1 | +0.0075 | +0.0176 | +0.0101 |
| 2 | +0.0088 | +0.0186 | +0.0098 |
| mean (pop sd) | **+0.0088** (0.0010) | **+0.0179** (0.0005) | **+0.0091** (0.0012), 3/3 |

diff/sd = 7.6 ✓; touch-alone 0.6093 → 0.6199 ✓; k=8 unseeded pair full
+0.0122 / scalar +0.0179 ✓; scalar-random vs pretrained-full +0.0179 vs
+0.0165 within ~1 sd ✓; seeded pretrained-vs-random k=2 gap +0.0165 vs
+0.0088 = 1.9x at 7 sd ✓. All match §2.36.

### 3.10 Shortcut analysis

No `results_shortcut_*.json` exists in the local repo — **cluster-only, not
locally verifiable**. What is verifiable locally: `scripts/shortcut_analysis.py`
exists, refuses non-conventional-target checkpoints, and the withdrawn
"addressable fraction of MSE" metric is **removed from the code** with an
explicit in-code explanation of why it was unsound
(`scripts/shortcut_analysis.py:150-161`); commit `da09239` records the numbers
(prediction 24–42% rotational vs target ~96%, refuting the shortcut
hypothesis, with models emitting only 14–37% of target motion magnitude).
The claim rests on cluster JSONs and a commit message; it appears in **no**
HANDOFF section (see §8).

### 3.11 Retrieval / GRU

Computed from the committed JSONs:

| quantity | computed | claimed |
|---|---|---|
| avg-pool clip test T→P mAP | 16.756 | 16.76 |
| avg-pool clip val | 14.411 | 14.41/14.42 |
| biGRU clip test | 45.463 | 45.46 |
| biGRU clip test, 3 seeds | 45.46 / 46.28 / 46.76 (mean 46.17, pop sd 0.54) | 46.17 ± 0.54 |
| clip ratio | **2.71x** | 2.7x |
| avg-pool scene (no-ReLU) test / val | 6.44 / 7.02 | 6.44 / 7.02 |
| biGRU scene test / val | 28.31 / 29.09 | 28.31 / 29.09 |
| scene ratio | **4.40x** | 4.40x |
| superseded ReLU avg-pool arm | 5.29 / 5.81 | 5.29 / 5.81 (quarantined §2.13) |
| bootstrap clip GRU test | 45.50 [42.09, 48.65], n=1399/296 clips | same |
| bootstrap scene GRU test | 28.31 [25.59, 31.01], n=1411/257 | same |
| bootstrap scene avg-pool test | 6.46 [5.45, 7.59] | same |

Scene-disjoint CIs **nowhere near overlapping** ✓. Bootstrap outputs record
`split_group_by` and `split_group_by_source` (observed `"cli"` in the
avg-pool files) — the §2.20 trap fix is real and in the artifacts. ✓
The 3-seed GRU figures previously sourced to `experiments.md` are indeed in
committed JSONs (`results_p2t_gru/seed0/seed1.json`). The clip avg-pool
baseline, by contrast, is a **single run** (no seed variants exist in
`results/`), sourced to a cluster log dir (`2026_06_22-20_49_48`) that is not
in this repo — see §5-D6 and §8.

Also re-derived while in the area (supporting the withdrawal decisions):
- §2.19b forecast CIs: k=8 frz−frzshuf −14.96% [−18.91, −11.24], frz−pose
  −6.87% [−11.06, −2.67], frzshuf−pose +9.52% [+7.36, +11.43]; k=16 frz−pose
  **−2.70% [−7.50, +1.82] includes zero** — the withdrawal is justified. ✓
- §2.19c encoder ladder: rungs and Pearson r = +0.388 match;
  `split_seed_observed` records the sign flip (seed 3: **+5.02%**) that
  underlies "do not publish §2.19's 15%". ✓

## 4. Methods & code review findings

Two independent line-by-line review passes were run over the method code (one
over the Kabsch/rotation-share/loader stack, one over the
ablation/probe/control stack), plus a direct pass by this audit. Merged
findings below; every item was checked against the working tree.

**Sound (verified in code, with pinning tests):**

- **Kabsch** (`src/opentouch/articulation_frames.py:102-129`): proper-rotation
  repair is correctly implemented — `det_sign = det(V @ U^T)` applied as a
  sign flip on the last singular vector (the smallest singular value, as the
  textbook `R = V·diag(1,1,d)·U^T` requires), so reflections can never absorb
  articulation; the SVD runs in float64 (`:124`, but see finding 3 — the
  covariance is accumulated in float32 first); the residual convention
  (`target @ R − source`, `:158`) is the correct un-rotation. Convention
  checked against the standard Kabsch solution. Pinned by
  `tests/test_articulation_frames.py` (known-rotation recovery, an
  adversarial reflected-source test, zero-residual for rigid motion,
  world-frame invariance).
- **`rigid_fraction`** (`articulation_frames.py:283-299`): mathematically an
  **upper bound** on rotation content — the Kabsch fit maximizes explained
  energy over rotations, and R=I is feasible, so the ratio is always in [0,1]
  and the reported share can only overstate, never understate, true rotation.
  Zero-motion samples return 0 (conservative). The bound property, however, is
  stated nowhere in the docs — §5-D6.
- **Palm frame** (`articulation_frames.py:162-221`): orthonormal by
  construction; degenerate hands detected and excluded rather than NaN'd;
  orthonormality/right-handedness/uses-only-t pinned by tests.
- **`scripts/rotation_share.py`**: correct (t, t+k) pairing per sequence, gap
  handling delegated to loaders (by contract), degenerate-palm exclusion,
  deterministic seeded 200k-pair cap with group alignment preserved through
  the subsample (`:115-120`), ragged object-array `.npy` handling with a
  round-trip regression test.
- **`scripts/load_public_hands.py`**: joint-order guard (`:231-298`) is a real
  guard that raises by default and caught the one real error that occurred
  (MANO-read-as-MediaPipe, 16.5% vs 88.4% — pinned by test); per-dataset
  `NATIVE_ORDER` with source-level evidence for DexYCB's MediaPipe order.
  See finding 8 for its blind spots. Rigid-template drop (`:129-144`) tests
  exact shape-constancy on the full pairwise-distance profile (correct and
  well calibrated; both directions pinned). `dedupe_views` (`:189-219`) is
  content-based bucket-then-verify (`np.allclose` rtol 1e-4), which cannot
  wrongly merge genuinely *articulating* takes (pinned by
  `test_genuinely_different_takes_are_not_collapsed`) — with the
  rigid-template exception in finding 9. Gap splitting correct incl. sort,
  min-length, and unordered input (pinned).
- **`scripts/rotation_share_motion_sensitivity.py`**: motion magnitude is the
  square root of `rigid_fraction`'s own denominator, so the conditioning
  variable is internally consistent with the metric, and `by_floor["0"]`
  reproduces the headline all-samples medians (verified). The subsample is
  *near*-identical, not identical, to `rotation_share.py`'s — see finding 7.
- **`tactile_reduce`** (`src/opentouch/pose_regression.py:361-378`): `"none"`
  is a pure pass-through — bit-identity pinned by
  `tests/test_tactile_reduce.py::test_none_is_bit_identical_to_the_default`;
  `"scalar"` is a spatial mean broadcast that preserves per-frame totals
  (pinned), applied **before the encoder in both fusion paths** (gate `:417`,
  film `:402`) with parameter count unchanged (pinned). The pose-only model
  asserts tactile is `None` (`:425-430`), making leakage structural, not
  behavioral.
- **`scripts/probe_rigid.py`**: `--encoder-seed` seeds only the random
  encoder's init (`:91-92`), enabling genuinely paired bit-identical-encoder
  contrasts; seed recorded in the output JSON (`:174`). `--tactile-reduce`
  applies the identical reduction to the **shared** tensor before both the
  real and deranged encodings (`:125-130`), so the ablation is not confounded
  with the derangement. `--random-tactile-encoder` leaves the pose encoder
  pretrained (`:104`). Motion threshold computed on train only (`:112`,
  passed frozen to eval). `--split-group-by` is explicit with a visible
  default and is recorded in the output.
- **`scripts/stag_scalar_control.py`**: standardization uses TRAIN statistics
  only (`:136-138`) and is a per-tensor affine applied before the reduction,
  which commutes with a spatial mean — so it cannot change what the reduction
  destroys (the argument at `:132-135` is correct). Uses STAG's own `splitId`;
  majority baseline computed on test; the broken unnormalized first attempt is
  documented in-code rather than erased.
- **`scripts/shortcut_analysis.py`**: the withdrawn "addressable MSE" metric
  **stayed withdrawn** — not computed anywhere; the in-code note (`:150-161`)
  explains the non-orthogonality reason. Hard guard rejects checkpoints not
  trained on the conventional target (`:82-88`); prediction wrist-row zeroed
  to match the target convention (`:134`).
- **Collectors**: `scripts/collect_randenc.py` reads the final `[rigid]`
  eval line, was validated against §2.19d's published 0.000261 (stated in its
  header), and encodes both project rules — branch cost reported next to the
  capacity-matched gap, and a resolvability guard at the logs' 6-decimal
  precision. `scripts/collect_aperture_randenc.py` pairs the correct
  pretrained files (including the historical k=8 split across
  `earlystop`/`splitseed_ss*`), guards missing cells, and computes each
  encoder's gap against its **own** shuffled control.
- **Known landmines, all handled:**
  - *Selection on val vs test*: `scripts/aperture_select_and_test.py` selects
    the epoch on val **by R² only** and scores the selected checkpoint on
    test via the standalone evaluator; AUC is never used to select (`:28-31`,
    `:124-125`). Verified structurally in the JSONs (`best_val_epoch`,
    `val_*` vs `test_*` fields).
  - *`--split-group-by` defaulting*: `bootstrap_eval.py:185-211` warns loudly
    when the checkpoint records nothing and stamps `split_group_by_source`
    into the output (observed in the committed bootstrap JSONs); regression
    pipeline forwarding pinned by
    `tests/test_pose_regression.py::test_regression_data_forwards_split_group_by_to_the_splitter`
    and default-is-clip pinned at `:1318`.
  - *Capacity-matched contrast read with branch cost*: institutionalized in
    both collectors and in the doc guidance (§2.22 FiLM trap, §2.31, §2.32).
  - *Single-partition results*: the 4-partition protocol with per-partition
    encoders is real in the JSONs (16 aperture cells, each with its own
    `p2t_scene_gru_ss*` encoder per the runner scripts).
  - *Encoder-seed provenance*: new runs record `encoder_seed`; the older
    unseeded RANDENC runs record `random_tactile_encoder: true` but no seed —
    see the fragility note below.

**Findings (wrong, fragile, or attackable):**

*Robustness blocker (does not affect any committed result, but will bite the
released diagnostic):*

1. **[blocker-robustness] One non-finite coordinate aborts the whole
   rotation-share run.** `torch.linalg.svd` at `articulation_frames.py:124`
   is batched — a single NaN sample raises `LinAlgError` for the entire
   concatenated tensor, after the full dataset load. `hand_frame_degenerate`
   does **not** catch NaN (all comparisons come out False, so the sample is
   kept — verified empirically). Only `load_public_hands.py` filters
   non-finite frames; `rotation_share.py`'s input contract never requires
   finiteness. Every committed run went through the loaders, so no number is
   affected — but the "piece worth releasing" dies opaquely on the first
   dirty array a third party feeds it. No NaN/inf test exists anywhere.

*Reviewer-attackable:*

2. **The headline rotation-share is an upper bound and is never framed as
   one — and `rotation_share.py:5-8` states the inverse.** `rigid_fraction`
   reports the energy the *best-fit* rotation explains — a maximum over
   rotations, so an **upper bound** on rotation content: coordinated
   articulation (all five fingers flexing together) is partly absorbed by the
   fit. The script's own docstring says "whatever fraction … the best rigid
   rotation explains **is not articulation at all**", which is false in
   exactly the damaging direction. No doc (HANDOFF §2.25/§2.28/§2.30,
   SESSION_HANDOFF §5) states the bound, and no test constructs the
   coordinated-flexion case (the existing `rigid_fraction` test uses
   independent per-joint jitter — the easy case). The motion-sensitivity
   result defuses the *stillness* attack (which biases the number down), not
   this one (which biases it up). One sentence fixes the docs; one test pins
   the bound.
3. **The float64 claim in Kabsch is half-implemented.**
   `articulation_frames.py:120-124`: the cross-covariance is accumulated in
   **float32** and only then cast (`covariance.double()`); the near-planarity
   the comment cites as the reason for float64 contaminates the third
   singular value *before* the cast. Measured effect on a synthetic
   near-planar hand: up to 0.012° rotation difference vs a full-float64
   pipeline — far too small to move any committed statistic (shares are
   quoted to 3 decimals), but the in-code justification overclaims. Fix:
   `source.double().transpose(1,2) @ target.double()`.
4. **Best-axis selection happens on the eval split.**
   `scripts/probe_rigid.py:195` picks `best_axis` by tactile-alone AUC on the
   same split whose marginal is then quoted. Containment: the pick was
   verified stable (`long`/`y`) across all 12 committed probe JSONs, it is a
   3-way choice made on touch-alone rather than on the marginal, and all
   per-axis marginals are reported with per-axis CIs. Still selection-on-eval
   by construction, and the paired `none`/`scalar` runs are not guaranteed to
   agree on the axis they are differenced over. Cheap fix: pre-register the
   axis or select on train.
5. **The §2.36 paired-scalar result is clip-split val throughout, and §2.36
   never says so.** All `results_probe_k2_RND_es*` and `*_RANDENC*` JSONs
   record `split_group_by: "clip"`, `eval_split: "val"` (verified). The flag
   is explicit and recorded — but the doc section quoting +0.0091 does not
   state the split, in a project whose own history includes a silent-clip
   failure worth 2.5x (§2.20). Related: `scripts/probe_rigid_ci.py:50` and
   `scripts/tactile_direction_probe.py:637` call `_load_and_split_dataset`
   with **no `group_by` argument at all** — silently clip, no CLI flag, no
   record in their outputs — so the CI companion cannot be pointed at a
   scene-split probe run, and it also lacks
   `--tactile-reduce`/`--encoder-seed`/`--random-tactile-encoder`, meaning
   the 7.6-sd claim for the scalar contrast has no clip-clustered CI and
   currently cannot be given one without code changes.
6. **`hand_frame_degenerate` is an exact-degeneracy test, not a stability
   test, with unit-dependent absolute thresholds.**
   `articulation_frames.py:206-221`: with `tol=1e-6` only sinθ ≈ 1e-7 palms
   are flagged (verified; sinθ ≈ 1e-5 passes), and the `long_norm`/
   `transverse_norm` cutoffs are absolute — 1e-6 means 1 µm in metre data and
   1 nm in millimetre data (DexYCB), so they essentially never fire.
   `load_public_hands.py:276` uses 1e-3 for the same geometric quantity —
   three orders of magnitude apart. Degenerate rows also silently fall back
   to the world identity basis with no flag in `all_target_variants`'s return
   (every current consumer filters separately; nothing enforces it).
7. **The two rotation-share scripts do not subsample identically, though one
   claims they do.** `rotation_share.py:90` creates its RNG once outside the
   horizon loop (so the k=8 draw depends on the `--horizons` list), while
   `rotation_share_motion_sensitivity.py:72` creates a fresh `default_rng(0)`
   per horizon and its docstring (`:55-56`) claims "subsampled identically".
   Selection is positional over the concatenated tensor, so it also depends
   on sequence concatenation order, which differs between the two OpenTouch
   drivers. Observable in the committed artifacts: 0.957216 vs 0.957237 at
   k=2 — a 5th-decimal divergence, invisible at quoted precision, but the
   parity claim is false as written and neither JSON records seed, cap, raw
   pair count, or degenerate-drop count (`n_pairs` is post-filter,
   post-subsample).
8. **The joint-order guard is necessary, not sufficient.**
   `load_public_hands.py:231-298`: only prediction A (per-finger monotone
   radius) gates; the palm-collinearity check (prediction B) is printed but
   never enforced despite the docstring calling them "two independent
   predictions". Monotonicity is invariant to permutations of whole finger
   *blocks* — an index↔ring block swap passes at 100% while joints 5/9/17 are
   wrong, which is precisely the failure class the guard exists for. It also
   cannot detect left/right handedness (flips `curl`/`spread` signs; relevant
   to ARCTIC's `--side`). The 0.80 threshold and the ">95% for a correct
   layout" claim were only ever calibrated on a synthetic flat-fingered hand,
   never on real HO-3D/DexYCB output — and MCP<PIP<DIP<TIP radius
   monotonicity is weakest exactly for flexed grasps, which is what these
   datasets contain. It did catch the one real error that occurred
   (MANO-read-as-MediaPipe, 16.5% vs 88.4%), which is pinned by test.
9. **`drop_rigid_templates` → `dedupe_views` ordering is load-bearing and
   undocumented.** A rigid-template annotation has a *constant*
   pairwise-distance profile from the same frozen MANO template, so MC1–MC6 —
   the six genuinely-different takes the docstring cites as the reason dedupe
   is content-based — have **identical** fingerprints and would be wrongly
   merged if they reached `dedupe_views`. The default pipeline is saved only
   because `main()` drops rigid templates first
   (`load_public_hands.py:550-553`); `--keep-rigid-templates` silently breaks
   that protection, and no test covers the interaction. For genuinely
   articulating takes the merge tolerance (~10 µm agreement over 64×210
   values) is unreachable, so the committed results are unaffected (HO-3D
   collapsed 55→9 exactly as the camera structure predicts).
10. **The contaminated HO-3D result holds the canonical filename.**
    `results/results_rotation_share_ho3d.json` (0.868 at k=8, 55 groups,
    `ABF10..ABF14` and the rigid-template families all present) is the
    unfiltered run, and it sits at the exact output path the usage block in
    `load_public_hands.py:78-79` tells a user to write; the correct number
    lives in `..._ho3d_clean.json` (0.822). HANDOFF §2.30 documents the
    contamination clearly, but the file layout invites quoting the wrong one.
11. **Collector provenance is convention-based, not verified.**
    `scripts/collect_randenc.py:64-69`: the protocol banner
    ("scene-disjoint, split_seed 42, …") is hardcoded, arms are located
    purely by directory-name template, and nothing cross-checks `params.txt`
    for split, seed, or encoder flags; the collected metric is the final-epoch
    **val** figure and the output JSON carries no `split` field.
    `scripts/collect_aperture_randenc.py:40-47` pairs pretrained and random
    cells by filename convention; the input JSONs contain no
    `split_group_by`/`split_seed`/encoder-checkpoint fields, so
    "same partition" cannot be verified from the artifacts alone.
    Similarly `regression_main.py` records `split_seed` but not `args.seed`
    in checkpoints — and `args.seed` is what determines the random tactile
    encoder under `--freeze-random-tactile-encoder`, i.e. the load-bearing
    quantity for the §2.34 scalar-vs-full pairing (the pairing itself was
    verified sound: seeding precedes construction and `tactile_reduce`
    consumes no constructor RNG, so same-seed runs get bit-identical
    encoders — but the seed survives only in `params.txt` on the cluster).
    The `results_aperture_scalar_*` JSONs carry no `tactile_reduce` marker;
    the filename is the only distinguishing provenance.
12. **`shortcut_analysis.py` does not reconstruct `tactile_reduce` from
    checkpoint meta** (`:91-98`, unlike `regression_eval.py:268`) — latent
    today (all shortcut checkpoints are pose-only) but silently wrong the
    moment a scalar checkpoint is analysed, and `strict=True` cannot catch it
    because the flag changes no parameter shapes. Its report also omits
    `split_group_by`/`split_seed`, so the "clip-split, pose-only" scoping of
    the shortcut result is not recoverable from its own output.

*Minor:*

13. `dedupe_views` bucket key (`load_public_hands.py:158-175`) rounds the
    profile mean/max to 2 decimals; a true duplicate straddling a rounding
    boundary (~1e-4 probability per scalar) lands in a different bucket and
    silently escapes the merge — bucket-then-verify eliminates false merges,
    not missed ones. `_view_profile` is also computed twice per sequence.
14. Unseeded RANDENC probe runs are unregenerable: `encoder_seed` is absent
    from `results_probe_rigid_k*_RANDENC*.json` (verified) — §2.27's random
    column and the §2.36 k=8 pair are single unrepeatable draws (the k=8
    "pair" is additionally unpaired on encoder init, and its `none` side
    predates the `tactile_reduce` flag). The seeded k=2 triple (init sd
    0.0010) is the containment. Also `--encoder-seed` is silently ignored
    without `--random-tactile-encoder` yet still written to the output JSON,
    and `split_seed` (which seeds the derangement) is not recorded.
15. Stale docstring naming at `articulation_frames.py:163-164` ("long, flex,
    normal" — the retracted names) vs the code's (radial, spread, curl); the
    result JSONs likewise carry the old key names (long→radial, flex→spread,
    normal→curl), so raw-JSON readers can mislabel axes without HANDOFF §3.2.
16. `pose_regression.py:365`'s "preserved bit-for-bit" (and §2.34's
    "preserved *exactly*") is false at floating-point precision — measured
    up to 2.3e-5 absolute on a ~128 sum (rel ~1.6e-7); the pinning test
    itself uses rtol 1e-5. Nothing turns on it; the phrasing is trivially
    falsifiable. Also `:371-375`'s `dim() < 2` guard would batch-average a
    2-D input (latent; encoder requires the grid — guard should be
    `dim() < 3`).
17. Committed `results_randenc_horizons.json` predates the current collector
    (lacks the `resolvable_at_logged_precision`/branch-cost/`contrasts_agree`
    keys `collect_randenc.py` now emits; has the older
    `random_beats_pretrained` instead). Numbers consistent; schema stale;
    regeneration needs cluster logs. `collect_randenc.py:55,90` would also
    `TypeError` on a zero denominator rather than degrade.
18. DexYCB sentinel check (`load_public_hands.py:432`) only drops frames
    where *all* entries are −1; a partially-invalid frame would pass and
    inject a fake delta (standard DexYCB fills the whole array, so likely
    never fires). `--limit-seqs` overcounts and leaks across subjects
    (`:416-418`, `:475-477`; smoke-test only). `_loadinfo.json:577-580`
    records `joint_order: None` on the default path (not the order actually
    applied) and omits the rigid-template/dedupe drop counts, which exist
    only on stdout despite materially moving the headline. Units doc
    contradiction: `:384-385` says millimetres, `:27-29` says /1000 (metres).
19. Wrong cross-reference at `shortcut_analysis.py:206` ("See HANDOFF 2.34");
    the shortcut analysis has no HANDOFF section at all (§8).
20. `stag_scalar_control.py:116-124` silently ignores middle `splitId` values
    (moot here: n_train + n_test = 52,650 = all kept frames, so exactly two
    values existed — verified); `drop_last=True` would train on zero batches
    below 256 frames and still report a number. Positive finding worth
    keeping: `torch.manual_seed` precedes both model construction and loader
    creation, so `none`/`scalar` share init *and* batch order per seed —
    genuinely paired.
21. `aperture_select_and_test.py`: subprocess `returncode` unchecked and the
    first regex match taken (`:47-53`; currently exactly one match exists per
    eval); `_s(\d)$` (`:66`) silently skips seeds ≥ 10.
22. `src/opentouch_train/data.py:342-343`: an on-disk `DatasetDict` is
    returned verbatim, silently discarding `seed` and `group_by` — for such a
    dataset `--split-group-by scene` would be a no-op while every artifact
    records "scene". No committed result used this path, but it is exactly
    the shape of the §2.20 trap.
23. Cross-script drift risk: the scalar reduction is copy-pasted in three
    places (`pose_regression.py:377-378`, `probe_rigid.py:130`,
    `stag_scalar_control.py:36`) with no test tying them together; no test
    covers `bootstrap_eval`'s assume-clip warning or the
    `split_group_by_source` field (the part that prevents the documented
    failure); `rotation_share_motion_sensitivity.py` and
    `stag_scalar_control.py` have no tests at all; the ARCTIC loader is
    untested; `hand_frame_degenerate`'s tolerance and the identity fallback
    are unpinned; module-level `assert` at `articulation_frames.py:78`
    vanishes under `python -O`.

## 5. Discrepancies found (doc vs JSON)

Discrepancies exist. None moves a conclusion; all should be fixed before any
external document quotes the affected sentences.

- **D1 — §2.29's decile-monotonicity sentence is wrong.** Claimed: "the trend
  is monotone across all ten deciles in three of the four cases (the exception
  is DexYCB k=8, where deciles 3 and 4 invert by 2.3 points)". Computed:
  monotone in **2 of 4** cases — OpenTouch k=8 also inverts, deciles 2→3,
  0.9307 → 0.9294 (−0.13 pts). Also the DexYCB k=8 inversion is 0.9005 →
  0.8785 = **2.2** points, not 2.3. The overall rising trend and every
  by-floor number are correct; only this sentence overstates.
- **D2 — §2.31's precision table calls k=2 "resolved — pretrained better" for
  the absolute comparison; by the project's own criterion it is not.** The
  k=2 absolute difference (rnd 4.0e-05 vs pre 3.9e-05) is 1e-06 — exactly one
  quantum of the 6-decimal logging, the same magnitude the same table rules
  "inside rounding, not resolvable" at k=4, and below the
  `resolvable_at_logged_precision` threshold in `collect_randenc.py:96`
  (requires > 1e-06). The *directional* k=2 conclusion is separately supported
  (both contrasts favor pretrained at k=2), but the absolute-comparison cell
  is mislabeled. Note this strengthens §2.31's own conclusion that the
  instrument cannot resolve the question.
- **D3 — §2.27's "corrected beats conventional ~4x at k=2/4/8"**: computed
  ratios are 3.9x / **3.3x** / 3.6x. "~4x" is generous at k=4; "3–4x" is the
  defensible phrasing.
- **D4 — §2.29's synthetic-noise figure (22.4%) has no committed artifact,
  and its value is construction-dependent.** No script flag, test, or JSON
  produces it. Re-running the repo's own `rigid_fraction` on synthetic input:
  pure rigid rotation scores exactly 1.0000 ✓; isotropic per-joint noise with
  the **wrist held fixed** scores ~0.04 (matching theory: 3 rotational DOF /
  60 noise DOF); with the **wrist also jittered** it scores ~0.23 — which is
  evidently the variant behind "22.4%". The published noise floor therefore
  swings ~6x on an unstated modelling choice. The argument's direction
  survives either way (both are far below the 82–96% headline shares), but
  the figure should be regenerated by a committed, parameterised test before
  it is quoted.
- **D5 — the "SMu4" take named in §2.30 appears as `SMu40` in the JSON** (the
  camera-suffix form). Cosmetic; the value (1.0000 at k=2) is confirmed.
- **D6 — two known limits are true but stated nowhere the finding is
  claimed:** (a) the rotation share is an **upper bound by construction**
  (finding 2 in §4) — absent from §2.25/§2.28/§2.30 and SESSION_HANDOFF §5;
  (b) the clip-split retrieval **avg-pool baseline (16.76) is a single run**
  — the GRU side has 3 seeds with sd 0.54, the baseline has none, and §2.1/
  §2.15 present the comparison without that asymmetry. The referenced source
  run lives only in a cluster log dir. (A fresh 3-seed avg-pool baseline is
  reportedly training on the cluster; not locally verifiable.)

Everything else checked — including every table in §2.23–§2.36, the §3
corrections 9–11, the retrieval and bootstrap numbers, and the do-not-publish
list — reproduced from the JSONs exactly.

## 6. OPENTOUCH RESUBMISSION MATERIAL

What belongs in the lab's ICLR/ICRA resubmission of the OpenTouch paper:

**Certainly in:**
- **The GRU pose-encoder change (PR #3).** The core numbers are verified
  end-to-end from committed JSONs: clip-disjoint T→P 16.76 → 45.46
  (**2.71x**, 3 GRU seeds 46.17 ± 0.54), and — the stronger, honest framing —
  scene-disjoint 6.44 → 28.31 (**4.40x**) with clip-clustered 95% CIs that do
  not remotely overlap ([5.45, 7.59] vs [25.59, 31.01] on test). The PR itself
  could not be inspected from this machine (no `gh`), but every number it
  should quote is in `results/`.
- **The scene-disjoint (participant-held-out) evaluation itself,** with both
  absolute levels and the ratio. The clip-split absolute numbers were inflated
  by participant memorization (45.46 → 28.31; 16.76 → 6.44); the architectural
  claim *widens* under the honest split. Reporting only clip-split numbers in
  a resubmission would be repeating the error this audit chain found.
- **The retrieval bootstrap CIs** (clip-clustered, fixed gallery). They are
  the difference between "2.7x" as a point estimate and a defensible claim,
  and the machinery is tested (`tests/test_bootstrap_eval.py`).
- **The avg-pool baseline fidelity fix** (§2.15: the ReLU that upstream
  avg-pool never had, worth 16.76 vs 10.25 on the upstream checkpoint;
  `temporal_mode={gru,mean}` runs both arms in one codebase). This is what
  makes the baseline comparison like-for-like, and reviewers of a
  resubmission will ask exactly this question.

**In, but as a methods footnote/appendix rather than a contribution:**
- **The `split_group_by` reproducibility trap** (scene checkpoint silently
  scored on a clip gallery: 72.09 vs 28.31, off by 2.5x) and its fix
  (`split_group_by_source` stamped in outputs, loud fallback warning). One
  paragraph in an appendix; it materially protects anyone reproducing the
  eval, but it is not a paper contribution.
- **mAP's gallery-size dependence** (same weights: 14.42 on 1572 windows,
  16.76 on 1399) — one sentence: always report N with mAP.

**Blocking caveat before the resubmission quotes 2.71x:** the clip avg-pool
baseline (16.76) is a single run with no in-repo seed replicates and its
source run recorded only as a cluster log path (§5-D6). The scene-disjoint
comparison does not share this problem in substance (its CI is committed), but
the clip-split headline does. The fresh 3-seed avg-pool baseline should land
before submission, or the paper should lead with the scene-disjoint pair.

**Out of scope for that paper:** everything measurement-side — the rotation
share, target correction, aperture, scalar/STAG, and pretraining-buys-nothing
results. Those contradict/extend a different claim set and belong to the
workshop paper; mixing them into an architecture resubmission would dilute
both. The forecasting line (§2.16–§2.19) should stay out entirely: its
pose-only margin did not survive partition redraws and its own docs withdraw
it.

## 7. WORKSHOP PAPER MATERIAL (the measurement paper)

Verified against the JSONs, with what must not be claimed (cross-checked
against SESSION_HANDOFF §5's do-not-publish list — every item on that list is
consistent with the JSONs, and the list itself was re-derived: the §2.19 sign
flip (+5.02% at seed 3) is in `results_encoder_ladder.json`; the k=16 probe
marginal's sd (0.0040) exceeds half its mean (+0.0075); k=2/k=4 aperture
cells are single-seed; seed 42 is mid-pack at k=2/k=4; HO-3D's 9-take spread
is 49–100%):

1. **Target composition, 3 datasets** — *status: VERIFIED; strength: the lead
   contribution.* 96.1% / 89.6% / 82.1% at k=8, with the
   transport-vs-articulation ordering predicted before HO-3D was measured, 26
   OpenTouch scenes all >90%, 261/300 DexYCB takes >80%, and the HO-3D
   contamination story (86.8% → 82.1%) as evidence of care. Must not claim:
   "~95%" as a general figure (OpenTouch-only); conceptual novelty over
   Procrustes (§2.33: PA-MPJPE is known — the novelty is the *quantification*,
   the temporal framing, and the flipped conclusion); HO-3D pooled without its
   9-take 49–100% range. **Add before submission:** the upper-bound sentence
   (§5-D6a) — it is one line and pre-empts the sharpest attack.
2. **Motion-sensitivity robustness** — *status: VERIFIED; strength: strong
   supporting result.* The share rises with motion on all three datasets;
   above-median 98.0/94.0/91.5 at k=8; the all-samples headline is the
   conservative one. Must not claim: the "monotone in 3 of 4 cases" sentence
   (§5-D1 — say "rises across deciles with two ≤2.2-pt local inversions");
   the 22.4% synthetic figure unless regenerated with a committed script
   (§5-D4).
3. **Corrected-target consequence** — *status: VERIFIED; strength: the
   centrepiece mechanism.* Conventional +0.0027..+0.0048 vs corrected
   +0.0075..+0.0165, 3-encoder means, four horizons; survives
   participant-disjoint splits at full strength (+0.0168..+0.0225 with a
   scene-trained encoder) and the held-out test split (+0.0192/+0.0199).
   Must not claim: the k=16 marginal (+0.0075; encoder-seed sd 0.0040); "~4x"
   (say 3–4x, §5-D3); curl as *the strongest* axis (only "the only axis whose
   k=16 CI excludes zero"); "seed 42 is the outlier high" unqualified.
4. **Grip aperture** — *status: VERIFIED; strength: the only forecasting-line
   result that survives partition redraws, and it is solid.* touch − pose-only
   +0.054..+0.088 AUC, 4/4 at every horizon; capacity-matched +0.110..+0.143,
   16/16 cells. Lead with touch − pose-only (per §2.32 — the shuffled control
   is encoder-dependent, and at k=16 roughly half the capacity-matched gap is
   the control failing); report the capacity-matched figure with its branch
   cost alongside. Also safe and cheap: the 25/25 epoch-selection asymmetry
   (touch 2–12, pose 30–60). Must not claim: the horizon *shape* (k=2/k=4 are
   single-seed); "the effect strengthens with horizon" (§2.24's withdrawn
   reading — the pose-only margin is largest at k=2).
5. **Scalar / STAG spatial-vs-sum** — *status: VERIFIED; strength: the
   sharpest single claim in the project.* 90%/86% retention on aperture, 4/4;
   the same reduction takes STAG object-ID from 27.8% to 5.0% against 3.7%
   (5% retained); the probe generalization is paired on bit-identical encoders
   (+0.0091, sd 0.0012, 3/3). The licensed sentence — *spatial tactile
   structure encodes what the hand is holding, not what it is about to do* —
   is supported by the artifacts. Must not claim: 27.8% as STAG's accuracy (a
   floor; their paper reports ~76%); the k=8 probe scalar magnitude before
   the pair is seeded; anything beyond OpenTouch + a rotation-invariant target
   for the aperture half (limits already stated in §2.34).
6. **Pretraining buys nothing (contribution 2)** — *status: VERIFIED on
   aperture; strength: good, if scoped exactly.* Pooled pretrained − random
   −0.0020, 8/16, both arms beating pose-only 16/16. Must not claim: "random
   beats pretrained" (§2.19d — withdrawn; the delta-MSE instrument cannot
   answer, and both docs and this audit confirm the k=2/k=4 contrasts
   disagree); any *general* "the representation is worthless" — the probe
   shows the learned representation is worth ~2x a random projection at
   k=2/k=4 (7–8.5 encoder-seed sd) and against raw kinematics and on
   magnitude (§2.19f). The honest composite: *on the one downstream task
   where tactile helps, retrieval pretraining adds nothing measurable; on
   short-horizon direction decoding it supplies about half the marginal.*
   Note the probe's random column is single-draw per horizon (finding 14,
   §4) — the seeded k=2 triple is the defensible anchor.
7. **Encoder-dependent shuffled-control methods note** — *status: VERIFIED;
   strength: genuinely useful methods contribution.* Deranging a pretrained
   encoder's input costs up to 0.074 AUC / 4.6–9.0% MSE against pose-only
   while deranging a random encoder's costs ~0 — so capacity-matched gaps are
   not comparable across encoders, and a shuffled control's severity depends
   on the encoder being controlled. Three independent sightings (FiLM §2.22,
   delta-MSE §2.31, aperture §2.32) are all in committed JSONs. This is the
   kind of small, transferable methods finding workshop reviewers reward.

## 8. Open items

**Not locally verifiable (cluster-only) — mark accordingly wherever quoted:**
- All `results_shortcut_*` JSONs (the 24–42%-vs-96% refutation). Locally
  there is only the script and commit `da09239`. The finding also has **no
  HANDOFF section** — before anyone quotes it, it needs both the JSONs
  committed and a doc section; the script's own pointer ("HANDOFF 2.34") is
  wrong (finding 19, §4). Its checkpoints are clip-split and pose-only (per
  the commit); that scoping is not verifiable from here either, and the
  script's own report omits the split fields that would prove it (finding 12).
- §2.31's k=8 row (pre −14.56/rnd −10.62 etc.) — rests on §2.19d cluster log
  dirs (`logs/rnd_*`), not on any committed JSON.
- The clip avg-pool retrieval source run (`opentouch/logs/2026_06_22-20_49_48`)
  and the in-flight fresh 3-seed avg-pool baseline.
- PR #3 itself (no `gh` on this machine); its underlying numbers are verified.
- The `p2t_scene_gru{,_ss1,_ss2,_ss3}` encoder checkpoints (~6.5h each,
  unregenerable after cluster access ends — preservation decision is still
  open per SESSION_HANDOFF §6).
- The cluster-side verification that `tactile_reduce="none"` reproduced
  historical runs bit-for-bit (§2.34 claims it; the local equivalent is the
  unit test, which passes).

**Highest-risk claims if a reviewer pushes, in order:**
1. **The upper-bound objection to the rotation share** — currently unstated
   (§5-D6a). One sentence to add; without it, "96% is rotation" can be
   attacked as "your metric attributes ambiguous energy to rotation by
   construction."
2. **Aperture k=2/k=4 single-seed cells** — the horizon shape must stay out
   of the paper (already the docs' position); a reviewer asking "how many
   seeds?" at k=2 gets "one" for the +0.088 headline cell. The 4/4-partition
   agreement is the defense; a second seed is cheap if the box still exists.
3. **The probe's unseeded, unregenerable random-encoder column** (§4 finding
   14) — the 44%/47%/88%/157% recovery ladder rests on one random draw per
   horizon. The seeded k=2 triple (sd 0.0010) is the containment.
4. **The §2.36 scalar contrast has no clip-clustered CI and cannot currently
   get one** (§4 finding 5) — `probe_rigid_ci.py` lacks the
   `--tactile-reduce`/`--encoder-seed`/`--random-tactile-encoder` flags and
   silently clip-splits; "7.6 sd" from an n=3 mean/sd with no sampling
   interval is the number a statistics-minded reviewer will go after first.
5. **Best-axis-on-eval selection in the probe** (§4 finding 4) — bounded (the
   pick is stable across all 12 committed JSONs), but fixable for free by
   pre-registering curl or reporting the axis-mean.
6. **Three held-out scenes per partition** — every participant-disjoint claim
   ultimately generalizes over 3 held-out person-locations × 4 partitions.
   Stated in the docs; keep it stated in the paper.
7. **STAG floor** — already loudly caveated; keep the ~76% comparison visible
   so no reviewer discovers it for you.

**Doc/code fixes to make (all small):** §2.29 monotonicity sentence (D1);
§2.31 k=2 "resolved" cell (D2); "~4x" → "3–4x" (D3); regenerate the synthetic
noise floor as a committed, parameterised test or drop the 22.4% figure (D4);
add the upper-bound sentence to the docs **and fix the inverted prose at
`rotation_share.py:5-8`** (D6a, §4 finding 2); add the single-run caveat to
§2.1/§2.15 or land the 3-seed avg-pool baseline (D6b); state clip-split/val in
§2.36; add a finiteness guard before the batched SVD (§4 finding 1); fix
`shortcut_analysis.py`'s missing `tactile_reduce`/split provenance and its
`:206` cross-reference, and give the shortcut result a HANDOFF section once
its JSONs are committed; add `--split-group-by` (and the ablation flags) to
`probe_rigid_ci.py`.
