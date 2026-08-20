# Retrieval seed-run provenance (clip-disjoint, paper table)

Written 2026-08-19, after the verification pass for `retrieval_section.tex`.
This file exists because the retrieval checkpoints predate two metadata
fields (`split_group_by`, and the `--split-seed`/`--seed` decoupling), so the
run-to-seed mapping could not be read off `params.txt` and had to be
established empirically. This is the record of how.

## The mapping

| arm | encoder seed | run directory (cluster, `logs/`) | committed eval artifact |
|---|---|---|---|
| biGRU | 42 | `2026_06_22-21_01_54-model_OpenTouch-DINOv3-B16-Retrieval-lr_0.0001-b_256-j_8-p_amp` | `results_retrieval_clip_gru_s42_test.json` |
| biGRU | 0 | `2026_07_06-22_35_40-…` | `results_retrieval_clip_gru_s0_test.json` |
| biGRU | 1 | `2026_07_06-22_36_12-…` | `results_retrieval_clip_gru_s1_test.json` |
| avg-pool | 42 | `clip_avgpool_s42` | `results_retrieval_clip_avgpool_s42_test.json` |
| avg-pool | 0 | `clip_avgpool_s0` | `results_retrieval_clip_avgpool_s0_test.json` |
| avg-pool | 1 | `clip_avgpool_s1` | `results_retrieval_clip_avgpool_s1_test.json` |

Every committed artifact carries a `_meta` block naming the exact checkpoint
it was produced from (`epoch_300.pt` in all six cases), the split (`test`),
and the gallery (`split_group_by: clip`, passed explicitly).

## How the biGRU seed runs were identified

`params.txt` for the biGRU runs records `seed` but not `split_group_by`
(the field postdates them). Two *pairs* of candidate seed-0/seed-1 runs
existed, launched 2026-07-06 at 15:20/15:21 and at 22:35/22:36. On
2026-08-19 every candidate checkpoint was re-evaluated on the standard
seed-42 clip-disjoint test gallery:

| run | T→P mAP | verdict |
|---|---|---|
| 22:35 (seed 0) | 46.2843 | matches `multiseed_table.csv` seed-0 column (46.28) |
| 22:36 (seed 1) | 46.7619 | matches CSV seed-1 column (46.76) |
| 21:01 (seed 42) | 45.4634 | matches CSV default column (45.46) |
| 15:20 (seed 0) | **72.53** | contaminated — see below |
| 15:21 (seed 1) | **71.66** | contaminated — see below |

## Why the 15:2x pair is contaminated, and what was done

Commit `028eba5` ("Decouple train/val/test split seed from training seed")
landed **2026-07-06 22:32** — after the 15:2x pair launched and three minutes
before the 22:3x pair. Before that commit, `--seed` also reseeded the data
split, so the 15:2x runs trained on a *different partition*: their training
clips are inside the standard seed-42 test gallery, which is why they score
~72 rather than ~46. This is the same failure signature as the wrong-gallery
trap (`results_bootstrap_scene_gru_clipgallery_demo_test.json`, 72.088).

Both directories were renamed with a `QUARANTINED_` prefix on the cluster and
carry a `README_QUARANTINED.txt` stating the above. No script or document
referenced them (verified by grep before renaming). The three true biGRU
runs carry a `PROVENANCE_NOTE.txt`.

## `multiseed_table.csv`

Copied into `results/` from `/scratch/bashar/multiseed_table.csv` on
2026-08-19 so the historical record survives the cluster. Notes:

- Its `gru_single_p2t*` columns are reproduced to 4 decimals by the committed
  per-seed JSONs, which supersede it as the citable source.
- Its `avgpool_single_p2t` column (16.76 / 15.64) is the **orphan single
  run** whose training provenance was never identified (HANDOFF §2.20
  caveat). The paper's avg-pool numbers come from the fully-provenanced
  `clip_avgpool_s{42,0,1}` runs instead; the orphan values are kept only as
  history.
- The V↔T / V↔P columns are empty in the source and always were.

## Split-membership argument for the pre-field runs

The biGRU runs' clip-split membership rests on two facts: (1) at their launch
dates no other split existed in the codebase (the scene split and the
`split_group_by` field postdate them), and (2) the empirical gallery match
above — a wrong-partition checkpoint scores ~72 on this gallery, and these
score ~46 with seed-to-seed agreement against the CSV at 4 decimals. The
avg-pool runs postdate the field and record `split_group_by: clip` directly.
