# OpenTouch GRU Fork: Experiments and Results

All experiments run on MIB cluster (`mib.media.mit.edu`, username `bashark`).
Code: [bkabbarah/opentouch](https://github.com/bkabbarah/opentouch)

---

## Baseline Architecture

The OpenTouch model aligns three modalities in a shared 64-dim embedding space using InfoNCE contrastive loss:

- **Visual encoder**: DINOv3-ViT-B/16, frozen
- **Tactile encoder**: 3-layer CNN + 2-layer bidirectional GRU (hidden dim 120)
- **Pose encoder**: 4-layer MLP per frame + temporal average pooling
- **Fusion heads**: `nn.Linear(128, 64)` for trimodal queries

Baseline numbers match the paper (Table 2, Raw Continuous, N=20).

---

## Experiment 1: GRU Pose Encoder

### Motivation

The tactile encoder models temporal dynamics via a bidirectional GRU. The pose encoder discarded temporal structure by averaging all T=20 frame embeddings. This asymmetry forces InfoNCE to align a temporally-aware tactile embedding against a temporally-flat pose embedding -- the loss has no gradient signal for temporal structure in pose space.

### Change

Replaced average pooling in the pose encoder with a bidirectional GRU matching the tactile encoder exactly: hidden dim 120, 2 layers, bidirectional, `batch_first=True`. Projection updated from `nn.Linear(128, 64)` to `nn.Linear(240, 64)` to accommodate concatenated forward/backward hidden states. No other changes.

### Single-task results (mAP %, test set, N=1399)

| Task | Paper | GRU (seed 42) | Delta |
|---|---|---|---|
| V->T | 15.47 | 17.48 | +2.01 |
| T->V | 15.28 | 16.75 | +1.47 |
| T->P | 13.43 | 45.46 | +32.03 |
| P->T | 13.13 | 44.39 | +31.26 |
| V->P | 19.01 | 21.36 | +2.35 |
| P->V | 19.98 | 20.93 | +0.95 |
| VP->T | 26.86 | 58.88 | +32.02 |
| TP->V | 23.46 | 26.99 | +3.53 |
| VT->P | 26.86 | 58.45 | +31.59 |

Note: V->T and T->V gains (+1-2 points) are within single-seed variance and should not be attributed to the GRU change, which does not affect the visual or tactile encoders.

### Multi-seed validation (fixed split-seed=42)

All three models evaluated on identical test clips (11985 train / 1572 val / 1399 test). The data split seed is fixed independently of the training seed via `--split-seed 42`.

| Task | Seed 0 | Seed 1 | Seed 42 | Mean | Std |
|---|---|---|---|---|---|
| T->P | 46.28 | 46.76 | 45.46 | 46.17 | 0.54 |
| P->T | 46.72 | 45.79 | 44.39 | 45.63 | 0.96 |
| VT->P | 58.99 | 57.12 | 58.45 | 58.19 | 0.79 |
| P->VT | 57.93 | 55.65 | 57.16 | 56.91 | 0.95 |
| VP->T | 59.21 | 58.05 | 58.88 | 58.71 | 0.49 |
| T->VP | 57.67 | 57.78 | 56.97 | 57.47 | 0.36 |

Std under 1 mAP point across all tasks. The improvement is stable and not seed-dependent.

---

## Experiment 2: Joint Multi-task Training

### Motivation

The paper trains a separate model per task, producing specialized models but no general encoder. Training all 6 task pairs simultaneously forces the three encoders to jointly satisfy all contrastive objectives.

### Change

Added `--task-type all` which computes InfoNCE loss for all 6 task pairs per batch and sums them. Eval script updated to report all 12 retrieval directions in one pass.

### Results (mAP %, test set)

| Task | Vanilla joint (avg pool) | GRU joint | GRU joint + pretrained tactile |
|---|---|---|---|
| V->T | 15.62 | 14.46 | 17.57 |
| T->V | 14.78 | 14.42 | 17.11 |
| P->T | 13.43 | 35.26 | 38.26 |
| T->P | 13.35 | 35.87 | 38.81 |
| V->P | 17.55 | 17.99 | 18.05 |
| P->V | 17.01 | 17.07 | 17.66 |
| VP->T | 28.47 | 57.67 | 58.88 |
| T->VP | 27.32 | 55.74 | 58.81 |
| TP->V | 27.72 | 27.41 | 28.70 |
| V->TP | 27.23 | 27.29 | 28.32 |
| VT->P | 28.45 | 53.99 | 54.69 |
| P->VT | 27.65 | 53.51 | 54.49 |

The vanilla joint baseline confirms joint training alone does not explain the pose-related gains. V->T and T->V are flat across all three conditions. Every pose-involved task jumps 20-30 points when the GRU encoder is used.

---

## Experiment 3: Tactile Autoencoder Pretraining

### Setup

A convolutional autoencoder (same 3-layer CNN encoder as the tactile encoder, mirrored transposed-conv decoder) was pretrained with MSE reconstruction loss on:

- OpenTouch: 327,030 frames, 16x16 piezoresistive pressure maps
- STAG dataset: 135,187 frames, 32x32 piezoresistive glove data downsampled to 16x16

100 epochs, final reconstruction loss: 0.000462. Pretrained CNN weights used to initialize the tactile encoder before contrastive joint training.

### Result

Adds 1-3 mAP points across most directions in the joint training setting (see joint results table above). Effect is small.

### Why the effect is small

The tactile encoder is a 3-layer CNN trained on 327K examples. It learns pressure map structure within the first few epochs of contrastive training regardless of initialization, leaving little room for pretraining to help. Pretraining is most useful for large models with limited labeled data -- the opposite of this setting.

---

## Experiment 4: Nonlinear Fusion Head (Negative Result)

### Change

Replaced the linear fusion head `nn.Linear(embed_dim*2, embed_dim)` for trimodal queries with a 2-layer MLP (`Linear -> ReLU -> Linear`), using `--fusion-head-type nonlinear`. GRU pose encoder and all other components held fixed. Single seed (42), split-seed 42.

### Result

Uniformly worse across all six trimodal directions.

| Task | Linear (seed 42) | Nonlinear (seed 42) | Delta |
|---|---|---|---|
| VT->P | 58.45 | 56.71 | -1.74 |
| P->VT | 57.16 | 56.03 | -1.13 |
| VP->T | 58.88 | 55.89 | -2.99 |
| T->VP | 56.97 | 55.06 | -1.91 |
| TP->V | 26.99 | 25.70 | -1.29 |
| V->TP | 26.70 | 26.51 | -0.19 |

### Interpretation

The two embeddings entering the fusion head have already been shaped by InfoNCE to be linearly comparable -- that is what the loss optimizes for directly (cosine similarity). A linear projection is sufficient for combining two already-aligned vectors. Adding nonlinear capacity does not unlock a real relationship to model; it adds parameters that overfit the trimodal training subset, which is smaller than the bimodal subsets because trimodal batches require all three modalities aligned simultaneously.

Note: V->TP shows a marginal drop (-0.19) that is not distinguishable from run-to-run noise at single-seed resolution. The other five directions provide the evidence for this conclusion.

This was treated as a confirmatory check of a pre-stated negative prediction, not a novel positive claim. Single-seed evidence is sufficient for that purpose.

---

## Failure Mode Analysis

Per-query rank analysis on the GRU single-task p2t checkpoint (N=1399 test samples):

- Median rank: 3 (more than half of queries retrieve the correct match in top 3)
- Mean rank: 15.1 (pulled up by outliers)
- Max rank: 420

The worst 100 queries span diverse scenes, actions, and grip types with no systematic pattern. Remaining failures are concentrated in brief or low-contact interactions (pressing, switching, placing) where tactile and pose signals are genuinely similar across different clips. A detailed per-query breakdown is in `failure_analysis_100.csv`.

---

## What Is Not Done

- Bootstrap confidence intervals: script (`bootstrap_eval.py`) written but not yet executed
- LoRA fine-tuning of DINOv3: would address the visual encoder bottleneck for V->P and P->V; not pursued per supervisor direction (risk of overfitting to dataset)
- VLA integration: downstream robot policy learning is out of scope for this internship period
- Two-glove, hat-mounted-camera pipeline: data collection started, modeling not yet begun
