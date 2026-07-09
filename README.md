# OPENTOUCH: Bringing Full-Hand Touch to Real-World Interaction
[Project](https://opentouch-tactile.github.io/) | [Paper](https://arxiv.org/abs/2512.16842) | [Hardware](https://wiresens-gloves.vercel.app/) | [Dataset](scripts/download_data.sh)

OpenTouch is an egocentric in-the-wild dataset and cross-modal learning framework for visual (RGB), tactile (pressure), and hand-pose modalities.

## Dataset

The OpenTouch data is organized as synchronized multimodal recordings:
- egocentric RGB video streams
- full hand tactile pressure maps
- hand pose

The dataset is hosted on Google Drive. We use [gdown](https://github.com/wkentaro/gdown) to download all files:

```bash
pip install gdown
bash scripts/download_data.sh
cd data && unzip final_annotations.zip && cd ..
```
See [`scripts/download_data.sh`](scripts/download_data.sh) for the full list of Google Drive file IDs.

## Environment Setup

```bash
conda create -n opentouch python=3.10
conda activate opentouch
pip install -e .
```

### MANO Mesh Visualization (Optional)

The rendering scripts require extra dependencies:

```bash
git submodule update --init --recursive
pip install -e ".[rendering]"
cd EasyMocap && pip install -e . && cd ..
```

You also need the MANO hand model:

1. Download `MANO_RIGHT.pkl` from the [MANO project](https://mano.is.tue.mpg.de/)
2. Place it in `preprocess/scratch/MANO_RIGHT.pkl`

```bash
# Generate a synchronized visualization from an HDF5 recording:
python preprocess/build_demo.py \
    --hdf5 data/fablab_ml_p1.hdf5 \
    --demo-id demo_05 \
    --fps 30
```

<img src="assets/fablab_ml_p1_demo_05_tri.gif" alt="Demo visualization (RGB + tactile MANO + hand pose)" width="320">

Example output: simple RGB+tactile view and tri-view with MANO/pose rendering.

Output path: `data/<dataset_name>/<demo_id>/combined.mp4`

### Convert HDF5 to Arrow Dataset
```bash
# Retrieval dataset
python build_retrieval_data.py \
    --input-dir data \
    --output-dir preprocessed_data/train_dataset

# Classification dataset
python build_label_data.py \
    --input-dir data \
    --output-dir preprocessed_data/classification_peak \
    --label-mapping-path final_annotations \
    --label-column action \
    --frame-index-column peak_idx \
    --temporal-radius 10
```
## Model Backbone
The default visual backbone is DINOv3 ViT-B/16 (`facebook/dinov3-vitb16-pretrain-lvd1689m`).
Access to this model may require approval from Meta. Please refer to [DINOv3](https://github.com/facebookresearch/dinov3) for more details.

## Retrieval
```bash
bash scripts/train.sh
```
Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m opentouch_train.main \
    --train-data preprocessed_data/train_dataset \
    --model OpenTouch-DINOv3-B16-Retrieval \
    --task-type v2t \
    --batch-size 128 \
    --lr 1e-4 \
    --epochs 300 \
    --precision amp \
    --workers 8 \
    --sequence-length 20
```

If you want to train with multiple GPUs, use distributed data parallel (DDP): please see [`scripts/train_multigpu.sh`](scripts/train_multigpu.sh) for the full reference configuration.

## Task Types
Set `--task-type` to choose the retrieval task:
| Task Type | Description |
| --- | --- |
| `v2t` | Visual $\leftrightarrow$ tactile |
| `p2t` | Pose $\leftrightarrow$ tactile |
| `v2p` | Visual $\leftrightarrow$ pose |
| `vp2t` | Visual + pose $\leftrightarrow$ tactile |
| `tp2v` | Tactile + pose $\leftrightarrow$ visual |
| `vt2p` | Visual + tactile $\leftrightarrow$ pose |


## Classification
Train action or grip classifiers on top of the same encoders:

```bash
bash scripts/train_classifier.sh
```

Or run directly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m opentouch_train.classification_main \
    --train-data preprocessed_data/classification_peak \
    --model OpenTouch-DINOv3-B16-Classify \
    --task action \
    --modalities visual tactile \
    --batch-size 64 \
    --lr 3e-3 \
    --epochs 500 \
    --precision amp
```

### Classification Options

| Flag | Description |
| --- | --- |
| `--task` | Classification task: `action` or `grip` |
| `--modalities` | Input modalities: `visual`, `tactile`, `pose` (any combination).|

## Evaluation

Model name, task type, and modalities are auto-detected from the checkpoint or `params.txt`.

### Retrieval

```bash
bash scripts/eval.sh logs/<run_name>/checkpoints/epoch_<N>.pt
```

### Classification

```bash
bash scripts/eval_classifier.sh logs/<run_name>/checkpoints/epoch_<N>.pt
```

## Citation

If you find this work helpful, please consider citing:

```bibtex
@article{song2025opentouch,
  title={OPENTOUCH: Bringing Full-Hand Touch to Real-World Interaction},
  author={Song, Yuxin Ray and Li, Jinzhou and Fu, Rao and Murphy, Devin and Zhou, Kaichen and Shiv, Rishi and Li, Yaqi and Xiong, Haoyu and Owens, Crystal Elaine and Du, Yilun and others},
  journal={arXiv preprint arXiv:2512.16842},
  year={2025}
}
```

## Acknowledgments

This codebase builds on [OpenCLIP](https://github.com/mlfoundations/open_clip).

---

## Fork: GRU Pose Encoder Experiments

This section documents changes and results from [bkabbarah/opentouch](https://github.com/bkabbarah/opentouch). For the full results writeup see [`experiments.md`](experiments.md).

### Setup notes

**Critical**: always run `pip install -e .` from the repo directory before any training or eval launch. The conda environment is shared across multiple repo checkouts. Skipping this will silently use whichever package was last installed.

Verify the correct architecture is active before trusting any run:

```bash
python -c "from opentouch.pose_encoder import PoseEncoder; import inspect; print('gru' in inspect.getsource(PoseEncoder.__init__))"
# must print: True
```

**DINOv3 access**: request access to `facebook/dinov3-vitb16-pretrain-lvd1689m` on HuggingFace before running any task involving the visual modality. DINOv3 weights are not included in any checkpoint.

**HuggingFace cache**: set `HF_HOME=/scratch/bashar/.cache/huggingface` to avoid filling the home directory quota.

### New training flags

| Flag | Description |
|---|---|
| `--task-type all` | Train all 6 task pairs jointly in one run |
| `--split-seed 42` | Fix data split independently of training seed. Required for comparable multi-seed runs. |
| `--tactile-pretrained <path>` | Initialize tactile CNN from pretrained autoencoder weights |
| `--fusion-head-type nonlinear` | Replace linear fusion head with 2-layer MLP (tested, performs worse) |
| `--tags "<text>"` | Free-text label written to `tags.txt` in the log directory at launch |

### Multi-seed training

Always pass `--split-seed 42` alongside `--seed <N>`:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m opentouch_train.main \
    --train-data ../opentouch/preprocessed_data/train_dataset \
    --model OpenTouch-DINOv3-B16-Retrieval \
    --task-type p2t \
    --batch-size 256 \
    --lr 1e-4 \
    --epochs 300 \
    --precision amp \
    --workers 8 \
    --sequence-length 20 \
    --seed 0 \
    --split-seed 42 \
    --tags "experiment: gru_pose_encoder task: p2t seed: 0 split_seed: 42"
```

### Tactile autoencoder pretraining

```bash
python -m opentouch_train.pretrain_tactile \
    --data ../opentouch/preprocessed_data/train_dataset \
    --stag-data /scratch/bashar/stag/pressure_16x16.npy \
    --output checkpoints/tactile_pretrained_encoder.pt \
    --epochs 100 \
    --batch-size 512 \
    --lr 1e-3
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=<gpu> python -m opentouch_train.eval \
    --checkpoint <path>/checkpoints/epoch_latest.pt \
    --data ../opentouch/preprocessed_data/train_dataset \
    --split test \
    --batch-size 128 \
    --precision amp \
    --output results_<label>.json
```

The eval script automatically derives `enabled_modalities` from the checkpoint's saved task type.

### Results logging

```bash
python log_results.py results_<label>.json <run_label> \
    --checkpoint "<full_checkpoint_path>" \
    --epoch 300 \
    --task-type <task> \
    --notes "<description>"
```

Results are appended to `/scratch/bashar/results_log.csv` (append-only, never overwritten). Each entry records git commit hash, repo path, checkpoint path, epoch, tags, and timestamp.

To view as a pivot table:

```bash
cd ~/scratch/bashar
python pivot_results.py
python pivot_results.py --runs <label1> <label2> --csv-out table.csv
```

### Checkpoints

All on MIB cluster at `~/scratch/bashar/`. Verified by checking GRU presence and projection weight shape directly from state dict.

| Experiment | Repo | Log directory timestamp | Task | GRU | Notes |
|---|---|---|---|---|---|
| Paper baseline | opentouch | 2026_06_22-20_49_48 | p2t | No | T->P mAP 16.76, matches paper |
| GRU p2t seed 42 | opentouch-gru | 2026_06_22-21_01_54 | p2t | Yes | T->P mAP 45.46 |
| GRU joint | opentouch-gru | 2026_06_26-16_30_29 | all | Yes | T->P mAP 35.87 |
| GRU + pretrained joint | opentouch-gru | 2026_06_29-13_56_23 | all | Yes | T->P mAP 38.81 |
| Vanilla joint | opentouch | 2026_06_29-14_25_05 | all | No | Baseline for isolating GRU contribution |
| GRU v2t seed 42 | opentouch-gru | 2026_06_29-14_52_25 | v2t | N/A | V->T mAP 17.48 |
| GRU v2p seed 42 | opentouch-gru | 2026_06_30-14_07_44 | v2p | Yes | V->P mAP 21.36 |
| GRU vt2p seed 42 | opentouch-gru | 2026_06_30-14_08_06 | vt2p | Yes | VT->P mAP 58.45 |
| GRU vp2t seed 42 | opentouch-gru | 2026_06_30-16_46_48 | vp2t | Yes | VP->T mAP 58.88 |
| GRU tp2v seed 42 | opentouch-gru | 2026_06_30-16_47_39 | tp2v | Yes | TP->V mAP 26.99 |
| GRU p2t seed 0 | opentouch-gru | 2026_07_06-22_35_40 | p2t | Yes | Multi-seed eval |
| GRU p2t seed 1 | opentouch-gru | 2026_07_06-22_36_12 | p2t | Yes | Multi-seed eval |
| GRU vt2p seed 0 | opentouch-gru | 2026_07_06-22_40_57 | vt2p | Yes | Multi-seed eval |
| GRU vt2p seed 1 | opentouch-gru | 2026_07_07-10_54_17 | vt2p | Yes | Multi-seed eval |
| GRU vp2t seed 0 | opentouch-gru | 2026_07_07-10_54_54 | vp2t | Yes | Multi-seed eval |
| GRU vp2t seed 1 | opentouch-gru | 2026_07_07-14_33_49 | vp2t | Yes | Multi-seed eval |
| Nonlinear fusion vt2p | opentouch-gru | 2026_07_08-14_07_58 | vt2p | Yes | Negative result |
| Nonlinear fusion vp2t | opentouch-gru | 2026_07_08-14_09_08 | vp2t | Yes | Negative result |
| Nonlinear fusion tp2v | opentouch-gru | 2026_07_08-14_09_47 | tp2v | Yes | Negative result |

Pretrained tactile encoder weights: `/scratch/bashar/opentouch-gru/checkpoints/tactile_pretrained_encoder.pt`

HuggingFace (GRU p2t checkpoint, no visual encoder weights): [basharkMIT/opentouch-gru-pose-encoder](https://huggingface.co/basharkMIT/opentouch-gru-pose-encoder)

### Key files added vs upstream

| File | Description |
|---|---|
| `src/opentouch/pose_encoder.py` | GRU replaces average pooling |
| `src/opentouch/tactile_autoencoder.py` | Autoencoder for tactile pretraining |
| `src/opentouch/model_configs/OpenTouch-DINOv3-B16-Retrieval.json` | All 3 modalities enabled |
| `src/opentouch_train/train.py` | Joint multi-task training, `--task-type all` |
| `src/opentouch_train/eval.py` | All 12 directions in one pass, enabled_modalities fix |
| `src/opentouch_train/main.py` | `--task-type all`, `--tactile-pretrained`, `--tags`, `--split-seed`, `--fusion-head-type` |
| `src/opentouch_train/data.py` | Split seed decoupled from training seed |
| `src/opentouch_train/params.py` | All new args |
| `src/opentouch_train/pretrain_tactile.py` | Autoencoder pretraining script |
| `log_results.py` | Append-only provenance logging |
| `pivot_results.py` | Pivot results_log.csv into comparison table |
| `bootstrap_eval.py` | Bootstrap CI estimation (written, not yet run) |
| `experiments.md` | Full results writeup |
| `failure_analysis_100.csv` | Per-query failure analysis, worst 100 queries |
