"""Data loading for the pose-transition regression task.

Reuses opentouch_train.data.VideoTactilePoseDataset and
_load_and_split_dataset UNMODIFIED for windowing and the clip-level
train/val/test split (--split-seed) -- this module only adds a thin
(t, t+k) sampling layer on top of the existing T=20-frame windows built by
data.py's VideoTactilePoseDataset._build_sliding_windows(). It does not
touch data.py.

PoseTransitionDataset is deliberately agnostic to --target-mode: every
sample carries the raw "world_delta" (pose[t+k]-pose[t]); the world_delta /
articulation_delta split (opentouch.pose_regression.decompose_world_delta)
happens in the training/eval loop, which needs both spaces at eval time
regardless of which one was trained on.

SHUFFLED-TACTILE CONTROL: PoseTransitionDataset(shuffle_tactile=True) pairs
each window's pose with a DIFFERENT window's tactile (same split, tactile
window index != pose window index), fixed once at construction time via a
deterministic derangement of the split's window indices (_make_derangement).
This is a fixed re-pairing, not a per-batch shuffle: the same (pose window,
tactile window) pairing is used for every epoch and by any later
regression_eval.py run given the same shuffle_seed, so train and eval always
see identical corrupted correspondence. The model architecture is completely
unchanged (still PoseTransitionRegressor(use_tactile=True, ...)) -- only
which window's tactile tensor gets paired with which window's pose changes
-- so its parameter count is exactly that of the real tactile+pose model by
construction, not by a separate capacity-matching step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from opentouch.pose_regression import decompose_world_delta
from opentouch.regression_metrics import fingertip_displacement
from opentouch_train.data import DataInfo, VideoTactilePoseDataset, _load_and_split_dataset

logger = logging.getLogger(__name__)


def _make_derangement(n: int, seed: int) -> np.ndarray:
    """Deterministic derangement (permutation of range(n) with NO fixed
    points) given a seed. Used so that pairing window i's pose with window
    perm[i]'s tactile never accidentally leaves a window paired with its own
    (real) tactile.

    Rejection sampling rather than a hand-rolled fixup: simpler to prove
    correct (the returned array is always re-checked for fixed points
    before returning), and for n >= 2 a uniformly random permutation is a
    derangement with probability ~1/e, so this converges in a handful of
    draws in practice.
    """
    if n < 2:
        raise ValueError(f"cannot build a derangement for n={n} windows (need >= 2)")
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    raise RuntimeError(f"failed to construct a derangement of {n} elements after 1000 attempts")


def _materialize_pose_and_tactile(
    base_dataset: VideoTactilePoseDataset, include_tactile: bool,
):
    """Iterate base_dataset's windows ONCE and pull every window's
    hand_landmarks (and, if include_tactile, tactile_pressure) into a
    single contiguous in-memory tensor indexed by window_idx.

    PoseTransitionDataset.__getitem__ then only ever indexes into these
    tensors -- it never calls base_dataset again -- which is what removes
    the ~19x-per-epoch full-window reconstruction (restacking HF rows,
    building tensors) base_dataset[window_idx] used to trigger on every
    single (window, t) access. Pose is small enough to hold whole (roughly
    13k windows x 20 x 21 x 3 x 4 bytes ~= 130MB for a train split);
    tactile is the same order of magnitude. RGB is never materialized here
    -- base_dataset must be built with include_visual=False so no image
    decode/transform work happens even during this one-time pass.
    """
    n_windows = len(base_dataset)
    first = base_dataset[0]
    seq_len, _, n_keypoints, coord_dim = first["hand_landmarks"].shape
    pose = torch.empty(n_windows, seq_len, n_keypoints, coord_dim, dtype=torch.float32)
    pose[0] = first["hand_landmarks"].squeeze(1)

    tactile: Optional[torch.Tensor] = None
    if include_tactile:
        tactile = torch.empty((n_windows, *first["tactile_pressure"].shape), dtype=torch.float32)
        tactile[0] = first["tactile_pressure"]

    scenes = [first["scene"]] * n_windows
    clip_ids = [first["clip_id"]] * n_windows

    for window_idx in range(1, n_windows):
        window = base_dataset[window_idx]
        pose[window_idx] = window["hand_landmarks"].squeeze(1)
        if include_tactile:
            tactile[window_idx] = window["tactile_pressure"]
        scenes[window_idx] = window["scene"]
        clip_ids[window_idx] = window["clip_id"]

    return pose, tactile, scenes, clip_ids


class PoseTransitionDataset(Dataset):
    """Wraps a VideoTactilePoseDataset and enumerates every valid (window, t)
    pair for a fixed horizon k: t ranges over [0, sequence_length - 1 - k]
    so both frame t and frame t+k fall inside the SAME existing T=20-frame
    window. data.py's windowing is untouched; this only indexes further
    into windows it already built.
    """

    def __init__(
        self,
        base_dataset: VideoTactilePoseDataset,
        horizon_k: int,
        shuffle_tactile: bool = False,
        shuffle_seed: int = 42,
    ) -> None:
        if horizon_k < 1:
            raise ValueError(f"horizon_k must be >= 1, got {horizon_k}")
        if horizon_k >= base_dataset.sequence_length:
            raise ValueError(
                f"horizon_k={horizon_k} must be < sequence_length="
                f"{base_dataset.sequence_length} so that frame t+k stays inside the window"
            )
        if not base_dataset.include_pose:
            raise ValueError("PoseTransitionDataset requires include_pose=True on the base dataset")
        if shuffle_tactile and not base_dataset.include_tactile:
            raise ValueError(
                "shuffle_tactile=True requires the base dataset to include real tactile "
                "data (include_tactile=True) -- shuffled tactile is still REAL tactile, "
                "just mispaired; it is not the pose-only path"
            )

        self.base_dataset = base_dataset
        self.horizon_k = horizon_k
        self.include_tactile = base_dataset.include_tactile
        self.valid_t_per_window = base_dataset.sequence_length - horizon_k
        self.shuffle_tactile = shuffle_tactile

        self.tactile_window_permutation: Optional[np.ndarray] = None
        if shuffle_tactile:
            self.tactile_window_permutation = _make_derangement(len(base_dataset), shuffle_seed)

        # Materialized ONCE here: __getitem__ below never touches
        # base_dataset again (see _materialize_pose_and_tactile's docstring
        # for why this is the actual bottleneck fix).
        self._pose, self._tactile, self._scenes, self._clip_ids = _materialize_pose_and_tactile(
            base_dataset, self.include_tactile,
        )

    def __len__(self) -> int:
        return len(self.base_dataset) * self.valid_t_per_window

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        window_idx, t = divmod(idx, self.valid_t_per_window)

        pose_t = self._pose[window_idx, t]
        pose_future = self._pose[window_idx, t + self.horizon_k]
        world_delta = pose_future - pose_t

        sample: Dict[str, Any] = {
            "pose_t": pose_t,
            "world_delta": world_delta,
            "scene": self._scenes[window_idx],
            "clip_id": self._clip_ids[window_idx],
            "t": t,
            "horizon_k": self.horizon_k,
        }
        if self.include_tactile:
            if self.shuffle_tactile:
                assert self.tactile_window_permutation is not None
                shuffled_idx = int(self.tactile_window_permutation[window_idx])
                assert shuffled_idx != window_idx, (
                    "derangement invariant violated: a window was paired with its own tactile"
                )
                tactile_window_idx = shuffled_idx
            else:
                tactile_window_idx = window_idx
            sample["tactile_pressure"] = self._tactile[tactile_window_idx]
        return sample


def compute_motion_threshold(dataset: PoseTransitionDataset, percentile: float = 25.0) -> float:
    """The default --motion-threshold: the given percentile of median-
    fingertip ARTICULATION displacement (not raw world-space displacement,
    which is ~76% wrist/arm translation at this dataset's scale -- see
    opentouch.pose_regression's module docstring) over `dataset`'s split.

    Reuses PoseTransitionDataset's already-materialized pose tensor
    (dataset._pose, built once in __init__ by _materialize_pose_and_tactile)
    and computes every valid t's delta in one vectorized shot
    (landmarks[:, k:] - landmarks[:, :-k]), rather than calling dataset[i]
    in a loop or re-iterating base_dataset -- both of which would redo work
    __init__ already did. Intended to be called ONCE (by regression_main.py,
    before the epoch loop) and the result cached in args.motion_threshold /
    checkpoint metadata, not recomputed per epoch.
    """
    horizon_k = dataset.horizon_k
    landmarks = dataset._pose  # (N_windows, T, 21, 3)
    world_delta = landmarks[:, horizon_k:] - landmarks[:, :-horizon_k]  # (N_windows, T-k, 21, 3)
    world_delta = world_delta.reshape(-1, world_delta.shape[-2], world_delta.shape[-1])
    _, articulation_delta = decompose_world_delta(world_delta)
    all_displacements = fingertip_displacement(articulation_delta)
    return torch.quantile(all_displacements, percentile / 100.0).item()


def regression_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}
    collated: Dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        else:
            collated[key] = values
    return collated


def _attach_loader_metadata(loader: DataLoader, dataset: PoseTransitionDataset) -> None:
    loader.num_samples = len(dataset)
    loader.num_batches = len(loader)


def get_regression_data(args, epoch: int = 0) -> Dict[str, DataInfo]:
    """Build train/val PoseTransitionDataset DataLoaders.

    Mirrors opentouch_train.data.get_data()'s exact
    _load_and_split_dataset(...) + VideoTactilePoseDataset(_preloaded=...)
    pattern, so this task's train/val/test clips are the exact same split
    (given the same --split-seed) as the retrieval and classification
    pipelines use -- not a re-derived split.

    --shuffle-tactile applies to BOTH train and val here (not just train):
    the shuffled-tactile control needs val metrics evaluated under the same
    corrupted correspondence it trained on, or the comparison would silently
    mix a shuffled-tactile model with real-tactile validation.
    """
    data: Dict[str, DataInfo] = {}
    dataset_path = getattr(args, "train_data", None)
    if dataset_path is None:
        return data

    horizon_k = args.horizon_k
    pose_only = args.pose_only
    shuffle_tactile = getattr(args, "shuffle_tactile", False)
    seq_len = getattr(args, "sequence_length", 20)
    val_ratio = getattr(args, "val_ratio", 0.1)
    test_ratio = getattr(args, "test_ratio", 0.1)
    seed = getattr(args, "split_seed", 42)

    splits = _load_and_split_dataset(dataset_path, val_ratio, test_ratio, seed)

    common_kwargs = dict(
        hf_dataset_path=dataset_path,
        sequence_length=seq_len,
        include_tactile=not pose_only,
        include_visual=False,
        include_pose=True,
    )

    train_preloaded = splits.get("train")
    train_base = VideoTactilePoseDataset(split="train", _preloaded=train_preloaded, **common_kwargs)
    train_dataset = PoseTransitionDataset(
        train_base, horizon_k, shuffle_tactile=shuffle_tactile, shuffle_seed=seed,
    )

    train_sampler = None
    if getattr(args, "distributed", False):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), num_workers=args.workers,
        pin_memory=True, sampler=train_sampler, drop_last=True,
        collate_fn=regression_collate_fn, persistent_workers=args.workers > 0,
    )
    _attach_loader_metadata(train_loader, train_dataset)
    data["train"] = DataInfo(dataloader=train_loader, sampler=train_sampler)

    val_preloaded = splits.get("val")
    if val_preloaded is not None and len(val_preloaded) > 0:
        val_base = VideoTactilePoseDataset(split="val", _preloaded=val_preloaded, **common_kwargs)
        val_dataset = PoseTransitionDataset(
            val_base, horizon_k, shuffle_tactile=shuffle_tactile, shuffle_seed=seed,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False,
            collate_fn=regression_collate_fn, persistent_workers=args.workers > 0,
        )
        _attach_loader_metadata(val_loader, val_dataset)
        data["val"] = DataInfo(dataloader=val_loader)

    return data
