"""Tests for train/val/test partitioning in opentouch_train.data.

The clip-level split was previously untested despite underpinning every
result. These pin both the historical behaviour (so the `--split-group-by
scene` addition provably changes nothing by default) and the new
scene-disjoint mode, whose whole purpose is to make participants disjoint
across splits.
"""

from __future__ import annotations

import pytest

from opentouch_train.data import split_clips_into_train_val_test


def clip_keys(n_scenes: int, clips_per_scene: int):
    return [
        (f"scene_{s:02d}", f"demo_{c:03d}")
        for s in range(n_scenes)
        for c in range(clips_per_scene)
    ]


# --------------------------------------------------------------------------
# Behaviour shared by both modes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("group_by", ["clip", "scene"])
def test_splits_are_disjoint_and_cover_every_clip(group_by):
    keys = clip_keys(20, 6)
    splits = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=42, group_by=group_by)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(keys), "no clip may be dropped"
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(keys)


@pytest.mark.parametrize("group_by", ["clip", "scene"])
def test_split_is_deterministic_given_a_seed(group_by):
    keys = clip_keys(15, 5)
    first = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=7, group_by=group_by)
    second = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=7, group_by=group_by)
    assert first == second


@pytest.mark.parametrize("group_by", ["clip", "scene"])
def test_different_seeds_give_different_splits(group_by):
    keys = clip_keys(20, 5)
    a = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=1, group_by=group_by)
    b = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=2, group_by=group_by)
    assert a["test"] != b["test"]


def test_empty_input_is_handled():
    splits = split_clips_into_train_val_test([], 0.1, 0.1, seed=42)
    assert splits == {"train": [], "val": [], "test": []}


def test_invalid_group_by_raises():
    with pytest.raises(ValueError, match="group_by"):
        split_clips_into_train_val_test(clip_keys(3, 3), 0.1, 0.1, seed=42, group_by="participant")


# --------------------------------------------------------------------------
# Backwards compatibility: the default must reproduce historical behaviour
# --------------------------------------------------------------------------


def test_default_group_by_is_clip_and_matches_the_explicit_call():
    """Every published number was produced by the clip path. If the default
    changed, results would silently stop being comparable."""
    keys = clip_keys(12, 8)
    implicit = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=42)
    explicit = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=42, group_by="clip")
    assert implicit == explicit


def test_clip_mode_ratios_are_respected():
    keys = clip_keys(20, 10)  # 200 clips
    splits = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=42, group_by="clip")
    assert len(splits["val"]) == 20
    assert len(splits["test"]) == 20
    assert len(splits["train"]) == 160


# --------------------------------------------------------------------------
# The point of the new mode
# --------------------------------------------------------------------------


def test_clip_mode_leaks_scenes_across_splits():
    """Documents the property that motivates the scene mode: under clip-level
    splitting essentially every held-out scene is also a training scene, so a
    high-capacity encoder can memorize participant-specific hand geometry and
    glove calibration and reapply it at eval."""
    keys = clip_keys(20, 10)
    splits = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=42, group_by="clip")
    train_scenes = {scene for scene, _ in splits["train"]}
    test_scenes = {scene for scene, _ in splits["test"]}
    assert test_scenes & train_scenes, "clip-level splitting is expected to share scenes"
    assert test_scenes <= train_scenes, (
        "with 10 clips per scene, every test scene should also appear in train"
    )


def test_scene_mode_makes_scenes_disjoint():
    keys = clip_keys(20, 10)
    splits = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=42, group_by="scene")
    train_scenes = {scene for scene, _ in splits["train"]}
    val_scenes = {scene for scene, _ in splits["val"]}
    test_scenes = {scene for scene, _ in splits["test"]}
    assert not (train_scenes & val_scenes)
    assert not (train_scenes & test_scenes)
    assert not (val_scenes & test_scenes)


def test_scene_mode_keeps_every_clip_of_a_scene_together():
    keys = clip_keys(15, 7)
    splits = split_clips_into_train_val_test(keys, 0.2, 0.2, seed=3, group_by="scene")
    where = {}
    for name, members in splits.items():
        for scene, _ in members:
            where.setdefault(scene, set()).add(name)
    for scene, names in where.items():
        assert len(names) == 1, f"scene {scene} was split across {sorted(names)}"


def test_scene_mode_is_invariant_to_input_ordering():
    """The clip path shuffles whatever list order it is handed, so the
    partition depends on HF row order. The scene path sorts first, so a
    rebuilt dataset with reshuffled rows yields the SAME split at the same
    seed."""
    keys = clip_keys(18, 4)
    shuffled = list(reversed(keys))
    a = split_clips_into_train_val_test(keys, 0.1, 0.1, seed=42, group_by="scene")
    b = split_clips_into_train_val_test(shuffled, 0.1, 0.1, seed=42, group_by="scene")
    assert {s for s, _ in a["test"]} == {s for s, _ in b["test"]}
    assert {s for s, _ in a["val"]} == {s for s, _ in b["val"]}


def test_scene_mode_handles_uneven_scene_sizes():
    """Real scenes differ in clip count, so clip-level ratios will not be hit
    exactly -- the split must still be valid and complete."""
    keys = (
        [("big", f"c{i}") for i in range(50)]
        + [("small", "c0")]
        + [(f"mid_{s}", f"c{i}") for s in range(8) for i in range(5)]
    )
    splits = split_clips_into_train_val_test(keys, 0.2, 0.2, seed=11, group_by="scene")
    assert sum(len(v) for v in splits.values()) == len(keys)
    all_scenes = [{s for s, _ in v} for v in splits.values()]
    assert not (all_scenes[0] & all_scenes[1]) and not (all_scenes[0] & all_scenes[2])


def test_scene_mode_with_one_scene_per_split_minimum():
    keys = clip_keys(3, 4)
    splits = split_clips_into_train_val_test(keys, 0.34, 0.33, seed=5, group_by="scene")
    assert sum(len(v) for v in splits.values()) == len(keys)
    assert len(splits["train"]) > 0
