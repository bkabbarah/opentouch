"""Tests for scripts/load_public_hands.py.

The load-bearing claim these tests defend: a WRONG joint layout must fail
loudly. rotation_share.py's own docstring says non-wrist ordering only has to
be "internally consistent", which is true of the Kabsch fit but NOT of
hand_frame_degenerate(), which indexes joints 5/9/17 by position to decide
which samples to throw away. A permuted layout therefore produces a
plausible-looking number computed over an arbitrary subset. The tests below
build a hand whose anatomy is known by construction and assert that the check
accepts the right layout and rejects the wrong one.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from load_public_hands import (  # noqa: E402
    FINGER_CHAINS,
    MANO_TO_MEDIAPIPE,
    NATIVE_ORDER,
    check_joint_order,
    load_dexycb,
    load_ho3d,
    reorder,
    split_on_gaps,
)


def synthetic_hand(n_frames=40, seed=0):
    """(T,21,3) in MediaPipe order, anatomically well-formed by construction.

    Wrist at the origin; five fingers fanned in the xy-plane; within each
    finger the four joints sit at increasing distance from the wrist. A little
    per-frame jitter and a global rotation keep it from being degenerate.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((21, 3), dtype=np.float32)
    for f in range(5):
        angle = -0.6 + 0.3 * f
        direction = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        for j, radius in enumerate((0.35, 0.6, 0.8, 1.0)):
            base[1 + 4 * f + j] = direction * radius
    # Push the palm anchors out of a single line so the palm triangle has area.
    base[FINGER_CHAINS[2][0]] += np.array([0.0, 0.12, 0.05], dtype=np.float32)

    frames = []
    for t in range(n_frames):
        theta = 0.02 * t
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        frames.append(base @ rot.T + 0.002 * rng.standard_normal((21, 3)).astype(np.float32))
    return np.stack(frames)


# --------------------------------------------------------------------------
# joint order
# --------------------------------------------------------------------------
def test_mano_to_mediapipe_is_a_permutation():
    assert sorted(MANO_TO_MEDIAPIPE.tolist()) == list(range(21))
    assert MANO_TO_MEDIAPIPE[0] == 0, "wrist must stay at index 0"


def test_check_accepts_correct_layout():
    diag = check_joint_order([synthetic_hand()], "synthetic")
    assert diag["monotone_chain_fraction"] > 0.95
    assert diag["median_palm_noncollinearity"] > 1e-2


def test_check_rejects_scrambled_layout():
    """The whole point of the file: a wrong permutation must raise."""
    hand = synthetic_hand()
    rng = np.random.default_rng(7)
    perm = np.concatenate([[0], 1 + rng.permutation(20)])
    with pytest.raises(ValueError, match="joint order looks WRONG"):
        check_joint_order([hand[:, perm, :]], "scrambled")


def test_check_rejects_mano_data_read_as_mediapipe():
    """The realistic failure: HO-3D/DexYCB ship MANO order. Reading those
    arrays as if they were already MediaPipe must not slip through."""
    mediapipe = synthetic_hand()
    # Invert the remap to manufacture the same hand in MANO order.
    inverse = np.argsort(MANO_TO_MEDIAPIPE)
    as_mano = mediapipe[:, inverse, :]

    # Declared correctly -> recovers the original exactly, and passes.
    assert np.allclose(reorder(as_mano, "mano"), mediapipe)
    check_joint_order([reorder(as_mano, "mano")], "declared-mano")

    # Declared wrongly -> must fail rather than return a plausible number.
    with pytest.raises(ValueError, match="joint order looks WRONG"):
        check_joint_order([reorder(as_mano, "mediapipe")], "declared-mediapipe")


def test_non_strict_mode_warns_instead_of_raising():
    hand = synthetic_hand()
    rng = np.random.default_rng(3)
    perm = np.concatenate([[0], 1 + rng.permutation(20)])
    diag = check_joint_order([hand[:, perm, :]], "scrambled", strict=False)
    assert diag["monotone_chain_fraction"] < 0.80


# --------------------------------------------------------------------------
# temporal gaps
# --------------------------------------------------------------------------
def test_split_on_gaps_breaks_at_discontinuity():
    j = lambda v: np.full((21, 3), v, dtype=np.float32)  # noqa: E731
    frames = [(0, j(0)), (1, j(1)), (2, j(2)), (7, j(7)), (8, j(8)), (9, j(9))]
    runs = split_on_gaps(frames, min_len=2)
    assert [len(r) for r in runs] == [3, 3]


def test_split_on_gaps_drops_short_runs():
    j = lambda v: np.full((21, 3), v, dtype=np.float32)  # noqa: E731
    frames = [(0, j(0)), (5, j(5)), (6, j(6)), (7, j(7))]
    assert [len(r) for r in split_on_gaps(frames, min_len=3)] == [3]


def test_split_on_gaps_sorts_unordered_input():
    j = lambda v: np.full((21, 3), v, dtype=np.float32)  # noqa: E731
    runs = split_on_gaps([(2, j(2)), (0, j(0)), (1, j(1))], min_len=3)
    assert len(runs) == 1 and runs[0][0][0, 0] == 0


# --------------------------------------------------------------------------
# HO-3D
# --------------------------------------------------------------------------
def _write_ho3d(root, seq_name, joints_per_frame):
    meta_dir = os.path.join(root, "train", seq_name, "meta")
    os.makedirs(meta_dir)
    for idx, joints in joints_per_frame:
        with open(os.path.join(meta_dir, "%04d.pkl" % idx), "wb") as fh:
            pickle.dump({"handJoints3D": joints, "camMat": np.eye(3)}, fh)


def test_load_ho3d_roundtrip(tmp_path):
    root = str(tmp_path / "HO3D_v3")
    inverse = np.argsort(MANO_TO_MEDIAPIPE)
    hand = synthetic_hand(30)[:, inverse, :]  # ship it in MANO order
    _write_ho3d(root, "ABF10", [(i, hand[i]) for i in range(30)])

    seqs, groups = load_ho3d(root, joint_order="mano")
    assert len(seqs) == 1 and seqs[0].shape == (30, 21, 3)
    assert groups == ["ABF10"]
    check_joint_order(seqs, "ho3d-fixture")  # must not raise


def test_load_ho3d_skips_missing_joints_and_splits(tmp_path):
    root = str(tmp_path / "HO3D_v3")
    inverse = np.argsort(MANO_TO_MEDIAPIPE)
    hand = synthetic_hand(30)[:, inverse, :]
    frames = [(i, hand[i]) for i in range(30)]
    frames[12] = (12, None)          # HO-3D's sentinel for an unannotated frame
    _write_ho3d(root, "MDF10", frames)

    seqs, _ = load_ho3d(root, min_len=5, joint_order="mano")
    # The dropped frame must break the recording in two, not silently join 11->13.
    assert len(seqs) == 2
    assert [len(s) for s in seqs] == [12, 17]


def test_load_ho3d_survives_a_stray_non_numeric_pkl(tmp_path):
    """This loader runs AFTER a 31.9 GB download. One oddly-named file must not
    take the whole run down with a ValueError from int()."""
    root = str(tmp_path / "HO3D_v3")
    inverse = np.argsort(MANO_TO_MEDIAPIPE)
    hand = synthetic_hand(20)[:, inverse, :]
    _write_ho3d(root, "ABF10", [(i, hand[i]) for i in range(20)])
    with open(os.path.join(root, "train", "ABF10", "meta", "README.pkl"), "wb") as fh:
        pickle.dump({"note": "not a frame"}, fh)

    seqs, groups = load_ho3d(root, joint_order="mano")
    assert len(seqs) == 1 and len(seqs[0]) == 20
    assert groups == ["ABF10"]


def test_load_ho3d_missing_root_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="Expected the extracted HO-3D layout"):
        load_ho3d(str(tmp_path / "nope"))


# --------------------------------------------------------------------------
# DexYCB
# --------------------------------------------------------------------------
def _write_dexycb(root, subject, take, cameras, joints_per_frame):
    for cam in cameras:
        cam_dir = os.path.join(root, subject, take, cam)
        os.makedirs(cam_dir)
        for idx, joints in joints_per_frame:
            np.savez(os.path.join(cam_dir, "labels_%06d.npz" % idx),
                     joint_3d=joints.reshape(1, 21, 3),
                     joint_2d=np.zeros((1, 21, 2), dtype=np.float32))


def test_native_orders_are_per_dataset():
    """The two datasets disagree, and the code must not paper over it.

    HO-3D stores MANO order (its vis_HO3D.py remaps train joints with
    jointsMapManoToSimple). DexYCB's joints come from manopth, which already
    applies that same remap inside forward(), so they arrive MediaPipe-ordered
    and must NOT be remapped again.
    """
    assert NATIVE_ORDER["ho3d"] == "mano"
    assert NATIVE_ORDER["dexycb"] == "mediapipe"


def test_load_dexycb_uses_one_camera_by_default(tmp_path):
    """Eight synchronised views of one grasp must not become eight samples."""
    root = str(tmp_path / "dexycb")
    hand = synthetic_hand(20)  # DexYCB ships MediaPipe order already
    cams = ["8402120609%02d" % i for i in range(8)]
    _write_dexycb(root, "20200709-subject-01", "20200709_141754", cams,
                  [(i, hand[i]) for i in range(20)])

    seqs, groups = load_dexycb(root)
    assert len(seqs) == 1, "default must take a single camera per take"
    assert groups == ["20200709-subject-01/20200709_141754"]
    check_joint_order(seqs, "dexycb-fixture")  # default order must be correct

    many, _ = load_dexycb(root, cameras_per_seq=8)
    assert len(many) == 8, "opting in should give every view"


def test_load_dexycb_default_order_does_not_double_remap(tmp_path):
    """Regression: an earlier version defaulted DexYCB to 'mano', which would
    have remapped already-correct joints into a scrambled layout."""
    root = str(tmp_path / "dexycb")
    hand = synthetic_hand(20)
    _write_dexycb(root, "20200709-subject-01", "20200709_141754",
                  ["840412060917"], [(i, hand[i]) for i in range(20)])

    seqs, _ = load_dexycb(root)
    assert np.allclose(seqs[0], hand, atol=1e-5), "default must pass joints through"

    wrong, _ = load_dexycb(root, joint_order="mano")
    with pytest.raises(ValueError, match="joint order looks WRONG"):
        check_joint_order(wrong, "dexycb-double-remapped")


def test_load_dexycb_drops_minus_one_sentinel(tmp_path):
    root = str(tmp_path / "dexycb")
    hand = synthetic_hand(20)
    frames = [(i, hand[i]) for i in range(20)]
    frames[9] = (9, np.full((21, 3), -1.0, dtype=np.float32))
    _write_dexycb(root, "20200709-subject-01", "20200709_141754", ["840412060917"], frames)

    seqs, _ = load_dexycb(root, min_len=4)
    assert [len(s) for s in seqs] == [9, 10]
    for s in seqs:
        assert not np.all(s <= -1.0 + 1e-6, axis=(1, 2)).any()


def test_load_dexycb_missing_root_is_explicit(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="subject-NN"):
        load_dexycb(str(empty))


# --------------------------------------------------------------------------
# the contract rotation_share.py actually consumes
# --------------------------------------------------------------------------
def test_output_satisfies_rotation_share_contract(tmp_path):
    """End to end: what the loader emits must run through the real diagnostic."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from rotation_share import rotation_share_report  # noqa: E402

    root = str(tmp_path / "HO3D_v3")
    inverse = np.argsort(MANO_TO_MEDIAPIPE)
    for name, seed in (("ABF10", 0), ("MDF10", 1)):
        hand = synthetic_hand(40, seed=seed)[:, inverse, :]
        _write_ho3d(root, name, [(i, hand[i]) for i in range(40)])

    seqs, groups = load_ho3d(root, joint_order="mano")
    report = rotation_share_report(seqs, horizons=[2, 8], fps=30, name="fixture",
                                   groups=groups)
    assert set(report["horizons"]) == {"2", "8"}
    for entry in report["horizons"].values():
        assert entry["n_pairs"] > 0
        assert 0.0 <= entry["median_rigid_share"] <= 1.0
        assert set(entry["by_group"]) == {"ABF10", "MDF10"}

    # This motion is a pure rigid rotation plus small noise, so the diagnostic
    # must report a share near 1 -- if it does not, the wiring is wrong.
    assert report["horizons"]["8"]["median_rigid_share"] > 0.9


def test_ragged_sequences_survive_an_npy_roundtrip(tmp_path):
    """Regression: the CLI path, not the in-process one.

    Gap-splitting makes sequences ragged, and np.save stores those as an OBJECT
    array. np.load hands that object array straight to rotation_share.py, which
    used to reject it -- so the loader and the diagnostic each worked alone and
    the documented two-command pipeline failed on the second command.
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from rotation_share import rotation_share_report  # noqa: E402

    seqs = [synthetic_hand(n, seed=i) for i, n in enumerate((30, 17, 22))]
    arr = np.empty(len(seqs), dtype=object)
    for i, s in enumerate(seqs):
        arr[i] = s
    path = str(tmp_path / "poses.npy")
    np.save(path, arr, allow_pickle=True)

    loaded = np.load(path, allow_pickle=True)
    assert loaded.dtype == object, "fixture must actually be ragged"

    report = rotation_share_report(loaded, horizons=[2, 8], fps=30, name="ragged")
    assert report["n_sequences"] == 3
    assert report["horizons"]["8"]["n_pairs"] > 0


def test_malformed_sequence_names_the_offender():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    from rotation_share import rotation_share_report  # noqa: E402

    bad = [synthetic_hand(20), np.zeros((10, 17, 3), dtype=np.float32)]
    with pytest.raises(ValueError, match=r"sequence 1 must be \(T,21,3\)"):
        rotation_share_report(bad, horizons=[2], fps=30)
