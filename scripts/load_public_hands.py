"""Convert public hand-motion datasets into the (N,T,21,3) contract that
scripts/rotation_share.py consumes.

WHY THIS EXISTS. rotation_share.py is deliberately dataset-agnostic: it takes a
plain array and knows nothing about anyone's directory layout. Something still
has to turn HO-3D pickles / DexYCB npz / ARCTIC npy into that array, and that
conversion is where the silent-wrong-answer risk lives. This file is that
conversion, with the traps made loud.

FOUR TRAPS THIS FILE EXISTS TO STOP. Traps 3 and 4 were both found by looking
at real HO-3D output that appeared entirely reasonable.

1. JOINT ORDER. OpenTouch uses the MediaPipe layout -- 0 wrist, then thumb,
   index, middle, ring, pinky, four joints each, so index MCP is 5, middle MCP
   is 9, pinky MCP is 17.

   The two datasets do NOT agree with each other, which is why the native
   layout is per-dataset (NATIVE_ORDER) rather than one global default:

     HO-3D    -> MANO order. Its own vis_HO3D.py applies
                 jointsMapManoToSimple = [0,13,14,15,16,1,2,3,17,4,5,6,18,
                 10,11,12,19,7,8,9,20] to train-split hand joints, which is
                 byte-identical to MANO_TO_MEDIAPIPE below.
     DexYCB   -> ALREADY MediaPipe order, despite being MANO-derived.
                 manopth's ManoLayer.forward() applies that same reorder
                 internally ("Reorder joints to match visualization
                 utilities") and dex-ycb-toolkit's MANOLayer.forward() returns
                 manopth's output unchanged apart from a /1000 unit scale.
                 Remapping it again would scramble a correct layout.

   For the rotation share itself this would not matter: rigid_fraction fits
   Kabsch to the whole point cloud, so it is invariant to any permutation of
   the non-wrist joints applied consistently to both frames. BUT
   hand_frame_degenerate() indexes joints 5/9/17 by position to test whether
   the palm anchors are collinear. Under a wrong ordering that test silently
   examines the wrong three points, and the set of samples excluded as
   "degenerate" becomes arbitrary. The number that comes out still looks
   perfectly reasonable. Hence check_joint_order() below, which refuses to
   proceed rather than let that happen.

2. TEMPORAL GAPS. Every one of these datasets has frames where the hand
   annotation is missing (HO-3D: handJoints3D is None; DexYCB: joint_3d is
   filled with -1). Concatenating across a gap fabricates a delta between two
   frames that were never adjacent, and fabricated deltas are large, which
   biases the rigid share. Sequences are split at every gap instead.

3. PSEUDO-REPLICATION. DexYCB records each sequence from 8 synchronised
   cameras. rigid_fraction is invariant to a global rigid transform, so all 8
   views of one grasp yield *identical* rigid shares -- including them all
   would multiply the apparent sample count by 8 while adding zero
   information, and would make any confidence interval eight times too tight.
   One camera per sequence, by default the first.

   HO-3D has the same problem in a form that a path-based rule MISSES: the
   camera is the last character of the sequence NAME, so ABF10..ABF14 are five
   views of one take. And the naming is not a reliable rule either -- MC1..MC6
   have the same shape but are six genuinely different takes. So dedupe_views()
   works on CONTENT, via a rigid-invariant fingerprint, not on filenames.

4. ANNOTATIONS WITH NO ARTICULATION IN THEM. Six HO-3D sequence families
   (MC, ND, SM, SMu1, SS, SiS -- 15 of 55 recordings) have EXACTLY zero
   variation in their intra-hand joint distances across the whole sequence:
   the annotation is one frozen hand template being re-posed rigidly.

   These score exactly 1.0 on the rotation-share diagnostic, because 100% of
   their wrist-relative motion genuinely is rigid rotation. That is a fact
   about the annotation pipeline, not about hand motion, and averaging it in
   inflates the headline. drop_rigid_templates() removes them and says so.
   This one is nasty because the resulting number looks entirely plausible.

USAGE
    python scripts/load_public_hands.py ho3d   --root /scratch/.../HO3D_v3 --out-prefix ho3d
    python scripts/load_public_hands.py dexycb --root /scratch/.../dexycb  --out-prefix dexycb
    python scripts/load_public_hands.py arctic --root /scratch/.../arctic  --out-prefix arctic

then

    python scripts/rotation_share.py --npy ho3d_poses.npy \
        --groups-npy ho3d_groups.npy --fps 30 --name HO3D --out results/results_rotation_share_ho3d.json
"""

from __future__ import annotations

import argparse

import json
import os
import pickle

import numpy as np

# MANO's 21-joint layout -> the MediaPipe layout OpenTouch uses.
# Reading: output slot 1 (thumb CMC) comes from MANO slot 13, and so on.
# This is the same remap HO-3D's own evaluation code applies.
MANO_TO_MEDIAPIPE = np.array(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)

# The layout each dataset actually ships in. See trap 1 in the module
# docstring for the source-level evidence behind each entry; they differ, so
# there is deliberately no single global default.
NATIVE_ORDER = {
    "ho3d": "mano",
    "dexycb": "mediapipe",
    "arctic": "mano",
}

WRIST = 0
INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 5, 9, 17
# In MediaPipe order each finger is four consecutive joints running outward
# from the palm: (MCP, PIP, DIP, TIP).
FINGER_CHAINS = [tuple(range(1 + 4 * f, 5 + 4 * f)) for f in range(5)]


def pairwise_distance_profile(seq: np.ndarray) -> np.ndarray:
    """(T,21,3) -> (T,210) of intra-hand joint-to-joint distances.

    These are invariant to ANY rigid motion of the hand, which makes them the
    right tool for both guards below: two camera views of one take give the
    SAME profile, and a hand whose shape never changes gives a CONSTANT one.
    """
    diff = seq[:, :, None, :] - seq[:, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(seq.shape[1], k=1)
    return dist[:, iu[0], iu[1]]


def is_rigid_template(seq: np.ndarray, tol: float = 1e-4) -> bool:
    """True when the hand's SHAPE never changes over the sequence -- i.e. the
    annotation is one frozen hand being re-posed rigidly, with no articulation
    in it at all.

    Such sequences are worse than useless for the rotation-share diagnostic:
    they score exactly 1.0 by construction, because 100% of the wrist-relative
    motion really is rigid rotation. Including them does not measure hand
    motion, it measures the annotation pipeline, and it inflates the result.
    HO-3D's MC / ND / SM / SMu1 / SS / SiS sequences are exactly this.
    """
    profile = pairwise_distance_profile(seq)
    scale = float(profile.mean())
    if scale <= 0:
        return True
    return float((profile.std(axis=0) / scale).max()) < tol


_VIEW_PROBE_FRAMES = 64


def _view_profile(seq: np.ndarray) -> np.ndarray:
    """Pairwise-distance profile on at most _VIEW_PROBE_FRAMES evenly spaced
    frames -- enough to separate genuinely different takes, and bounded in
    memory for HO-3D's 2000-frame sequences."""
    idx = np.linspace(0, seq.shape[0] - 1, min(seq.shape[0], _VIEW_PROBE_FRAMES)).astype(int)
    return pairwise_distance_profile(seq[idx]).astype(np.float32)


def view_signature(seq: np.ndarray, decimals: int = 2) -> str:
    """COARSE bucket key, deliberately not an equality test.

    Keyed on the pairwise-distance profile rather than on joint coordinates,
    because each camera reports the same hand in its own frame: the
    coordinates differ, the profile does not. Content-based and not
    filename-based because HO-3D encodes the camera inconsistently --
    ABF10..ABF14 are five views of one take, but MC1..MC6 are six different
    takes.

    Only coarse. Two views of one take agree to ~4e-7 in float32, which is far
    too tight to hash exactly: with ~6,000 values per sequence, some value
    always straddles a rounding boundary no matter which decimal you pick.
    So this buckets candidates and dedupe_views() confirms with an explicit
    tolerant comparison.
    """
    p = _view_profile(seq)
    return "%d|%.*f|%.*f" % (seq.shape[0], decimals, float(p.mean()), decimals, float(p.max()))


def drop_rigid_templates(seqs, groups):
    keep = [i for i, s in enumerate(seqs) if not is_rigid_template(s)]
    dropped = len(seqs) - len(keep)
    if dropped:
        names = sorted({groups[i] for i in range(len(seqs)) if i not in set(keep)})
        print("  dropped %d sequence(s) with NO articulation (rigid template, "
              "would score 1.0 by construction): %s"
              % (dropped, ", ".join(names[:12]) + (" ..." if len(names) > 12 else "")))
    return [seqs[i] for i in keep], [groups[i] for i in keep]


def dedupe_views(seqs, groups):
    """Keep one sequence per distinct rigid-invariant signature.

    Multiple synchronised views of one take have identical rigid shares by
    construction, so keeping them all multiplies apparent n while adding zero
    information and makes any interval far too tight.
    """
    buckets: dict = {}
    keep, members = [], []
    for i, s in enumerate(seqs):
        key = view_signature(s)
        prof = _view_profile(s)
        hit = None
        for slot, ref in buckets.get(key, []):
            if ref.shape == prof.shape and np.allclose(ref, prof, rtol=1e-4, atol=1e-6):
                hit = slot
                break
        if hit is not None:
            members[hit].append(groups[i])
            continue
        buckets.setdefault(key, []).append((len(keep), prof))
        keep.append(i)
        members.append([groups[i]])

    collapsed = len(seqs) - len(keep)
    if collapsed:
        sizes = sorted((len(m) for m in members if len(m) > 1), reverse=True)
        print("  collapsed %d duplicate view(s): %d take(s) had multiple "
              "synchronised cameras (group sizes %s)"
              % (collapsed, len(sizes), sizes[:10]))
    return [seqs[i] for i in keep], [groups[i] for i in keep]


def reorder(seq: np.ndarray, order: str) -> np.ndarray:
    """(T,21,3) in `order` -> (T,21,3) in MediaPipe order."""
    if order == "mediapipe":
        return seq
    if order == "mano":
        return seq[:, MANO_TO_MEDIAPIPE, :]
    raise ValueError("order must be 'mano' or 'mediapipe', got %r" % order)


def check_joint_order(seqs, name, strict=True):
    """Fail loudly if the joint layout is not the one we think it is.

    Two independent anatomical predictions that hold for a real hand in
    MediaPipe order, and break under a wrong permutation:

      A. Along each finger, distance from the wrist increases monotonically:
         MCP < PIP < DIP < TIP. A permuted layout scrambles the chains and
         this collapses.
      B. The palm anchors (wrist, index MCP, middle MCP, pinky MCP) are not
         collinear -- exactly the assumption hand_frame_degenerate() makes.

    Returns the diagnostics dict either way; raises when strict and A fails.
    """
    pooled = np.concatenate([s for s in seqs], axis=0)
    finite = np.isfinite(pooled).all(axis=(1, 2))
    pooled = pooled[finite]
    if pooled.shape[0] == 0:
        raise ValueError("%s: no finite frames to validate joint order on" % name)

    centered = pooled - pooled[:, WRIST : WRIST + 1, :]
    radius = np.linalg.norm(centered, axis=-1)  # (B,21)

    per_finger = []
    for chain in FINGER_CHAINS:
        r = radius[:, list(chain)]
        # fraction of frames where this finger runs strictly outward
        per_finger.append(float((np.diff(r, axis=1) > 0).all(axis=1).mean()))
    monotone = float(np.mean(per_finger))

    # Collinearity of the palm triangle, normalised to [0,1]; 0 == collinear.
    long_axis = centered[:, MIDDLE_MCP, :]
    transverse = centered[:, INDEX_MCP, :] - centered[:, PINKY_MCP, :]
    cross = np.cross(long_axis, transverse)
    denom = np.linalg.norm(long_axis, axis=-1) * np.linalg.norm(transverse, axis=-1)
    palm_area = np.divide(
        np.linalg.norm(cross, axis=-1), denom,
        out=np.zeros(denom.shape), where=denom > 0,
    )

    diag = {
        "n_frames_checked": int(pooled.shape[0]),
        "monotone_chain_fraction": monotone,
        "per_finger_monotone": per_finger,
        "median_palm_noncollinearity": float(np.median(palm_area)),
        "frac_palm_degenerate": float((palm_area < 1e-3).mean()),
    }

    print("\n[joint-order check] %s" % name)
    print("  fingers running outward from wrist : %.1f%%  (per finger: %s)"
          % (100 * monotone, ", ".join("%.0f%%" % (100 * p) for p in per_finger)))
    print("  palm non-collinearity (median)     : %.3f   degenerate: %.2f%%"
          % (diag["median_palm_noncollinearity"], 100 * diag["frac_palm_degenerate"]))

    if monotone < 0.80:
        msg = (
            "%s: joint order looks WRONG. Only %.1f%% of frames have all five "
            "fingers running outward from the wrist; a correct layout gives "
            ">95%%. Per-finger: %s. Re-check --joint-order (mano vs mediapipe) "
            "before trusting any number computed from this array."
            % (name, 100 * monotone, per_finger)
        )
        if strict:
            raise ValueError(msg)
        print("  WARNING: " + msg)
    else:
        print("  -> layout consistent with MediaPipe order")
    return diag


def split_on_gaps(frames, min_len):
    """[(frame_index, (21,3)), ...] -> list of (T,21,3), split where the frame
    index is not consecutive. Sorted by frame index first."""
    frames = sorted(frames, key=lambda x: x[0])
    out, run = [], []
    prev = None
    for idx, joints in frames:
        if prev is not None and idx != prev + 1:
            if len(run) >= min_len:
                out.append(np.stack(run))
            run = []
        run.append(joints)
        prev = idx
    if len(run) >= min_len:
        out.append(np.stack(run))
    return out


# --------------------------------------------------------------------------
# HO-3D
# --------------------------------------------------------------------------
def load_ho3d(root, split="train", min_len=8, joint_order="mano", limit_seqs=None):
    """<root>/<split>/<seq>/meta/<frame>.pkl, each holding 'handJoints3D'.

    Only the train split carries all 21 joints; the evaluation split
    historically ships the wrist alone, which cannot be decomposed.
    """
    base = os.path.join(root, split)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            "%s not found. Expected the extracted HO-3D layout "
            "<root>/train/<seq>/meta/*.pkl" % base
        )
    seq_names = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
    if limit_seqs:
        seq_names = seq_names[:limit_seqs]

    seqs, groups, n_missing, n_total, n_unnamed = [], [], 0, 0, 0
    for sname in seq_names:
        meta_dir = os.path.join(base, sname, "meta")
        if not os.path.isdir(meta_dir):
            continue
        frames = []
        for fn in sorted(os.listdir(meta_dir)):
            if not fn.endswith(".pkl"):
                continue
            # Frame index comes from the filename, which is how temporal order
            # and gap detection work. Anything not named <int>.pkl is skipped
            # rather than allowed to crash -- this runs after a 31.9 GB
            # download, so it must not die on one stray file.
            stem = os.path.splitext(fn)[0]
            try:
                frame_idx = int(stem)
            except ValueError:
                n_unnamed += 1
                continue
            n_total += 1
            with open(os.path.join(meta_dir, fn), "rb") as fh:
                meta = pickle.load(fh, encoding="latin1")
            j = meta.get("handJoints3D")
            if j is None:
                n_missing += 1
                continue
            j = np.asarray(j, dtype=np.float32)
            if j.shape != (21, 3) or not np.isfinite(j).all():
                n_missing += 1
                continue
            frames.append((frame_idx, j))
        for run in split_on_gaps(frames, min_len):
            seqs.append(reorder(run, joint_order))
            groups.append(sname)

    print("HO-3D: %d sequences kept from %d recordings (%d/%d frames lacked joints)"
          % (len(seqs), len(seq_names), n_missing, n_total))
    if n_unnamed:
        print("  note: skipped %d .pkl files not named <frame-index>.pkl" % n_unnamed)
    return seqs, groups


# --------------------------------------------------------------------------
# DexYCB
# --------------------------------------------------------------------------
def load_dexycb(root, min_len=8, joint_order="mediapipe", cameras_per_seq=1, limit_seqs=None):
    """<root>/<subject>/<YYYYMMDD_HHMMSS>/<camera>/labels_*.npz, 'joint_3d'
    shaped (1,21,3) in millimetres, -1 where unavailable.

    Note the default order differs from HO-3D's: DexYCB's joints come out of
    manopth, which already reorders to the MediaPipe layout internally.

    Only `cameras_per_seq` cameras are used -- see trap 3 in the module
    docstring. Units are millimetres but rigid_fraction is a ratio of energies,
    so no rescaling is needed.
    """
    subjects = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and "subject" in d
    )
    if not subjects:
        raise FileNotFoundError(
            "no <date>-subject-NN directories under %s -- point --root at the "
            "extracted DexYCB tree" % root
        )

    seqs, groups, n_missing, n_total = [], [], 0, 0
    n_seq_dirs = 0
    for subj in subjects:
        subj_dir = os.path.join(root, subj)
        for take in sorted(os.listdir(subj_dir)):
            take_dir = os.path.join(subj_dir, take)
            if not os.path.isdir(take_dir):
                continue
            cams = sorted(d for d in os.listdir(take_dir)
                          if os.path.isdir(os.path.join(take_dir, d)))
            if not cams:
                continue
            n_seq_dirs += 1
            if limit_seqs and n_seq_dirs > limit_seqs:
                break
            for cam in cams[:cameras_per_seq]:
                cam_dir = os.path.join(take_dir, cam)
                frames = []
                for fn in sorted(os.listdir(cam_dir)):
                    if not (fn.startswith("labels_") and fn.endswith(".npz")):
                        continue
                    n_total += 1
                    with np.load(os.path.join(cam_dir, fn)) as z:
                        if "joint_3d" not in z:
                            n_missing += 1
                            continue
                        j = np.asarray(z["joint_3d"], dtype=np.float32).reshape(-1, 21, 3)[0]
                    # -1 is DexYCB's sentinel for "no hand annotation here".
                    if not np.isfinite(j).all() or np.all(j <= -1.0 + 1e-6):
                        n_missing += 1
                        continue
                    idx = int(os.path.splitext(fn)[0].split("_")[-1])
                    frames.append((idx, j))
                for run in split_on_gaps(frames, min_len):
                    seqs.append(reorder(run, joint_order))
                    groups.append("%s/%s" % (subj, take))

    print("DexYCB: %d sequences from %d takes across %d subjects "
          "(%d cam/take, %d/%d frames unannotated)"
          % (len(seqs), n_seq_dirs, len(subjects), cameras_per_seq, n_missing, n_total))
    return seqs, groups


# --------------------------------------------------------------------------
# ARCTIC
# --------------------------------------------------------------------------
def load_arctic(root, min_len=8, joint_order="mano", side="right", limit_seqs=None):
    """ARCTIC raw sequences: <root>/raw_seqs/<subject>/<action>.mano.npy.

    CAVEAT, READ BEFORE USING. ARCTIC's raw annotation drop stores MANO
    *parameters* (pose/shape/trans), not joint positions. Recovering (21,3)
    needs a MANO forward pass, which needs the MANO model download and its
    own licence. If the file turns out to carry a joints array directly this
    loader uses it; otherwise it says exactly what it found and stops, rather
    than guessing.
    """
    raw = os.path.join(root, "raw_seqs")
    if not os.path.isdir(raw):
        raise FileNotFoundError(
            "%s not found. Expected ARCTIC's raw_seqs/ layout" % raw
        )
    seqs, groups = [], []
    n_seen = 0
    joint_keys = ("joints3d", "joints", "j3d", "joints_3d")
    for subj in sorted(os.listdir(raw)):
        sdir = os.path.join(raw, subj)
        if not os.path.isdir(sdir):
            continue
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith(".mano.npy"):
                continue
            n_seen += 1
            if limit_seqs and n_seen > limit_seqs:
                break
            blob = np.load(os.path.join(sdir, fn), allow_pickle=True).item()
            entry = blob.get(side, blob)
            found = next((k for k in joint_keys if k in entry), None)
            if found is None:
                raise NotImplementedError(
                    "ARCTIC %s/%s carries %s -- MANO parameters, not joint "
                    "positions. A MANO forward pass is required; this loader "
                    "does not do one. Use HO-3D or DexYCB first."
                    % (subj, fn, sorted(entry.keys()))
                )
            j = np.asarray(entry[found], dtype=np.float32)
            if j.ndim != 3 or j.shape[1] < 21:
                raise ValueError("unexpected joints shape %s in %s/%s"
                                 % (j.shape, subj, fn))
            j = j[:, :21, :]
            ok = np.isfinite(j).all(axis=(1, 2))
            frames = [(i, j[i]) for i in np.nonzero(ok)[0]]
            for run in split_on_gaps(frames, min_len):
                seqs.append(reorder(run, joint_order))
                groups.append("%s/%s" % (subj, fn.split(".")[0]))

    print("ARCTIC: %d sequences from %d files" % (len(seqs), n_seen))
    return seqs, groups


LOADERS = {"ho3d": load_ho3d, "dexycb": load_dexycb, "arctic": load_arctic}


def main():
    ap = argparse.ArgumentParser(
        description="Convert a public hand dataset into rotation_share.py's (N,T,21,3) contract")
    ap.add_argument("dataset", choices=sorted(LOADERS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-prefix", required=True,
                    help="writes <prefix>_poses.npy and <prefix>_groups.npy")
    ap.add_argument("--joint-order", default=None, choices=["mano", "mediapipe"],
                    help="override the layout this dataset ships in; by default "
                         "each dataset uses its own (HO-3D mano, DexYCB mediapipe)")
    ap.add_argument("--min-len", type=int, default=8,
                    help="drop contiguous runs shorter than this")
    ap.add_argument("--limit-seqs", type=int, default=None,
                    help="stop after N sequences -- use for a fast smoke test")
    ap.add_argument("--cameras-per-seq", type=int, default=1,
                    help="DexYCB only; >1 duplicates identical motion")
    ap.add_argument("--split", default="train", help="HO-3D only")
    ap.add_argument("--side", default="right", help="ARCTIC only")
    ap.add_argument("--no-strict-order-check", action="store_true",
                    help="warn instead of failing when the layout check trips")
    ap.add_argument("--keep-rigid-templates", action="store_true",
                    help="keep sequences whose hand shape never changes; they "
                         "score 1.0 by construction and inflate the result")
    ap.add_argument("--keep-duplicate-views", action="store_true",
                    help="keep synchronised camera views of the same take; "
                         "they are numerically identical and inflate n")
    args = ap.parse_args()

    order = args.joint_order or NATIVE_ORDER[args.dataset]
    print("%s ships %s-ordered joints%s"
          % (args.dataset, order, " (overridden)" if args.joint_order else ""))
    kw = dict(min_len=args.min_len, joint_order=order, limit_seqs=args.limit_seqs)
    if args.dataset == "ho3d":
        kw["split"] = args.split
    if args.dataset == "dexycb":
        kw["cameras_per_seq"] = args.cameras_per_seq
    if args.dataset == "arctic":
        kw["side"] = args.side

    seqs, groups = LOADERS[args.dataset](args.root, **kw)
    if not seqs:
        raise SystemExit("no usable sequences found under %s" % args.root)

    n_raw = len(seqs)
    if not args.keep_rigid_templates:
        seqs, groups = drop_rigid_templates(seqs, groups)
    if not args.keep_duplicate_views:
        seqs, groups = dedupe_views(seqs, groups)
    if not seqs:
        raise SystemExit(
            "every sequence was filtered out: %d were rigid templates or "
            "duplicate views. Pass --keep-rigid-templates / "
            "--keep-duplicate-views to inspect the raw data." % n_raw
        )
    if len(seqs) != n_raw:
        print("  %d -> %d sequences after filtering" % (n_raw, len(seqs)))

    diag = check_joint_order(seqs, args.dataset, strict=not args.no_strict_order_check)

    lengths = np.array([len(s) for s in seqs])
    print("\n%s: %d sequences, %d frames total, length min/median/max %d/%d/%d"
          % (args.dataset, len(seqs), int(lengths.sum()),
             lengths.min(), int(np.median(lengths)), lengths.max()))
    print("groups: %d distinct" % len(set(groups)))

    arr = np.empty(len(seqs), dtype=object)
    for i, s in enumerate(seqs):
        arr[i] = s.astype(np.float32)
    np.save(args.out_prefix + "_poses.npy", arr, allow_pickle=True)
    np.save(args.out_prefix + "_groups.npy", np.asarray(groups, dtype=object),
            allow_pickle=True)
    with open(args.out_prefix + "_loadinfo.json", "w") as fh:
        json.dump({"dataset": args.dataset, "root": args.root,
                   "joint_order": args.joint_order, "n_sequences": len(seqs),
                   "n_frames": int(lengths.sum()), "order_check": diag}, fh, indent=2)
    print("\nWrote %s_poses.npy, %s_groups.npy, %s_loadinfo.json"
          % (args.out_prefix, args.out_prefix, args.out_prefix))
    print("\nNext:\n  python scripts/rotation_share.py --npy %s_poses.npy "
          "--groups-npy %s_groups.npy --fps 30 --name %s "
          "--out results/results_rotation_share_%s.json"
          % (args.out_prefix, args.out_prefix, args.dataset.upper(), args.dataset))


if __name__ == "__main__":
    main()
