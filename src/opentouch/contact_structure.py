"""Taxel-to-joint correspondence matrix B for the OpenTouch tactile encoder.

Builds an anatomical inductive bias so a 21-MANO-keypoint attention query can
carry an additive pre-softmax bias over the 169 active taxels, derived from
which joints' motion actually deforms the mesh patch each taxel sits on
(MANO skinning weights), not from a hand-labeled or guessed layout.

Data provenance (see scripts/diag_layout.py for the full diagnostic trail):
  - assets/handLayoutNewest_meshid.json ("highres") is the only layout file
    with a traceable generation path (preprocess/scratch/mano_densifincation.py:
    subdivide() -> mesh_point_mapping_new() -> rewrite_mapjson()), and its
    mano_vid ids index assets/mano_right_neutral_subdiv.obj (13,614 verts).
  - assets/handLayoutNewest_meshid_lowres.json ("lowres") has 22/169 taxels
    with mano_vid ids out of range for the 778-vertex base mesh and no
    matching generation script; it is used here only as an independent
    cross-check (see tests / diag script), never as a primary data source.
  - The subdiv mesh is a pure edge-split subdivision of the base mesh in the
    same coordinate frame (point-to-face distance from subdiv vertices to
    the base mesh surface is ~1e-9, i.e. float precision noise -- see
    scripts/diag_layout.py section 5). A subdiv vertex id is therefore
    bridged to a base-mesh / skinning-weight row id via nearest-base-vertex.
"""

from __future__ import annotations

import argparse
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"

OBJ_BASE = ASSETS / "mano_right_neutral.obj"
OBJ_SUBDIV = ASSETS / "mano_right_neutral_subdiv.obj"
LAYOUT_HIGHRES = ASSETS / "handLayoutNewest_meshid.json"
LAYOUT_LOWRES = ASSETS / "handLayoutNewest_meshid_lowres.json"
MANO_PKL = ASSETS / "MANO_RIGHT.pkl"
B_MATRICES_PATH = ASSETS / "B_matrices.npz"

NUM_TAXELS = 169
GRID_SIZE = 256
NUM_KEYPOINTS = 21
NUM_SKINNING_JOINTS = 16
NUM_REGIONS = 6  # palm/wrist + 5 fingers
REGION_NAMES = ("palm", "thumb", "index", "middle", "ring", "pinky")


# ---------------------------------------------------------------------------
# Low-level IO
# ---------------------------------------------------------------------------

def _parse_obj_vertices(path: Path) -> np.ndarray:
    """Parse `v x y z` lines from an OBJ file, in file order. Returns (N,3)."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
    return np.asarray(verts, dtype=np.float64)


def _load_layout(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def load_mano_pkl(path: Path = MANO_PKL) -> dict:
    """Load MANO_RIGHT.pkl (v_template, weights, J, kintree_table, ...).

    These files are pickled chumpy/numpy objects created under a Python
    2 / old-numpy stack. Two process-local monkeypatches are required to
    unpickle them under a modern (numpy>=2, Python>=3.11) interpreter:
      - chumpy 0.70 calls inspect.getargspec, removed in Python 3.11+.
      - chumpy 0.70 imports np.bool/np.int/np.float/... aliases, removed
        in numpy 2.x.
    Neither shim touches site-packages; both only fill in names that are
    otherwise absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Skinning-based B (build_skinning_B) requires "
            "MANO_RIGHT.pkl for v_template, weights, and kintree_table. "
            "Use build_geometric_fallback_B() / --geometric on the CLI "
            "instead -- that path needs no pkl, but it is SCIENTIFICALLY "
            "WEAKER: with no skinning weights or kinematic tree it cannot "
            "tell 'physically driven by joint J' apart from 'merely close "
            "to joint J in 3D', so it only resolves 6 finger-level regions "
            "broadcast uniformly across each finger's 4 keypoint columns, "
            "not the full 16 kinematic joints."
        )
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
    for name, val in [
        ("bool", bool), ("int", int), ("float", float), ("complex", complex),
        ("object", object), ("unicode", str), ("str", str),
    ]:
        if not hasattr(np, name):
            setattr(np, name, val)
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


# ---------------------------------------------------------------------------
# Taxel ordering: single source of truth
# ---------------------------------------------------------------------------

def build_taxel_index(layout_path: Path = LAYOUT_HIGHRES) -> list[str]:
    """Canonical taxel ordering: valid (non-erased) grid keys sorted by
    (int(row), int(col)). Every downstream consumer of taxel order must
    import this function rather than re-deriving or re-sorting the key
    list; row index in every returned matrix corresponds to this order.

    lowres and highres layouts have been verified (scripts/diag_layout.py,
    and re-asserted here) to share an identical 256-key grid and 87-key
    erased set, so this ordering does not depend on which layout file
    supplies it.
    """
    layout = _load_layout(layout_path)
    erased = set(layout["erasedNodes"])  # erasedNodes has duplicate entries
    valid = [k for k in layout["positions"].keys() if k not in erased]
    assert len(valid) == NUM_TAXELS, (
        f"expected {NUM_TAXELS} valid taxels (256 grid - 87 unique erased), "
        f"got {len(valid)}"
    )
    ordered = sorted(valid, key=lambda k: tuple(int(x) for x in k.split("-")))
    return ordered


# ---------------------------------------------------------------------------
# Geometric bridge: subdiv vertex id -> base-mesh / skinning-weight row id
# ---------------------------------------------------------------------------

def _build_subdiv_to_base_bridge(verts_base: np.ndarray, verts_subdiv: np.ndarray) -> np.ndarray:
    """Nearest-base-vertex id for every subdiv vertex.

    Validated in scripts/diag_layout.py (section 5): the point-to-face
    distance from every subdiv vertex to the nearest base-mesh triangle is
    ~1e-9 (float precision noise from the OBJ export's 6-vs-8 decimal
    truncation), confirming mano_right_neutral_subdiv.obj is a pure
    edge-split subdivision of mano_right_neutral.obj in the same coordinate
    frame (no smoothing/displacement, no rescale). Nearest-vertex lookup is
    therefore a safe (if slightly lossy near large original triangles) way
    to carry a subdiv vertex id back to a skinning-weight row id.
    """
    tree = cKDTree(verts_base)
    _, base_ids = tree.query(verts_subdiv, k=1)
    return base_ids.astype(np.int64)


# ---------------------------------------------------------------------------
# MANO kinematic tree -> finger chains -> 21-keypoint column mapping
# ---------------------------------------------------------------------------

def _derive_finger_chains(kintree_table: np.ndarray) -> tuple[int, list[list[int]]]:
    """From MANO's kinematic tree alone, derive the root joint id and 5
    length-3 chains (one per finger) in kinematic near-to-far order
    (MCP -> PIP -> DIP). Purely topological: does not assume which chain
    is which finger, or what MANO's internal joint numbering means.
    """
    kintree = kintree_table.astype(np.int64).copy()
    if kintree[0, 0] > 1_000_000:
        # MANO stores the "no parent" sentinel as a wrapped uint32 -1.
        kintree[0, 0] = -1
    parents, children = kintree[0], kintree[1]

    root_candidates = np.where(parents == -1)[0]
    assert len(root_candidates) == 1, (
        f"expected exactly one root joint (parent == -1), got {root_candidates}"
    )
    root = int(root_candidates[0])

    finger_roots = [int(c) for c in children if parents[c] == root]
    assert len(finger_roots) == 5, (
        f"expected 5 finger root joints (direct children of root), got "
        f"{len(finger_roots)}: {finger_roots}"
    )

    chains = []
    for fr in finger_roots:
        chain, cur = [fr], fr
        while True:
            kids = [int(c) for c in children if parents[c] == cur]
            if not kids:
                break
            assert len(kids) == 1, (
                f"expected a single kinematic child per joint, got {kids} for joint {cur}"
            )
            cur = kids[0]
            chain.append(cur)
        assert len(chain) == 3, (
            f"expected a 3-joint finger chain (MCP,PIP,DIP), got {chain}"
        )
        chains.append(chain)
    return root, chains


def _order_finger_chains(J: np.ndarray, chains: list[list[int]]) -> list[list[int]]:
    """Reorder the 5 topological chains into [thumb, index, middle, ring,
    pinky], derived purely from the neutral-pose 3D joint positions J.

    VERIFY (geometric derivation, not a memorized MANO joint-id table):
      - Thumb identification: leave-one-out PCA line fit over the 5 MCP
        positions. In a neutral/flat hand pose, index/middle/ring/pinky
        MCPs lie roughly along the palm's knuckle line; the thumb MCP is
        rotated out of that line (opposable thumb). The chain whose
        exclusion gives the tightest line fit over the remaining 4 is the
        thumb.
      - index/middle/ring/pinky ordering: uses the anatomical invariant
        that the index finger is always adjacent to the thumb (basic hand
        anatomy, not MANO-specific numbering) to fix the direction of the
        fitted knuckle line; the remaining 3 fingers follow monotonically
        along that line.
    A runtime assert cross-checks the result against this repo's own
    21-keypoint convention (src/opentouch/pose_encoder.py's
    _THUMB_MCP/_INDEX_MCP/_MIDDLE_MCP), so a wrong derivation fails loudly
    instead of silently mislabeling fingers.
    """
    mcp_pos = np.array([J[chain[0]] for chain in chains])  # (5,3)

    def _line_fit_residual(points: np.ndarray) -> float:
        centroid = points.mean(axis=0)
        centered = points - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
        proj = centered @ direction
        residual = centered - np.outer(proj, direction)
        return float(np.sum(residual ** 2))

    residuals = [
        _line_fit_residual(np.delete(mcp_pos, i, axis=0)) for i in range(5)
    ]
    thumb_idx = int(np.argmin(residuals))

    other_idx = [i for i in range(5) if i != thumb_idx]
    other_pos = mcp_pos[other_idx]
    centroid = other_pos.mean(axis=0)
    _, _, vt = np.linalg.svd(other_pos - centroid, full_matrices=False)
    direction = vt[0]
    proj = (other_pos - centroid) @ direction
    order = np.argsort(proj)

    thumb_pos = mcp_pos[thumb_idx]
    dists_to_thumb = np.linalg.norm(other_pos - thumb_pos, axis=1)
    closest_to_thumb = int(np.argmin(dists_to_thumb))
    if order[0] != closest_to_thumb:
        order = order[::-1]

    finger_order_idx = [thumb_idx] + [other_idx[i] for i in order]
    return [chains[i] for i in finger_order_idx]


def _skinning_to_kp21_map(root: int, ordered_chains: list[list[int]]) -> dict[int, int]:
    """21-keypoint column layout (see src/opentouch/pose_encoder.py):
    0 = wrist; for finger slot i in [thumb, index, middle, ring, pinky],
    columns 1+4i, 2+4i, 3+4i, 4+4i = MCP, PIP, DIP, TIP. The 16 skinning
    joints (root + 5*3 chain joints) map onto the 16 non-TIP columns; the
    5 TIP columns have no skinning-joint counterpart (see
    _derive_fingertip_positions) and are filled geometrically.
    """
    mapping = {root: 0}
    for slot, chain in enumerate(ordered_chains):
        block = 1 + 4 * slot
        for pos_in_chain, joint in enumerate(chain):
            mapping[joint] = block + pos_in_chain
    return mapping


def _derive_fingertip_positions(
    v_template: np.ndarray, weights: np.ndarray, J: np.ndarray, ordered_chains: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate a 3D fingertip location and length scale per finger from
    mesh geometry + skinning weights alone (no hardcoded vertex ids).

    VERIFY: MANO's skinning basis has no joint at the fingertip (the DIP
    joint is the most distal skinning joint), so "the fingertip" is not
    directly present in the file. This derives it as: among base-mesh
    vertices whose dominant skinning weight (argmax over the 16 joints) is
    a given finger's DIP joint, take the one that extends farthest past the
    DIP joint along that finger's MCP->PIP->DIP direction. The tip's
    distance from the DIP joint is used as that finger's proximity kernel
    length scale (each finger's own last-phalanx length, not a shared
    magic constant).
    """
    dominant = np.argmax(weights, axis=1)  # (778,) per-vertex dominant joint
    tip_positions, tip_sigmas = [], []
    for chain in ordered_chains:
        _, pip, dip = chain
        candidates = np.where(dominant == dip)[0]
        assert candidates.size > 0, (
            f"no base-mesh vertices have dominant skinning joint == DIP joint {dip}; "
            "cannot derive a fingertip location for this finger"
        )
        chain_dir = J[dip] - J[pip]
        chain_dir = chain_dir / np.linalg.norm(chain_dir)
        proj = (v_template[candidates] - J[dip]) @ chain_dir
        tip_vertex = candidates[int(np.argmax(proj))]
        tip_pos = v_template[tip_vertex]
        sigma = max(float(np.linalg.norm(tip_pos - J[dip])), 1e-6)
        tip_positions.append(tip_pos)
        tip_sigmas.append(sigma)
    return np.asarray(tip_positions, dtype=np.float64), np.asarray(tip_sigmas, dtype=np.float64)


# ---------------------------------------------------------------------------
# B matrices
# ---------------------------------------------------------------------------

def build_skinning_B(taxel_order: list[str] | None = None) -> np.ndarray:
    """(169, 21) taxel-to-keypoint soft assignment from MANO skinning
    weights, mapped from the 16 kinematic joints into the 21-keypoint
    layout, with the 5 fingertip columns filled by geometric proximity
    (see _derive_fingertip_positions). Rows sum to 1.
    """
    if taxel_order is None:
        taxel_order = build_taxel_index()
    assert len(taxel_order) == NUM_TAXELS

    mano = load_mano_pkl()
    v_template = np.asarray(mano["v_template"], dtype=np.float64)
    weights = np.asarray(mano["weights"], dtype=np.float64)
    J = np.asarray(mano["J"], dtype=np.float64)
    kintree = np.asarray(mano["kintree_table"])
    assert v_template.shape[1] == 3
    assert weights.shape == (v_template.shape[0], NUM_SKINNING_JOINTS), weights.shape
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-4), (
        "MANO skinning weight rows must sum to 1"
    )

    verts_base = _parse_obj_vertices(OBJ_BASE)
    verts_subdiv = _parse_obj_vertices(OBJ_SUBDIV)
    assert verts_base.shape[0] == v_template.shape[0], (
        f"base OBJ vertex count {verts_base.shape[0]} != MANO v_template "
        f"count {v_template.shape[0]}"
    )
    assert float(np.abs(verts_base - v_template).max()) < 1e-4, (
        "base OBJ does not match MANO v_template within tolerance; geometry "
        "and skinning weights would be misaligned (see scripts/diag_layout.py "
        "section 3)"
    )
    bridge = _build_subdiv_to_base_bridge(verts_base, verts_subdiv)

    layout = _load_layout(LAYOUT_HIGHRES)
    positions = layout["positions"]

    root, chains = _derive_finger_chains(kintree)
    ordered_chains = _order_finger_chains(J, chains)  # [thumb, index, middle, ring, pinky]

    from .pose_encoder import _INDEX_MCP, _MIDDLE_MCP, _THUMB_MCP
    skinning_to_kp21 = _skinning_to_kp21_map(root, ordered_chains)
    assert skinning_to_kp21[ordered_chains[0][0]] == _THUMB_MCP
    assert skinning_to_kp21[ordered_chains[1][0]] == _INDEX_MCP
    assert skinning_to_kp21[ordered_chains[2][0]] == _MIDDLE_MCP

    tip_kp21_cols = [1 + 4 * slot + 3 for slot in range(5)]
    tip_positions, tip_sigmas = _derive_fingertip_positions(v_template, weights, J, ordered_chains)

    skinning_B = np.zeros((NUM_TAXELS, NUM_KEYPOINTS), dtype=np.float64)
    for row, key in enumerate(taxel_order):
        vids = positions[key]["mano_vid"]
        assert len(vids) > 0, (
            f"taxel {key} has an empty mano_vid list in the highres layout "
            "(expected 0 such taxels, see scripts/diag_layout.py section 4)"
        )
        taxel_xyz = verts_subdiv[vids].mean(axis=0)
        base_ids = bridge[vids]
        w = weights[base_ids].mean(axis=0)  # (16,) averaged over the taxel's patch
        for joint, col in skinning_to_kp21.items():
            skinning_B[row, col] = w[joint]

        d2 = np.sum((tip_positions - taxel_xyz) ** 2, axis=1)
        tip_vals = np.exp(-d2 / (2.0 * tip_sigmas ** 2))
        for slot, col in enumerate(tip_kp21_cols):
            skinning_B[row, col] = tip_vals[slot]

    row_sums = skinning_B.sum(axis=1, keepdims=True)
    assert np.all(row_sums > 0), "a taxel ended up with zero total correspondence mass"
    skinning_B = skinning_B / row_sums

    assert skinning_B.shape == (NUM_TAXELS, NUM_KEYPOINTS)
    assert not np.isnan(skinning_B).any()
    assert np.allclose(skinning_B.sum(axis=1), 1.0, atol=1e-8)
    return skinning_B


def _kp21_col_to_region(col: int) -> int:
    if col == 0:
        return 0  # palm/wrist
    return 1 + (col - 1) // 4  # thumb=1, index=2, middle=3, ring=4, pinky=5


def build_region_B(skinning_B: np.ndarray | None = None) -> np.ndarray:
    """(169, 6) coarsening of build_skinning_B: per taxel, take the argmax
    keypoint column of skinning_B and bucket it into palm/thumb/index/
    middle/ring/pinky. This is a strict coarsening (not an independent
    hand-labeling), so a region-vs-skinning ablation isolates resolution
    rather than comparing two unrelated construction methods.
    """
    if skinning_B is None:
        skinning_B = build_skinning_B()
    assert skinning_B.shape[1] == NUM_KEYPOINTS

    dominant_col = np.argmax(skinning_B, axis=1)
    region_idx = np.array([_kp21_col_to_region(int(c)) for c in dominant_col])

    region_B = np.zeros((skinning_B.shape[0], NUM_REGIONS), dtype=np.float64)
    region_B[np.arange(skinning_B.shape[0]), region_idx] = 1.0

    assert region_B.shape == (skinning_B.shape[0], NUM_REGIONS)
    assert np.allclose(region_B.sum(axis=1), 1.0)
    return region_B


def build_geometric_fallback_B(taxel_order: list[str] | None = None) -> np.ndarray:
    """(169, 21) SCIENTIFICALLY WEAKER fallback for when MANO_RIGHT.pkl is
    unavailable: no skinning weights, no kinematic tree, geometry only.

    Method: 6 anchor points (1 wrist/palm hub + 5 fingertip extrema) are
    found on the base mesh via iterative farthest-point sampling, with the
    hub identified as the anchor closest on average to the other 5 (a
    "central hub vs. extremities" heuristic, not a memorized vertex id).
    Each taxel's region assignment is a softmax over inverse-squared
    distance to these 6 anchors; a finger's region weight is then spread
    UNIFORMLY across its 4 keypoint columns (MCP/PIP/DIP/TIP), because
    without skinning weights there is no basis to tell those 4 apart.
    This resolves finger-level structure only, not joint-level structure.
    """
    print(
        "WARNING: build_geometric_fallback_B is a scientifically weaker "
        "fallback (no skinning weights, no kinematic tree, finger-level "
        "resolution only). Prefer build_skinning_B whenever MANO_RIGHT.pkl "
        "is available.",
    )
    if taxel_order is None:
        taxel_order = build_taxel_index()
    assert len(taxel_order) == NUM_TAXELS

    verts_base = _parse_obj_vertices(OBJ_BASE)
    verts_subdiv = _parse_obj_vertices(OBJ_SUBDIV)

    anchors_idx = [int(np.argmax(np.linalg.norm(verts_base - verts_base.mean(axis=0), axis=1)))]
    for _ in range(5):
        d_to_anchors = np.linalg.norm(
            verts_base[:, None, :] - verts_base[anchors_idx][None, :, :], axis=2
        )
        dmin = d_to_anchors.min(axis=1)
        anchors_idx.append(int(np.argmax(dmin)))
    anchors = verts_base[anchors_idx]  # (6,3)

    d_between = np.linalg.norm(anchors[:, None, :] - anchors[None, :, :], axis=2)
    hub = int(np.argmin(d_between.sum(axis=1)))
    finger_slots = [i for i in range(6) if i != hub]
    assert len(finger_slots) == 5

    layout = _load_layout(LAYOUT_HIGHRES)
    positions = layout["positions"]

    region_B = np.zeros((NUM_TAXELS, NUM_REGIONS), dtype=np.float64)
    for row, key in enumerate(taxel_order):
        vids = positions[key]["mano_vid"]
        assert len(vids) > 0, f"taxel {key} has an empty mano_vid list"
        taxel_xyz = verts_subdiv[vids].mean(axis=0)
        d2 = np.sum((anchors - taxel_xyz) ** 2, axis=1)
        inv = 1.0 / np.maximum(d2, 1e-12)
        region_B[row, 0] = inv[hub]
        for slot, anchor_i in enumerate(finger_slots):
            region_B[row, 1 + slot] = inv[anchor_i]
    region_B = region_B / region_B.sum(axis=1, keepdims=True)

    geometric_B = np.zeros((NUM_TAXELS, NUM_KEYPOINTS), dtype=np.float64)
    geometric_B[:, 0] = region_B[:, 0]
    for slot in range(5):
        block = 1 + 4 * slot
        geometric_B[:, block:block + 4] = region_B[:, 1 + slot:2 + slot] / 4.0

    assert geometric_B.shape == (NUM_TAXELS, NUM_KEYPOINTS)
    assert np.allclose(geometric_B.sum(axis=1), 1.0, atol=1e-8)
    return geometric_B


def to_attention_bias(B: np.ndarray, temperature: float = 1.0, floor: float = -10.0) -> np.ndarray:
    """Log-space additive pre-softmax bias from a (taxels, queries)
    correspondence matrix. Zero correspondence maps to `floor`, not -inf,
    so a learnable residual can still recover it. Returns (queries, taxels)
    to match attention logits [queries=joints/regions, keys=taxels].
    """
    assert B.ndim == 2 and B.shape[0] == NUM_TAXELS
    n_queries = B.shape[1]
    with np.errstate(divide="ignore"):
        log_b = np.log(B)
    bias = np.where(B > 0, log_b / temperature, floor)
    bias = np.maximum(bias, floor)
    bias = bias.T
    assert bias.shape == (n_queries, NUM_TAXELS)
    assert np.isfinite(bias).all()
    return bias


# ---------------------------------------------------------------------------
# CLI / build entrypoint
# ---------------------------------------------------------------------------

def _sanity_report(taxel_order: list[str], skinning_B: np.ndarray, region_B: np.ndarray) -> None:
    nonzero_thresh = 1e-6
    mean_nonzero_skinning = float((skinning_B > nonzero_thresh).sum(axis=1).mean())
    mean_nonzero_region = float((region_B > nonzero_thresh).sum(axis=1).mean())
    print(f"mean nonzero joints/taxel (skinning_B, > {nonzero_thresh}): {mean_nonzero_skinning:.2f}")
    print(f"mean nonzero regions/taxel (region_B, > {nonzero_thresh}): {mean_nonzero_region:.2f}")

    dominant_region = np.argmax(region_B, axis=1)
    print("\nper-finger breakdown (dominant skinning_B keypoint column per taxel, grouped by region_B):")
    for r_idx, r_name in enumerate(REGION_NAMES):
        rows = np.where(dominant_region == r_idx)[0]
        if rows.size == 0:
            print(f"  {r_name:6s}: 0 taxels")
            continue
        cols = np.argmax(skinning_B[rows], axis=1)
        col_counts = np.bincount(cols, minlength=NUM_KEYPOINTS)
        top_cols = np.argsort(-col_counts)[:4]
        top_str = ", ".join(f"kp{c}:{col_counts[c]}" for c in top_cols if col_counts[c] > 0)
        print(f"  {r_name:6s}: {rows.size:3d} taxels -> dominant skinning kp columns [{top_str}]")

    thumb_rows = np.where(dominant_region == 1)[0]
    thumb_cols = set(int(c) for c in np.argmax(skinning_B[thumb_rows], axis=1)) if thumb_rows.size else set()
    thumb_block = set(range(1, 5))
    if thumb_rows.size and not thumb_cols.issubset(thumb_block):
        print(
            f"  WARNING: thumb-region taxels have dominant skinning columns "
            f"{thumb_cols - thumb_block} outside the thumb block {thumb_block} "
            "-- this looks anatomically wrong, do not ship it as-is."
        )
    elif thumb_rows.size:
        print(f"  OK: all {thumb_rows.size} thumb-region taxels route to thumb keypoint columns {sorted(thumb_cols)}.")


def _cross_check_lowres(taxel_order: list[str], skinning_B: np.ndarray) -> None:
    """Independent correctness check (not a data source): for the subset
    of taxels lowres maps cleanly (in-range ids, non-empty list), build B
    from lowres ids directly (no bridge needed, they already index the base
    mesh) and compare against the highres-bridged skinning_B for the same
    taxels.
    """
    mano = load_mano_pkl()
    weights = np.asarray(mano["weights"], dtype=np.float64)
    n_base = weights.shape[0]
    J = np.asarray(mano["J"], dtype=np.float64)
    kintree = np.asarray(mano["kintree_table"])
    root, chains = _derive_finger_chains(kintree)
    ordered_chains = _order_finger_chains(J, chains)
    skinning_to_kp21 = _skinning_to_kp21_map(root, ordered_chains)

    lowres = _load_layout(LAYOUT_LOWRES)
    lo_positions = lowres["positions"]

    compared, mean_abs_diffs, argmax_agree = 0, [], 0
    for row, key in enumerate(taxel_order):
        vids = lo_positions[key]["mano_vid"]
        if len(vids) == 0 or any(v < 0 or v >= n_base for v in vids):
            continue
        w = weights[vids].mean(axis=0)
        lo_row = np.zeros(NUM_KEYPOINTS, dtype=np.float64)
        for joint, col in skinning_to_kp21.items():
            lo_row[col] = w[joint]
        lo_sum = lo_row.sum()
        if lo_sum <= 0:
            continue
        lo_row = lo_row / lo_sum

        hi_row = skinning_B[row].copy()
        hi_row[[1 + 4 * s + 3 for s in range(5)]] = 0.0  # lowres has no tip-geometry term to compare
        hi_sum = hi_row.sum()
        if hi_sum <= 0:
            continue
        hi_row = hi_row / hi_sum

        mean_abs_diffs.append(float(np.abs(lo_row - hi_row).mean()))
        argmax_agree += int(np.argmax(lo_row) == np.argmax(hi_row))
        compared += 1

    if compared == 0:
        print("lowres cross-check: 0 taxels were cleanly comparable; skipping.")
        return
    print(
        f"lowres cross-check over {compared} cleanly-mapped taxels "
        f"(non-tip columns, renormalized): mean abs diff = "
        f"{np.mean(mean_abs_diffs):.4f}, argmax agreement = "
        f"{argmax_agree / compared:.1%}"
    )
    if argmax_agree / compared < 0.8:
        print(
            "  WARNING: low agreement between the highres-bridged and lowres-direct "
            "skinning assignment -- something is wrong with one of the two paths, "
            "investigate before trusting either."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--geometric", action="store_true",
        help="Build the scientifically weaker geometric-only fallback instead of "
             "the skinning-based B (use when MANO_RIGHT.pkl is unavailable).",
    )
    args = ap.parse_args()

    taxel_order = build_taxel_index()
    print(f"canonical taxel order: {len(taxel_order)} taxels, "
          f"first={taxel_order[0]!r} last={taxel_order[-1]!r}")

    if args.geometric:
        skinning_B = build_geometric_fallback_B(taxel_order)
    else:
        skinning_B = build_skinning_B(taxel_order)
        _cross_check_lowres(taxel_order, skinning_B)

    region_B = build_region_B(skinning_B)
    bias_skinning = to_attention_bias(skinning_B)
    bias_region = to_attention_bias(region_B)
    assert bias_skinning.shape == (NUM_KEYPOINTS, NUM_TAXELS)
    assert bias_region.shape == (NUM_REGIONS, NUM_TAXELS)

    _sanity_report(taxel_order, skinning_B, region_B)

    ASSETS.mkdir(parents=True, exist_ok=True)
    np.savez(
        B_MATRICES_PATH,
        taxel_order=np.array(taxel_order),
        B_region=region_B,
        B_skinning=skinning_B,
        bias_region=bias_region,
        bias_skinning=bias_skinning,
    )
    print(f"\nsaved: {B_MATRICES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
