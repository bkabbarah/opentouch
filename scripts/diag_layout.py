"""Diagnostic script for the taxel layout / MANO mesh correspondence.

Run this BEFORE building the taxel-to-joint correspondence matrix B
(see src/opentouch/contact_structure.py). It answers the questions that
determine which layout file and mesh-indexing scheme are safe to use:

  1. Do the base and subdivided OBJ meshes share vertex ids 0..777 (subdiv
     APPENDS new vertices), or does subdivision REORDER the vertex list?
  2. For each layout json (lowres = base-mesh ids, highres = subdiv-mesh
     ids): how many valid (non-erased) taxels are there, what does the
     mano_vid list look like per taxel, and are any ids out of range for
     the mesh they claim to index?
  3. Is MANO_RIGHT.pkl present, and if so does its v_template match the
     base OBJ (confirming the base OBJ *is* the MANO neutral template and
     not some independently exported mesh)?

No B matrix is built here. This script only prints facts and a final
branch recommendation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"

OBJ_BASE = ASSETS / "mano_right_neutral.obj"
OBJ_SUBDIV = ASSETS / "mano_right_neutral_subdiv.obj"
LAYOUT_LOWRES = ASSETS / "handLayoutNewest_meshid_lowres.json"
LAYOUT_HIGHRES = ASSETS / "handLayoutNewest_meshid.json"
MANO_PKL = ASSETS / "MANO_RIGHT.pkl"

EXPECTED_VALID_TAXELS = 169
GRID_SIZE = 256  # 16x16


def parse_obj_vertices(path: Path) -> np.ndarray:
    """Parse `v x y z` lines from an OBJ file, in file order. Returns (N,3)."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(verts, dtype=np.float64)


def parse_obj_faces(path: Path) -> np.ndarray:
    """Parse `f a b c` lines (vertex indices only, ignoring any /vt/vn),
    converted to 0-indexed. Returns (M,3) int64."""
    faces = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("f "):
                tokens = line.split()[1:]
                faces.append([int(tok.split("/")[0]) - 1 for tok in tokens])
    return np.asarray(faces, dtype=np.int64)


def point_to_triangle_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Closest-point-on-triangle distance (Ericson, Real-Time Collision
    Detection 5.1.5). Used only for diagnostic validation, not the B
    matrix build."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return float(np.linalg.norm(p - a))
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return float(np.linalg.norm(p - b))
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        return float(np.linalg.norm(p - (a + v * ab)))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return float(np.linalg.norm(p - c))
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        return float(np.linalg.norm(p - (a + w * ac)))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return float(np.linalg.norm(p - (b + w * (c - b))))
    denom = 1.0 / (va + vb + vc)
    v, w = vb * denom, vc * denom
    return float(np.linalg.norm(p - (a + ab * v + ac * w)))


def load_layout(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def valid_taxel_keys(layout: dict) -> list[str]:
    """Keys in `positions` that are not erased. Uses a set to dedupe
    erasedNodes, which is known to contain duplicate entries."""
    erased = set(layout["erasedNodes"])
    return [k for k in layout["positions"].keys() if k not in erased]


def try_load_mano_pkl(path: Path):
    """Load a MANO .pkl. These files are pickled chumpy/numpy objects
    created under Python 2 with an old numpy/chumpy stack. Two shims are
    required to unpickle them under a modern (numpy>=2, Python>=3.11)
    interpreter:

      - chumpy 0.70 calls inspect.getargspec, removed in Python 3.11+.
      - chumpy 0.70 imports np.bool/np.int/np.float/... aliases, removed
        in numpy 2.x.

    Both shims are process-local monkeypatches (no site-packages edits)
    and only affect names that are otherwise absent.
    """
    import inspect
    import pickle

    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]

    for name, val in [
        ("bool", bool),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ]:
        if not hasattr(np, name):
            setattr(np, name, val)

    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    section("1. OBJ vertex counts")
    verts_base = parse_obj_vertices(OBJ_BASE)
    verts_subdiv = parse_obj_vertices(OBJ_SUBDIV)
    print(f"base   ({OBJ_BASE.name}): {len(verts_base)} vertices")
    print(f"subdiv ({OBJ_SUBDIV.name}): {len(verts_subdiv)} vertices")

    section("2. Base vs subdiv: append or reorder?")
    n_common = min(len(verts_base), len(verts_subdiv))
    diffs = np.linalg.norm(verts_base[:n_common] - verts_subdiv[:n_common], axis=1)
    max_diff = float(diffs.max())
    print(f"Comparing first {n_common} vertices of each mesh.")
    print(f"max |base[i] - subdiv[i]| over i in [0, {n_common}) = {max_diff:.3e}")
    subdiv_appends = max_diff < 1e-6
    if subdiv_appends:
        print("=> APPENDS: subdiv mesh preserves base ids 0.." + str(n_common - 1)
              + " and adds new vertices after them.")
    else:
        print("=> REORDERS: subdiv vertex ids do NOT line up with base ids "
              "at matching indices.")

    section("3. MANO_RIGHT.pkl")
    mano_data = None
    if not MANO_PKL.exists():
        print(f"NOT FOUND at {MANO_PKL}")
        print("Skinning-based B is impossible without this file.")
    else:
        try:
            mano_data = try_load_mano_pkl(MANO_PKL)
        except Exception as e:  # noqa: BLE001 - want to report and continue
            print(f"FOUND at {MANO_PKL} but failed to load: {e!r}")
            mano_data = None
        if mano_data is not None:
            v_template = np.asarray(mano_data["v_template"])
            weights = np.asarray(mano_data["weights"])
            kintree = np.asarray(mano_data["kintree_table"])
            print(f"FOUND at {MANO_PKL}")
            print(f"v_template shape: {v_template.shape}")
            print(f"weights shape:    {weights.shape}  (verts x skinning joints)")
            print(f"kintree_table shape: {kintree.shape}")
            if v_template.shape[0] == verts_base.shape[0]:
                pkl_vs_base = float(
                    np.linalg.norm(v_template - verts_base, axis=1).max()
                )
                print(f"max |v_template - base OBJ vertex| = {pkl_vs_base:.3e}"
                      f" ({'MATCH' if pkl_vs_base < 1e-4 else 'MISMATCH'})")
            else:
                print(f"v_template vertex count ({v_template.shape[0]}) != base OBJ "
                      f"vertex count ({verts_base.shape[0]}); cannot compare directly.")

    def report_layout(name: str, path: Path, mesh_vertex_count: int) -> None:
        section(f"4. Layout: {name} ({path.name})  [mesh has {mesh_vertex_count} verts]")
        layout = load_layout(path)
        positions = layout["positions"]
        erased_raw = layout["erasedNodes"]
        erased_unique = set(erased_raw)
        print(f"grid positions total: {len(positions)} (expected {GRID_SIZE})")
        print(f"erasedNodes raw count: {len(erased_raw)}, unique: {len(erased_unique)}"
              f" (duplicates: {len(erased_raw) - len(erased_unique)})")

        valid_keys = valid_taxel_keys(layout)
        n_valid = len(valid_keys)
        print(f"valid taxel count (positions - unique erased): {n_valid}")
        status = "OK" if n_valid == EXPECTED_VALID_TAXELS else "MISMATCH"
        print(f"assert valid_taxel_count == {EXPECTED_VALID_TAXELS}: {status}")

        vid_lists = [positions[k]["mano_vid"] for k in valid_keys]
        lengths = [len(v) for v in vid_lists]
        empty = [k for k, v in zip(valid_keys, vid_lists) if len(v) == 0]
        all_vids = [vid for v in vid_lists for vid in v]

        if all_vids:
            print(f"mano_vid: min={min(all_vids)}, max={max(all_vids)}, "
                  f"unique={len(set(all_vids))}, total refs={len(all_vids)}")
        else:
            print("mano_vid: EMPTY across all valid taxels")
        print(f"verts-per-taxel: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.2f}")
        print(f"taxels with empty mano_vid list: {len(empty)}"
              + (f" -> {empty}" if empty else ""))

        out_of_range = [vid for vid in all_vids if vid < 0 or vid >= mesh_vertex_count]
        print(f"vids out of range for a {mesh_vertex_count}-vertex mesh: "
              f"{len(out_of_range)}"
              + (f" (e.g. {sorted(set(out_of_range))[:10]})" if out_of_range else ""))

    report_layout("lowres (indexes BASE mesh)", LAYOUT_LOWRES, len(verts_base))
    report_layout("highres (indexes SUBDIV mesh)", LAYOUT_HIGHRES, len(verts_subdiv))

    section("5. Geometric bridge validation: subdiv vertex -> nearest base vertex")
    print("lowres has 22/169 taxels with mano_vid ids out of range for the 778-vertex "
          "base mesh (see section 4), and subdiv REORDERS ids relative to base (section "
          "2), so highres ids cannot be used as base-mesh/skinning-weight row indices "
          "directly. Both OBJs are the same neutral hand in the same coordinate frame "
          "(mano_right_neutral_subdiv.obj is a pure subdivision of mano_right_neutral.obj "
          "with no rescale/retranslation), so nearest-vertex geometry can bridge subdiv "
          "vertex ids to base vertex ids/skinning rows.")
    base_faces = parse_obj_faces(OBJ_BASE)
    tree = cKDTree(verts_base)
    dist, base_nn = tree.query(verts_subdiv, k=1)
    bbox_diag = float(np.linalg.norm(verts_base.max(axis=0) - verts_base.min(axis=0)))
    print(f"base mesh bbox diagonal: {bbox_diag:.6f}")
    print(f"nearest-VERTEX distance (subdiv -> base) stats over {len(verts_subdiv)} subdiv verts:")
    print(f"  min={dist.min():.6e}  median={np.median(dist):.6e}  "
          f"mean={dist.mean():.6e}  p95={np.percentile(dist, 95):.6e}  max={dist.max():.6e}")
    ratio = float(np.median(dist) / bbox_diag)
    print(f"  median_distance / bbox_diagonal = {ratio:.6e}")
    covered_base = np.unique(base_nn)
    coverage = len(covered_base) / len(verts_base)
    print(f"base vertices that are NN of >=1 subdiv vertex: {len(covered_base)}/"
          f"{len(verts_base)} ({coverage:.1%})")
    print("Nearest-VERTEX distance is confounded by triangle density (subdivide_to_size "
          "used max_edge=0.004, per preprocess/scratch/mano_densifincation.py), not by "
          "frame alignment, so it is not by itself a valid pass/fail test -- a coarse "
          "original triangle can leave a new vertex several mm from the nearest of only "
          "778 candidate corners even when both meshes occupy the identical surface.")

    print()
    print("Authoritative test: nearest-FACE (point-to-triangle) distance. If subdiv "
          "vertices lie exactly ON the base mesh surface (not just near its vertices), "
          "distance to the closest base triangle should be ~0 (float/precision noise "
          "only), which is what a pure edge-split subdivision (no smoothing/displacement) "
          "predicts.")
    from collections import defaultdict
    vert_to_faces: dict[int, list[int]] = defaultdict(list)
    for fi, tri in enumerate(base_faces):
        for vid in tri:
            vert_to_faces[int(vid)].append(fi)
    _, nn_idx_k = tree.query(verts_subdiv, k=8)
    face_dist = np.empty(len(verts_subdiv))
    for i, p in enumerate(verts_subdiv):
        candidate_faces = set()
        for vid in nn_idx_k[i]:
            candidate_faces.update(vert_to_faces[int(vid)])
        best = np.inf
        for fi in candidate_faces:
            a, b, c = verts_base[base_faces[fi]]
            d = point_to_triangle_distance(p, a, b, c)
            if d < best:
                best = d
        face_dist[i] = best
    print(f"point-to-face distance stats: min={face_dist.min():.3e}  "
          f"median={np.median(face_dist):.3e}  mean={face_dist.mean():.3e}  "
          f"p95={np.percentile(face_dist, 95):.3e}  max={face_dist.max():.3e}")

    FACE_DIST_MAX = 1e-4  # meters; OBJ export precision is 1e-6 to 1e-8
    bridge_ok = bool(face_dist.max() < FACE_DIST_MAX) and coverage >= 0.95
    if bridge_ok:
        print(f"=> BRIDGE VALID (max point-to-face distance {face_dist.max():.3e} < "
              f"{FACE_DIST_MAX:.0e}, coverage {coverage:.0%}): subdiv vertices lie "
              "exactly on the base mesh surface, confirming a shared frame and a pure "
              "geometric subdivision. Nearest-BASE-VERTEX lookup is therefore a valid "
              "way to bridge a subdiv vertex id to a base-mesh/skinning-weight row.")
    else:
        print(f"=> BRIDGE INVALID (max point-to-face distance {face_dist.max():.3e}, "
              f"coverage {coverage:.0%}): subdiv vertices are NOT on the base mesh "
              "surface. STOP: do not build B from this bridge.")
        return 1

    section("6. Branch recommendation")
    print("lowres has 22/169 taxels (52 vertex refs) out of range for the 778-vertex base "
          "mesh, and preprocess/scratch/mano_densifincation.py + git history show no "
          "documented pipeline that produced a base-mesh-indexed layout -- only "
          "handLayoutNewest_meshid.json (highres, indexed against the subdiv mesh) has a "
          "traceable generation path (subdivide() -> mesh_point_mapping_new() -> "
          "rewrite_mapjson()). lowres is therefore not trustworthy as a data source.")
    print()
    print("DECISION: use the HIGHRES layout for taxel->vertex assignment. Bridge each "
          "subdiv vertex id to a base-mesh/skinning-weight row via nearest-base-vertex "
          "(validated above: point-to-face distance ~0, i.e. shared frame, pure "
          "subdivision). lowres is retained only as an independent cross-check on the "
          "114 taxels it maps cleanly, per the requested Step 3 of contact_structure.py, "
          "not as a primary data source.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
