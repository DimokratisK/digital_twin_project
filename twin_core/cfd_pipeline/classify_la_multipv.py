"""
Classify faces of the LA blood-pool STL into wall / outlet_MV / inlet_PV_N
using per-PV STLs as spatial probes.

Inputs:
    - BloodPool.stl: single watertight surface (LA + LAA + all PV stumps).
    - PV_1.stl, PV_2.stl, ...: per-PV closed surfaces (from pv_splitter +
      predictions_to_stl --dataset bjonze_cfd).
    - [optional] LAA.stl: not required — LAA is classified as wall regardless.

Algorithm:
    1. PCA of blood pool -> LA long axis + MV base plane (same as classify_la_faces).
    2. For each BloodPool face whose center is within `proximity_threshold` of
       any PV_N STL, mark as "near PV_N".
    3. For each near-PV_N face, project onto PV_N's own PCA long axis. Faces
       in the distal `distal_frac` of that axis with normals aligned to the
       distal direction form the inlet cap for PV_N.
    4. Faces near the LA base with outward-pointing normals form outlet_MV.
    5. Everything else is wall (LA body, LAA, PV tube walls).

Output: OpenFOAM-compatible multi-region ASCII STL with named solids:
    wall, outlet_MV, inlet_PV_1, ..., inlet_PV_N

Usage:
    python -m twin_core.cfd_pipeline.classify_la_multipv \\
        --bloodpool ~/cfd_runs/bjonze_168_LA_stls/BloodPool.stl \\
        --pv ~/cfd_runs/bjonze_168_LA_stls/PV_1.stl \\
        --pv ~/cfd_runs/bjonze_168_LA_stls/PV_2.stl \\
        --pv ~/cfd_runs/bjonze_168_LA_stls/PV_3.stl \\
        -o ~/cfd_runs/bjonze_168_LA_stls/bjonze_168_multiregion.stl \\
        --scale 0.001      # mm -> metres for OpenFOAM

    # Analyze only (no output):
    python -m twin_core.cfd_pipeline.classify_la_multipv \\
        --bloodpool ... --pv ... --analyze
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from twin_core.cfd_pipeline.cut_valve_openings import (
    find_chamber_base,
    write_multi_region_stl,
)


def _pca_long_axis(vertices: np.ndarray) -> np.ndarray:
    centered = vertices - vertices.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return eigvecs[:, np.argmax(eigvals)]


def classify_la_multipv(
    bloodpool: trimesh.Trimesh,
    pv_meshes: List[trimesh.Trimesh],
    mv_mesh: Optional[trimesh.Trimesh] = None,
    laa_mesh: Optional[trimesh.Trimesh] = None,
    proximity_threshold: float = 2.0,
    mv_proximity_threshold: Optional[float] = None,
    distal_frac: float = 0.20,
    tip_normal_alignment: float = 0.4,
    mv_depth_frac: float = 0.04,
    mv_normal_alignment: float = 0.7,
) -> Dict[str, np.ndarray]:
    """Return face-region masks for the blood-pool mesh.

    If mv_mesh is provided, outlet_MV is classified by proximity to that probe
    (typically LA voxels adjacent to LV voxels). Otherwise falls back to the
    anti-PV direction heuristic.

    If laa_mesh is provided, faces near the LAA are excluded from PV inlet
    classification (forced to wall).

    All length parameters are in the units of the input STL (mm if from
    predictions_to_stl.py without scaling).
    """
    face_centers = bloodpool.triangles_center
    face_normals = bloodpool.face_normals
    n_faces = len(face_centers)

    la_centroid = bloodpool.vertices.mean(axis=0)
    mv_prox = mv_proximity_threshold if mv_proximity_threshold is not None else proximity_threshold

    # Pre-compute near-LAA mask if LAA probe given
    near_laa = np.zeros(n_faces, dtype=bool)
    if laa_mesh is not None:
        print(f"  [LAA] KDTree ({len(laa_mesh.vertices)} vertices) + query...",
              flush=True)
        laa_tree = cKDTree(laa_mesh.vertices)
        laa_dists, _ = laa_tree.query(face_centers, k=1)
        near_laa = laa_dists < proximity_threshold
        print(f"  [LAA] near-LAA faces: {int(near_laa.sum())}", flush=True)

    # 1. MV classification
    if mv_mesh is not None:
        print(f"  [MV] KDTree ({len(mv_mesh.vertices)} vertices) + query...",
              flush=True)
        mv_tree = cKDTree(mv_mesh.vertices)
        mv_dists, _ = mv_tree.query(face_centers, k=1)
        outlet_mv = mv_dists < mv_prox
        print(f"  [MV] near-MV faces: {int(outlet_mv.sum())}", flush=True)
    else:
        # Fallback: anti-PV direction heuristic
        pv_centroid_mean = np.mean(
            [pv.vertices.mean(axis=0) for pv in pv_meshes], axis=0
        )
        mv_axis = la_centroid - pv_centroid_mean
        mv_axis = mv_axis / np.linalg.norm(mv_axis)

        face_pos_along_mv = (face_centers - la_centroid) @ mv_axis
        mv_pos_range = face_pos_along_mv.max() - face_pos_along_mv.min()
        mv_pos_threshold = face_pos_along_mv.max() - mv_depth_frac * mv_pos_range
        near_mv = face_pos_along_mv > mv_pos_threshold
        mv_normal_aligned = (face_normals @ mv_axis) > mv_normal_alignment
        outlet_mv = near_mv & mv_normal_aligned

    # Never let MV overlap LAA
    outlet_mv &= ~near_laa

    # 2. Per-PV inlet caps
    inlet_masks: List[np.ndarray] = []
    any_near_pv = np.zeros(n_faces, dtype=bool)
    for i, pv in enumerate(pv_meshes, start=1):
        print(f"  [PV_{i}] building KDTree ({len(pv.vertices)} vertices)...", flush=True)
        tree = cKDTree(pv.vertices)
        print(f"  [PV_{i}] querying {len(face_centers)} face centers...", flush=True)
        dists, _ = tree.query(face_centers, k=1)
        near_pv = dists < proximity_threshold
        any_near_pv |= near_pv
        print(f"  [PV_{i}] near-PV faces: {int(near_pv.sum())}", flush=True)

        pv_axis = _pca_long_axis(pv.vertices)
        pv_centroid = pv.vertices.mean(axis=0)
        if (pv_centroid - la_centroid) @ pv_axis < 0:
            pv_axis = -pv_axis  # point distally (away from LA)

        pv_vert_proj = (pv.vertices - pv_centroid) @ pv_axis
        pv_min, pv_max = pv_vert_proj.min(), pv_vert_proj.max()
        pv_extent = pv_max - pv_min

        face_proj = (face_centers - pv_centroid) @ pv_axis
        in_distal = face_proj > pv_max - distal_frac * pv_extent
        aligned_with_tip = (face_normals @ pv_axis) > tip_normal_alignment

        inlet_cap = near_pv & in_distal & aligned_with_tip
        # Exclude any LAA-adjacent faces from PV inlet (prevents LAA tip being
        # mis-tagged as PV inlet if model confused LAA voxels with PV)
        inlet_cap &= ~near_laa
        inlet_masks.append(inlet_cap)

    # Exclude PV tube walls from MV candidates
    outlet_mv &= ~any_near_pv

    # 3. Compose regions with priority: inlet > outlet > wall
    region_labels = np.full(n_faces, -1, dtype=np.int32)  # -1 = wall
    for i, m in enumerate(inlet_masks, start=1):
        region_labels[m & (region_labels == -1)] = i
    region_labels[outlet_mv & (region_labels == -1)] = 0

    regions: Dict[str, np.ndarray] = {
        "wall": region_labels == -1,
        "outlet_MV": region_labels == 0,
    }
    for i in range(1, len(pv_meshes) + 1):
        regions[f"inlet_PV_{i}"] = region_labels == i

    return regions


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bloodpool", required=True, type=Path,
                    help="Blood-pool STL (single watertight surface)")
    ap.add_argument("--pv", required=True, type=Path, action="append",
                    help="Per-PV STL (repeat for each PV: --pv PV_1.stl --pv PV_2.stl ...)")
    ap.add_argument("--mv-stl", type=Path, default=None,
                    help="Optional MV probe STL (LA voxels adjacent to LV). "
                         "If given, outlet_MV is classified via proximity to this probe "
                         "instead of the anti-PV heuristic.")
    ap.add_argument("--laa-stl", type=Path, default=None,
                    help="Optional LAA STL. Excludes LAA-adjacent faces from PV inlet "
                         "classification (safety net if model confused LAA with PV).")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output multi-region STL (omit for --analyze)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Vertex scale factor applied to output (0.001 = mm->m for OpenFOAM)")
    ap.add_argument("--proximity-threshold", type=float, default=2.0,
                    help="mm: face is 'near PV/LAA' if within this distance")
    ap.add_argument("--mv-proximity-threshold", type=float, default=None,
                    help="mm: face is 'near MV probe' if within this distance "
                         "(defaults to --proximity-threshold)")
    ap.add_argument("--distal-frac", type=float, default=0.20,
                    help="Fraction of PV long axis (from distal end) that is candidate inlet cap")
    ap.add_argument("--tip-normal-alignment", type=float, default=0.4,
                    help="Min cos(angle) between face normal and PV distal axis for inlet cap")
    ap.add_argument("--mv-depth-frac", type=float, default=0.08)
    ap.add_argument("--mv-normal-alignment", type=float, default=0.5)
    ap.add_argument("--analyze", action="store_true",
                    help="Print diagnostics only, do not write output")
    args = ap.parse_args()

    if not args.analyze and args.output is None:
        ap.error("Must pass -o/--output unless --analyze is set")

    print(f"Loading blood pool: {args.bloodpool}")
    bp = trimesh.load(str(args.bloodpool), force="mesh")
    print(f"  faces={len(bp.faces)}  vertices={len(bp.vertices)}  "
          f"watertight={bp.is_watertight}  euler={bp.euler_number}")
    if not bp.is_watertight:
        print("  WARNING: blood pool is not watertight. "
              "Consider running prepare_cfd_mesh --repair first.")

    pv_meshes = []
    for p in args.pv:
        m = trimesh.load(str(p), force="mesh")
        print(f"  loaded PV probe {p.name}: faces={len(m.faces)}  "
              f"watertight={m.is_watertight}")
        pv_meshes.append(m)

    mv_mesh = None
    if args.mv_stl is not None:
        mv_mesh = trimesh.load(str(args.mv_stl), force="mesh")
        print(f"  loaded MV probe {args.mv_stl.name}: faces={len(mv_mesh.faces)}")

    laa_mesh = None
    if args.laa_stl is not None:
        laa_mesh = trimesh.load(str(args.laa_stl), force="mesh")
        print(f"  loaded LAA probe {args.laa_stl.name}: faces={len(laa_mesh.faces)}")

    regions = classify_la_multipv(
        bp, pv_meshes,
        mv_mesh=mv_mesh,
        laa_mesh=laa_mesh,
        proximity_threshold=args.proximity_threshold,
        mv_proximity_threshold=args.mv_proximity_threshold,
        distal_frac=args.distal_frac,
        tip_normal_alignment=args.tip_normal_alignment,
        mv_depth_frac=args.mv_depth_frac,
        mv_normal_alignment=args.mv_normal_alignment,
    )

    print("\nFace classification:")
    total = sum(m.sum() for m in regions.values())
    for name, mask in regions.items():
        n = int(mask.sum())
        pct = 100.0 * n / total if total else 0.0
        print(f"  {name:<12}: {n:>8} faces  ({pct:5.2f}%)")
    assert total == len(bp.faces), "region masks do not partition the mesh"

    # Sanity checks
    issues = []
    for i in range(1, len(pv_meshes) + 1):
        if regions[f"inlet_PV_{i}"].sum() < 20:
            issues.append(
                f"inlet_PV_{i} has only {regions[f'inlet_PV_{i}'].sum()} faces "
                "(may need larger --distal-frac or --proximity-threshold)"
            )
    if regions["outlet_MV"].sum() < 50:
        issues.append(f"outlet_MV has only {regions['outlet_MV'].sum()} faces "
                      "(may need larger --mv-depth-frac)")
    if issues:
        print("\nWARNINGS:")
        for s in issues:
            print(f"  - {s}")

    if args.analyze:
        print("\n--analyze mode: no output written.")
        return

    write_multi_region_stl(bp, regions, str(args.output), scale=args.scale)


if __name__ == "__main__":
    main()
