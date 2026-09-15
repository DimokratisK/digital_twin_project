"""
Cut open the inlet/outlet caps of a closed cardiac blood-pool STL for
SimVascular (or any solver that needs open boundaries in the geometry itself).

Reuses classify_la_multipv to tag faces as wall / outlet_MV / inlet_PV_N,
then DELETES the tagged inlet/outlet faces so the resulting STL has open
holes where SimVascular can create face IDs and BCs.

For LA CFD this produces:
    - <name>_open.stl        : wall-only surface with holes at MV + each PV
    - <name>_cap_outlet_MV.stl, <name>_cap_inlet_PV_N.stl  (optional)

Same primitive extends to other chambers later — you just need appropriate
probe STLs per valve (e.g. dilated LV probe as MV probe for the LA side).

Usage:
    # Just the open wall STL:
    python -m twin_core.cfd_pipeline.open_stl_caps \\
        --bloodpool ~/cfd_runs/bjonze_218/BloodPool.stl \\
        --pv ~/cfd_runs/bjonze_218/PV_1.stl \\
        --pv ~/cfd_runs/bjonze_218/PV_2.stl \\
        --pv ~/cfd_runs/bjonze_218/PV_3.stl \\
        -o ~/cfd_runs/bjonze_218/BloodPool_open.stl

    # Also emit each cap as its own STL (useful for re-closing later):
    python -m twin_core.cfd_pipeline.open_stl_caps \\
        --bloodpool ... --pv ... --pv ... \\
        -o ~/cfd_runs/bjonze_218/BloodPool_open.stl \\
        --export-caps

    # With MV + LAA probes (recommended if you have them):
    python -m twin_core.cfd_pipeline.open_stl_caps \\
        --bloodpool ... --pv ... --pv ... \\
        --mv-stl ~/cfd_runs/bjonze_218/MV_probe.stl \\
        --laa-stl ~/cfd_runs/bjonze_218/LAA.stl \\
        -o ~/cfd_runs/bjonze_218/BloodPool_open.stl
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh

from twin_core.cfd_pipeline.classify_la_multipv import classify_la_multipv


def open_caps(
    bloodpool: trimesh.Trimesh,
    regions: dict,
) -> tuple:
    """Return (open_wall_mesh, cap_meshes_by_name).

    open_wall_mesh   : bloodpool with all inlet/outlet faces removed
    cap_meshes_by_name: dict[str, trimesh.Trimesh] one entry per cap
    """
    wall_mask = regions["wall"]
    n_wall = int(wall_mask.sum())
    if n_wall == 0:
        raise RuntimeError("No wall faces left after classification — every face was tagged as inlet/outlet.")

    open_wall = bloodpool.copy()
    open_wall.update_faces(wall_mask)
    open_wall.remove_unreferenced_vertices()

    cap_meshes = {}
    for name, mask in regions.items():
        if name == "wall":
            continue
        if int(mask.sum()) == 0:
            continue
        cap = bloodpool.copy()
        cap.update_faces(mask)
        cap.remove_unreferenced_vertices()
        cap_meshes[name] = cap

    return open_wall, cap_meshes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bloodpool", required=True, type=Path,
                    help="Closed blood-pool STL (LA + LAA + PV stumps)")
    ap.add_argument("--pv", required=True, type=Path, action="append",
                    help="Per-PV probe STL (repeat: --pv PV_1.stl --pv PV_2.stl ...)")
    ap.add_argument("--mv-stl", type=Path, default=None,
                    help="Optional MV probe STL (LA voxels adjacent to LV). "
                         "Strongly recommended for reliable MV cap detection.")
    ap.add_argument("--laa-stl", type=Path, default=None,
                    help="Optional LAA STL (excludes LAA-adjacent faces from PV inlet classification)")
    ap.add_argument("-o", "--output", required=True, type=Path,
                    help="Output open wall STL (SimVascular-ready)")
    ap.add_argument("--export-caps", action="store_true",
                    help="Also write each cap as a separate STL alongside the output")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Vertex scale applied to output (0.001 = mm->m). "
                         "SimVascular usually wants mm, so leave at 1.0.")
    ap.add_argument("--proximity-threshold", type=float, default=2.0,
                    help="mm: face is 'near PV/LAA' if within this distance")
    ap.add_argument("--mv-proximity-threshold", type=float, default=None,
                    help="mm: face is 'near MV probe' if within this distance "
                         "(defaults to --proximity-threshold)")
    ap.add_argument("--distal-frac", type=float, default=0.20,
                    help="Fraction of PV long axis (from distal end) that is candidate inlet cap")
    ap.add_argument("--tip-normal-alignment", type=float, default=0.4,
                    help="Min cos(angle) between face normal and PV distal axis for inlet cap")
    ap.add_argument("--mv-depth-frac", type=float, default=0.08,
                    help="Only used when --mv-stl is not given (fallback heuristic)")
    ap.add_argument("--mv-normal-alignment", type=float, default=0.5,
                    help="Only used when --mv-stl is not given (fallback heuristic)")
    args = ap.parse_args()

    print(f"Loading blood pool: {args.bloodpool}")
    bp = trimesh.load(str(args.bloodpool), force="mesh")
    print(f"  faces={len(bp.faces)}  vertices={len(bp.vertices)}  "
          f"watertight={bp.is_watertight}  euler={bp.euler_number}")
    if not bp.is_watertight:
        print("  WARNING: blood pool is not watertight. Cap detection may leak into "
              "existing holes. Consider running prepare_cfd_mesh --repair first.")

    pv_meshes = []
    for p in args.pv:
        m = trimesh.load(str(p), force="mesh")
        print(f"  loaded PV probe {p.name}: faces={len(m.faces)}")
        pv_meshes.append(m)

    mv_mesh = None
    if args.mv_stl is not None:
        mv_mesh = trimesh.load(str(args.mv_stl), force="mesh")
        print(f"  loaded MV probe {args.mv_stl.name}: faces={len(mv_mesh.faces)}")
    else:
        print("  no --mv-stl given; falling back to anti-PV direction heuristic for MV")

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
    total = sum(int(m.sum()) for m in regions.values())
    for name, mask in regions.items():
        n = int(mask.sum())
        pct = 100.0 * n / total if total else 0.0
        print(f"  {name:<14}: {n:>8} faces  ({pct:5.2f}%)")

    for i in range(1, len(pv_meshes) + 1):
        if int(regions[f"inlet_PV_{i}"].sum()) < 20:
            print(f"  WARN: inlet_PV_{i} has < 20 faces — try larger --distal-frac or "
                  "--proximity-threshold before cutting.")
    if int(regions["outlet_MV"].sum()) < 50:
        print("  WARN: outlet_MV has < 50 faces — try larger --mv-depth-frac or supply --mv-stl.")

    open_wall, cap_meshes = open_caps(bp, regions)

    if args.scale != 1.0:
        open_wall.vertices *= args.scale
        for c in cap_meshes.values():
            c.vertices *= args.scale

    args.output.parent.mkdir(parents=True, exist_ok=True)
    open_wall.export(str(args.output))
    print(f"\nWrote open wall: {args.output}  "
          f"({len(open_wall.vertices)} verts, {len(open_wall.faces)} faces, "
          f"watertight={open_wall.is_watertight})")

    if args.export_caps:
        stem = args.output.stem
        for name, cap in cap_meshes.items():
            cap_path = args.output.parent / f"{stem}_cap_{name}.stl"
            cap.export(str(cap_path))
            print(f"  cap: {cap_path.name}  "
                  f"({len(cap.vertices)} verts, {len(cap.faces)} faces)")
