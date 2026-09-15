"""
Reconstruct the inputs `open_stl_caps.py` expects, starting only from the
per-structure STLs (LA, LAA, LV, PV) already on disk — without needing the
raw prediction NIfTI.

Normally the CFD input files come from
    pv_splitter.py           (needs the prediction .nii.gz)
    predictions_to_stl.py    (needs the split labelmap)

This script substitutes for both when the NIfTI is unavailable. It performs
three operations on the local STLs:

1. Split lumped `PV.stl` into per-vein probes (PV_1.stl … PV_N.stl) by
   connected components. The four PVs are anatomically disjoint in the
   segmentation (they only meet through the LA, which is not in this file),
   so component-splitting cleanly separates them.

2. Derive an MV probe by taking the LA faces that lie within a few mm of
   the LV surface. LA and LV were marched from adjacent voxel labels, so
   their surfaces coincide almost exactly along the MV plane.

3. Build a unified BloodPool.stl (LA ∪ LAA ∪ PV as one closed shell) by
   voxelising each mesh onto a common grid, OR-ing the three occupancy
   masks, then running marching cubes. This replicates what
   `pv_splitter.py --bloodpool-output` produces from the NIfTI, but at the
   mesh level.

The output folder can then be fed directly to `open_stl_caps.py`.

Usage:
    python -m twin_core.cfd_pipeline.stl_to_cfd_inputs \\
        -i C:/Users/dimok/Desktop/bjonze_218 \\
        -o C:/Users/dimok/Desktop/bjonze_218/cfd_inputs

    # Then:
    python -m twin_core.cfd_pipeline.open_stl_caps \\
        --bloodpool C:/Users/dimok/Desktop/bjonze_218/cfd_inputs/BloodPool.stl \\
        --pv        C:/Users/dimok/Desktop/bjonze_218/cfd_inputs/PV_1.stl \\
        --pv        C:/Users/dimok/Desktop/bjonze_218/cfd_inputs/PV_2.stl \\
        --pv        C:/Users/dimok/Desktop/bjonze_218/cfd_inputs/PV_3.stl \\
        --pv        C:/Users/dimok/Desktop/bjonze_218/cfd_inputs/PV_4.stl \\
        --mv-stl    C:/Users/dimok/Desktop/bjonze_218/cfd_inputs/MV_probe.stl \\
        --laa-stl   C:/Users/dimok/Desktop/bjonze_218/LAA.stl \\
        -o          C:/Users/dimok/Desktop/bjonze_218/cfd_inputs/BloodPool_open.stl
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np
import trimesh
from scipy.cluster.vq import kmeans2
from scipy.spatial import KDTree
from skimage import measure


REQUIRED = ["LA.stl", "LAA.stl", "LV.stl", "PV.stl"]


def _bisect_by_kmeans(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    # k-means (k=2) on vertex coordinates, then assign each face to the cluster
    # holding the majority of its 3 vertices. Robust to the thin voxel-bridge
    # that fuses two neighbouring PVs into one marching-cubes shell.
    _, labels = kmeans2(mesh.vertices, k=2, minit="++")
    face_labels = labels[mesh.faces]
    face_cluster = (face_labels.sum(axis=1) >= 2).astype(int)
    halves = []
    for c in (0, 1):
        mask = face_cluster == c
        if not mask.any():
            continue
        sub = mesh.copy()
        sub.update_faces(mask)
        sub.remove_unreferenced_vertices()
        halves.append(sub)
    return halves


def split_pv(
    pv_mesh: trimesh.Trimesh,
    min_faces: int,
    target_count: int | None = None,
) -> List[trimesh.Trimesh]:
    components = pv_mesh.split(only_watertight=False)
    kept = [c for c in components if len(c.faces) >= min_faces]
    kept.sort(key=lambda c: -len(c.faces))

    if target_count is None:
        return kept

    # Iteratively bisect the largest component until we hit target_count.
    while len(kept) < target_count:
        biggest = kept.pop(0)
        halves = _bisect_by_kmeans(biggest)
        if len(halves) < 2:
            kept.insert(0, biggest)
            print(f"  WARN: k-means bisection collapsed to 1 cluster on the largest "
                  f"component — stopping at {len(kept)} PVs.")
            break
        kept.extend(halves)
        kept.sort(key=lambda c: -len(c.faces))
    return kept


def derive_mv_probe(
    la_mesh: trimesh.Trimesh,
    lv_mesh: trimesh.Trimesh,
    proximity_mm: float,
) -> trimesh.Trimesh:
    # Distance from each LA face centroid to nearest LV vertex. LV marching-cubes
    # produces densely spaced vertices (~voxel pitch), so vertex-nearest is a
    # tight upper bound on surface-nearest — well within proximity_mm tolerance.
    tree = KDTree(lv_mesh.vertices)
    distances, _ = tree.query(la_mesh.triangles_center, k=1)
    mask = distances < proximity_mm
    if not mask.any():
        raise RuntimeError(
            f"No LA faces within {proximity_mm}mm of LV — check that LA.stl and "
            f"LV.stl are in the same world coordinate frame."
        )
    probe = la_mesh.copy()
    probe.update_faces(mask)
    probe.remove_unreferenced_vertices()
    return probe


def build_bloodpool(
    meshes: List[trimesh.Trimesh],
    voxel_size: float,
) -> trimesh.Trimesh:
    all_bounds = np.stack([m.bounds for m in meshes], axis=0)
    mins = all_bounds[:, 0, :].min(axis=0) - voxel_size * 5
    maxs = all_bounds[:, 1, :].max(axis=0) + voxel_size * 5
    shape = np.ceil((maxs - mins) / voxel_size).astype(int)

    combined = np.zeros(shape, dtype=bool)
    for m in meshes:
        vg = m.voxelized(pitch=voxel_size).fill()
        pts = vg.points
        idx = np.round((pts - mins) / voxel_size).astype(int)
        valid = np.all((idx >= 0) & (idx < shape), axis=1)
        idx = idx[valid]
        combined[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    verts, faces, _, _ = measure.marching_cubes(
        combined.astype(np.uint8),
        level=0.5,
        spacing=(voxel_size,) * 3,
    )
    verts = verts + mins
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input-dir", type=Path, required=True,
                    help=f"Folder containing {', '.join(REQUIRED)} (case-sensitive filenames)")
    ap.add_argument("-o", "--output-dir", type=Path, required=True,
                    help="Folder where reconstructed CFD inputs will be written")
    ap.add_argument("--voxel-size", type=float, default=0.5,
                    help="Voxel pitch (mm) for BloodPool rasterisation. Smaller = finer surface, "
                         "quadratic memory (default 0.5)")
    ap.add_argument("--mv-proximity", type=float, default=3.0,
                    help="LA faces within this many mm of LV surface become the MV probe "
                         "(default 3.0)")
    ap.add_argument("--min-pv-faces", type=int, default=50,
                    help="Discard PV components with fewer faces (default 50 — filters noise)")
    ap.add_argument("--target-pvs", type=int, default=None,
                    help="If set, force this many PV probes by k-means bisecting the largest "
                         "component until reaching the count. Use 4 for full per-vein resolution "
                         "when veins in a pair have fused into one shell (default: no bisection)")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        ap.error(f"Input dir does not exist: {args.input_dir}")
    missing = [n for n in REQUIRED if not (args.input_dir / n).is_file()]
    if missing:
        ap.error(f"Missing required STLs in {args.input_dir}: {', '.join(missing)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading STLs from {args.input_dir}")
    la  = trimesh.load(str(args.input_dir / "LA.stl"),  force="mesh")
    laa = trimesh.load(str(args.input_dir / "LAA.stl"), force="mesh")
    lv  = trimesh.load(str(args.input_dir / "LV.stl"),  force="mesh")
    pv  = trimesh.load(str(args.input_dir / "PV.stl"),  force="mesh")
    for name, m in [("LA", la), ("LAA", laa), ("LV", lv), ("PV", pv)]:
        print(f"  {name:<4}: faces={len(m.faces):>7}  watertight={m.is_watertight}")

    print("\n[1/3] Splitting PV.stl into per-vein probes")
    pv_components = split_pv(pv, min_faces=args.min_pv_faces, target_count=args.target_pvs)
    print(f"  produced {len(pv_components)} components with >= {args.min_pv_faces} faces "
          f"(expected ~4 for typical anatomy"
          f"{f'; target={args.target_pvs}' if args.target_pvs else ''})")
    if len(pv_components) < 2:
        print("  WARNING: PV split produced < 2 components. Check PV.stl in ParaView "
              "(Filters > Connectivity) to confirm the veins are actually disjoint.")
    pv_paths = []
    for i, comp in enumerate(pv_components, start=1):
        out = args.output_dir / f"PV_{i}.stl"
        comp.export(str(out))
        pv_paths.append(out)
        print(f"  wrote {out.name}  (faces={len(comp.faces)})")

    print("\n[2/3] Deriving MV probe from LA∩LV proximity")
    mv_probe = derive_mv_probe(la, lv, proximity_mm=args.mv_proximity)
    mv_path = args.output_dir / "MV_probe.stl"
    mv_probe.export(str(mv_path))
    print(f"  wrote {mv_path.name}  (faces={len(mv_probe.faces)}, "
          f"proximity_mm={args.mv_proximity})")
    if len(mv_probe.faces) < 100:
        print("  WARNING: MV probe has < 100 faces. Try --mv-proximity 5.0 for a larger patch.")

    print(f"\n[3/3] Building unified BloodPool from LA ∪ LAA ∪ PV (voxel={args.voxel_size}mm)")
    bloodpool = build_bloodpool([la, laa, pv], voxel_size=args.voxel_size)
    bp_path = args.output_dir / "BloodPool.stl"
    bloodpool.export(str(bp_path))
    print(f"  wrote {bp_path.name}  "
          f"(faces={len(bloodpool.faces)}, watertight={bloodpool.is_watertight}, "
          f"euler={bloodpool.euler_number})")
    if not bloodpool.is_watertight:
        print("  WARNING: BloodPool is not watertight. Marching-cubes on the OR'd mask "
              "should always produce a closed shell — inspect in ParaView before feeding "
              "to open_stl_caps.py.")

    print("\nDone. To cut the open caps for SimVascular, run:\n")
    pv_args = " ".join(f"--pv {p}" for p in pv_paths)
    print(f"  python -m twin_core.cfd_pipeline.open_stl_caps \\\n"
          f"    --bloodpool {bp_path} \\\n"
          f"    {pv_args} \\\n"
          f"    --mv-stl {mv_path} \\\n"
          f"    --laa-stl {args.input_dir / 'LAA.stl'} \\\n"
          f"    -o {args.output_dir / 'BloodPool_open.stl'}\n")


if __name__ == "__main__":
    main()
