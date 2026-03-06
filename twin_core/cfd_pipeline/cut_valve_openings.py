"""
Identify valve regions on a closed cardiac chamber STL and create a
multi-region STL for OpenFOAM meshing with inlet/outlet patches.

The LV segmentation from nnU-Net produces a closed surface (all wall, no
openings). For CFD, we need to designate regions at the base of the LV as:
  - inlet: mitral valve (blood enters during diastole)
  - outlet: aortic valve (blood exits during systole)
  - wall: myocardial surface (no-slip)

The surface stays watertight — no holes are cut. Instead, we label face
regions so snappyHexMesh creates separate patches for each. Blood "passes
through" the inlet/outlet patches via boundary conditions.

Usage:
    # Analyze LV geometry and show base plane:
    python -m twin_core.cfd_pipeline.cut_valve_openings \
        --analyze meshes/patient006_frame00/LV.stl

    # Create multi-region STL (in metres, ready for OpenFOAM):
    python -m twin_core.cfd_pipeline.cut_valve_openings \
        --stl meshes/patient006_frame00/LV.stl \
        -o cfd_meshes/patient006_frame00/LV_valves.stl

    # Also generate topoSet/createPatch dicts for existing mesh:
    python -m twin_core.cfd_pipeline.cut_valve_openings \
        --stl meshes/patient006_frame00/LV.stl \
        --toposet-dir cfd_cases/patient006_frame00/system
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh


def find_chamber_base(
    mesh: trimesh.Trimesh,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the base plane of an LV mesh using PCA.

    The LV is roughly bullet-shaped: narrow apex at one end, wide base
    (where valves are) at the other. We find the long axis via PCA,
    then identify the wider end as the base.

    Returns
    -------
    base_center : (3,) center of the base region
    base_normal : (3,) outward normal at the base (pointing away from apex)
    long_axis : (3,) unit vector along the LV long axis (apex → base)
    """
    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # PCA: eigenvectors of covariance matrix
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Long axis = eigenvector with largest eigenvalue
    long_axis = eigenvectors[:, np.argmax(eigenvalues)]

    # Project vertices onto long axis
    projections = centered @ long_axis

    # Check which end is the base (wider cross-section)
    extent = projections.max() - projections.min()
    threshold = 0.15 * extent  # top/bottom 15%

    near_min = projections < (projections.min() + threshold)
    near_max = projections > (projections.max() - threshold)

    # Perpendicular spread at each end
    perp_axes = eigenvectors[:, np.argsort(eigenvalues)[:2]]

    spread_min = np.var(centered[near_min] @ perp_axes) if near_min.sum() > 3 else 0
    spread_max = np.var(centered[near_max] @ perp_axes) if near_max.sum() > 3 else 0

    if spread_max >= spread_min:
        base_direction = 1.0
        base_proj_val = projections.max()
    else:
        base_direction = -1.0
        base_proj_val = projections.min()

    # Long axis points from apex toward base
    long_axis = long_axis * base_direction
    base_normal = long_axis.copy()

    # Recalculate projections with corrected axis direction
    projections = centered @ long_axis

    # Base center: mean of vertices in the top 15%
    near_base = projections > (projections.max() - threshold)
    if near_base.sum() > 0:
        base_center = vertices[near_base].mean(axis=0)
    else:
        base_center = centroid + base_proj_val * long_axis

    return base_center, base_normal, long_axis


def classify_faces(
    mesh: trimesh.Trimesh,
    base_center: np.ndarray,
    base_normal: np.ndarray,
    base_depth_frac: float = 0.12,
    inlet_angle_range: float = 220.0,
) -> Dict[str, np.ndarray]:
    """Classify mesh faces into wall, inlet (MV), and outlet (AV).

    Faces near the base plane whose normals point roughly outward (along
    the base normal) are candidates for valve patches. These are split
    angularly: the larger portion (~60%) becomes the inlet (mitral valve),
    and the smaller portion (~40%) becomes the outlet (aortic valve).

    Parameters
    ----------
    base_depth_frac : fraction of total LV length to include from the base
    inlet_angle_range : angular span (degrees) for the inlet region

    Returns
    -------
    dict with 'wall', 'inlet', 'outlet' keys → boolean arrays of len(faces)
    """
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals

    # Distance along base normal from base center
    to_face = face_centers - base_center
    dist_along = to_face @ base_normal

    # Total extent along long axis
    to_vert = mesh.vertices - base_center
    vert_dist = to_vert @ base_normal
    total_extent = vert_dist.max() - vert_dist.min()
    depth_threshold = base_depth_frac * total_extent

    # Faces near the base: within depth_threshold AND past the base center
    # (i.e., on the base side, not deep inside the LV)
    near_base = dist_along > -depth_threshold

    # Face normal should be somewhat aligned with base normal
    alignment = face_normals @ base_normal
    facing_outward = alignment > 0.3

    # Candidate valve faces
    candidates = near_base & facing_outward

    if candidates.sum() < 10:
        # Fallback: use top N% of faces by position along normal
        cutoff = np.percentile(dist_along, 100 * (1 - base_depth_frac))
        candidates = dist_along > cutoff
        print(f"  Warning: few base-aligned faces found, using position-only "
              f"criterion ({candidates.sum()} faces)")

    # Split candidates into inlet (MV) and outlet (AV) by angular position
    # in the base plane
    arbitrary = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(base_normal, arbitrary)) > 0.9:
        arbitrary = np.array([0.0, 1.0, 0.0])
    axis1 = np.cross(base_normal, arbitrary)
    axis1 /= np.linalg.norm(axis1)
    axis2 = np.cross(base_normal, axis1)

    candidate_indices = np.where(candidates)[0]
    relative = face_centers[candidates] - base_center
    x = relative @ axis1
    y = relative @ axis2
    angles = np.arctan2(y, x)  # [-pi, pi]

    # Inlet spans inlet_angle_range degrees, outlet gets the rest
    inlet_half = np.radians(inlet_angle_range / 2)

    inlet_mask = np.zeros(len(mesh.faces), dtype=bool)
    outlet_mask = np.zeros(len(mesh.faces), dtype=bool)

    for i, idx in enumerate(candidate_indices):
        if -inlet_half <= angles[i] <= inlet_half:
            inlet_mask[idx] = True
        else:
            outlet_mask[idx] = True

    wall_mask = ~(inlet_mask | outlet_mask)

    return {
        "wall": wall_mask,
        "inlet": inlet_mask,
        "outlet": outlet_mask,
    }


def analyze_stl(stl_path: str) -> Dict:
    """Analyze an LV STL and return base plane information."""
    mesh = trimesh.load(str(stl_path), force="mesh")

    base_center, base_normal, long_axis = find_chamber_base(mesh)
    bounds = mesh.bounds

    # Quick face classification to show counts
    regions = classify_faces(mesh, base_center, base_normal)

    return {
        "file": str(stl_path),
        "n_faces": len(mesh.faces),
        "n_vertices": len(mesh.vertices),
        "bounds_min": bounds[0].tolist(),
        "bounds_max": bounds[1].tolist(),
        "center": mesh.centroid.tolist(),
        "base_center": base_center.tolist(),
        "base_normal": base_normal.tolist(),
        "long_axis": long_axis.tolist(),
        "inlet_faces": int(regions["inlet"].sum()),
        "outlet_faces": int(regions["outlet"].sum()),
        "wall_faces": int(regions["wall"].sum()),
    }


def write_multi_region_stl(
    mesh: trimesh.Trimesh,
    face_regions: Dict[str, np.ndarray],
    output_path: str,
    scale: float = 1.0,
):
    """Write an ASCII STL with named solid regions.

    OpenFOAM reads named solids as separate surface patches.

    Parameters
    ----------
    scale : multiply all vertex coordinates by this factor (e.g. 0.001 for mm→m)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vertices = mesh.vertices * scale

    with open(output_path, "w") as f:
        for region_name, mask in face_regions.items():
            if not mask.any():
                continue

            region_faces = mesh.faces[mask]
            region_normals = mesh.face_normals[mask]

            f.write(f"solid {region_name}\n")
            for face, normal in zip(region_faces, region_normals):
                v0, v1, v2 = vertices[face]
                f.write(
                    f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
                )
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid {region_name}\n")

    n_total = sum(m.sum() for m in face_regions.values())
    print(f"  Wrote multi-region STL: {output_path}")
    print(f"    Total faces: {n_total}")
    for name, mask in face_regions.items():
        print(f"    {name}: {mask.sum()} faces")
    if scale != 1.0:
        print(f"    Scale applied: {scale} (coordinates in metres)")


def generate_toposet_dict(
    base_center: np.ndarray,
    base_normal: np.ndarray,
    depth: float,
    scale: float = 0.001,
) -> str:
    """Generate OpenFOAM topoSetDict to select valve faces on existing mesh.

    Uses a box region near the base to select faces, then subsets by patch.
    Coordinates are in metres (scaled from mm).

    Parameters
    ----------
    base_center : base center in original STL units (mm)
    base_normal : base normal direction
    depth : depth from base to select, in original units (mm)
    scale : coordinate scaling factor (0.001 for mm→m)
    """
    bc = base_center * scale
    bn = base_normal
    d = depth * scale

    # Create a box around the base region
    # The box extends from (base_center - depth*normal) to (base_center + small_offset*normal)
    # and is wide enough to cover the entire cross-section
    p1 = bc - d * bn - 0.05 * np.ones(3)  # generous padding
    p2 = bc + 0.002 * bn + 0.05 * np.ones(3)

    box_min = np.minimum(p1, p2)
    box_max = np.maximum(p1, p2)

    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}

actions
(
    // Select all faces on the LV wall patch
    {{
        name    baseFaceSet;
        type    faceSet;
        action  new;
        source  boxToFace;
        sourceInfo
        {{
            box ({box_min[0]:.6f} {box_min[1]:.6f} {box_min[2]:.6f}) ({box_max[0]:.6f} {box_max[1]:.6f} {box_max[2]:.6f});
        }}
    }}
);
"""


def generate_create_patch_dict() -> str:
    """Generate OpenFOAM createPatchDict to split wall into inlet/outlet."""
    return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      createPatchDict;
}

pointSync false;

patches
(
    {
        name            inlet;
        patchInfo
        {
            type patch;
        }
        constructFrom set;
        set baseFaceSet;
    }
);
"""


def prepare_valve_stl(
    stl_path: str,
    output_path: str,
    scale_to_metres: bool = True,
    base_depth_frac: float = 0.12,
    inlet_angle_range: float = 220.0,
) -> Dict:
    """Full pipeline: load STL, find base, classify faces, write multi-region STL.

    Parameters
    ----------
    stl_path : input closed STL (mm units from segmentation)
    output_path : output multi-region STL
    scale_to_metres : if True, scale coordinates by 0.001 (mm → m)
    base_depth_frac : fraction of LV length to treat as base region
    inlet_angle_range : angular span (degrees) for inlet (MV) region

    Returns
    -------
    dict with analysis results and region face counts
    """
    mesh = trimesh.load(str(stl_path), force="mesh")

    print(f"\n=== Preparing valve openings ===")
    print(f"  Input: {stl_path}")
    print(f"  Faces: {len(mesh.faces):,}")
    print(f"  Vertices: {len(mesh.vertices):,}")

    # Find base
    base_center, base_normal, long_axis = find_chamber_base(mesh)
    print(f"  Base center (mm): [{base_center[0]:.1f}, {base_center[1]:.1f}, {base_center[2]:.1f}]")
    print(f"  Base normal: [{base_normal[0]:.3f}, {base_normal[1]:.3f}, {base_normal[2]:.3f}]")

    # Classify faces
    regions = classify_faces(
        mesh, base_center, base_normal,
        base_depth_frac=base_depth_frac,
        inlet_angle_range=inlet_angle_range,
    )

    # Write multi-region STL
    scale = 0.001 if scale_to_metres else 1.0
    write_multi_region_stl(mesh, regions, output_path, scale=scale)

    return {
        "base_center_mm": base_center.tolist(),
        "base_normal": base_normal.tolist(),
        "long_axis": long_axis.tolist(),
        "inlet_faces": int(regions["inlet"].sum()),
        "outlet_faces": int(regions["outlet"].sum()),
        "wall_faces": int(regions["wall"].sum()),
        "total_faces": len(mesh.faces),
        "scale_applied": scale,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Identify valve regions on cardiac STL for CFD"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--analyze", type=str,
        help="Analyze STL and print base plane info"
    )
    group.add_argument(
        "--stl", type=str,
        help="Input STL to process"
    )

    parser.add_argument(
        "-o", "--output", type=str,
        help="Output path for multi-region STL"
    )
    parser.add_argument(
        "--toposet-dir", type=str,
        help="Also write topoSetDict/createPatchDict to this directory"
    )
    parser.add_argument(
        "--no-scale", action="store_true",
        help="Don't scale to metres (keep original mm units)"
    )
    parser.add_argument(
        "--base-depth", type=float, default=0.12,
        help="Base depth as fraction of LV length (default: 0.12)"
    )
    parser.add_argument(
        "--inlet-angle", type=float, default=220.0,
        help="Angular span for inlet/MV in degrees (default: 220)"
    )

    args = parser.parse_args()

    if args.analyze:
        info = analyze_stl(args.analyze)
        print(f"\n=== LV Geometry Analysis ===")
        print(f"  File: {info['file']}")
        print(f"  Faces: {info['n_faces']:,}")
        print(f"  Vertices: {info['n_vertices']:,}")
        print(f"  Bounds (mm):")
        print(f"    min: [{info['bounds_min'][0]:.1f}, {info['bounds_min'][1]:.1f}, {info['bounds_min'][2]:.1f}]")
        print(f"    max: [{info['bounds_max'][0]:.1f}, {info['bounds_max'][1]:.1f}, {info['bounds_max'][2]:.1f}]")
        print(f"  Base center (mm): [{info['base_center'][0]:.1f}, {info['base_center'][1]:.1f}, {info['base_center'][2]:.1f}]")
        print(f"  Base normal: [{info['base_normal'][0]:.3f}, {info['base_normal'][1]:.3f}, {info['base_normal'][2]:.3f}]")
        print(f"  Long axis: [{info['long_axis'][0]:.3f}, {info['long_axis'][1]:.3f}, {info['long_axis'][2]:.3f}]")
        print(f"  Face classification:")
        print(f"    Inlet (MV):  {info['inlet_faces']:,} faces")
        print(f"    Outlet (AV): {info['outlet_faces']:,} faces")
        print(f"    Wall:        {info['wall_faces']:,} faces")
        return

    if args.stl:
        if not args.output:
            parser.error("--output is required with --stl")

        result = prepare_valve_stl(
            stl_path=args.stl,
            output_path=args.output,
            scale_to_metres=not args.no_scale,
            base_depth_frac=args.base_depth,
            inlet_angle_range=args.inlet_angle,
        )

        if args.toposet_dir:
            from twin_core.cfd_pipeline.cut_valve_openings import (
                generate_toposet_dict,
                generate_create_patch_dict,
            )
            base_center = np.array(result["base_center_mm"])
            base_normal = np.array(result["base_normal"])

            toposet_dir = Path(args.toposet_dir)
            toposet_dir.mkdir(parents=True, exist_ok=True)

            toposet_path = toposet_dir / "topoSetDict"
            with open(toposet_path, "w") as f:
                mesh = trimesh.load(str(args.stl), force="mesh")
                extent = mesh.extents.max()
                f.write(generate_toposet_dict(
                    base_center, base_normal, depth=extent * 0.12
                ))
            print(f"  Wrote: {toposet_path}")

            createpatch_path = toposet_dir / "createPatchDict"
            with open(createpatch_path, "w") as f:
                f.write(generate_create_patch_dict())
            print(f"  Wrote: {createpatch_path}")


if __name__ == "__main__":
    main()
