"""
Validate and prepare STL meshes for CFD simulation.

Takes STL files from predictions_to_stl.py and ensures they are watertight,
properly oriented, and smooth enough for OpenFOAM meshing.

Usage:
    # Check quality of a single STL:
    python -m twin_core.cfd_pipeline.prepare_cfd_mesh \
        --check meshes/acdc_smoke/patient006_frame01/LV.stl

    # Repair and prepare for CFD:
    python -m twin_core.cfd_pipeline.prepare_cfd_mesh \
        --repair meshes/acdc_smoke/patient006_frame01/LV.stl \
        -o cfd_meshes/patient006_frame01/LV.stl

    # Process all STLs in a folder:
    python -m twin_core.cfd_pipeline.prepare_cfd_mesh \
        --repair-all meshes/acdc_smoke/patient006_frame01 \
        -o cfd_meshes/patient006_frame01
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import trimesh


def check_mesh_quality(stl_path: str) -> Dict:
    """Analyze an STL mesh and return a quality report for CFD suitability.

    Returns a dict with:
        - watertight: bool
        - face_count: int
        - vertex_count: int
        - surface_area: float (mm²)
        - volume: float (mm³, only if watertight)
        - euler_number: int (should be 2 for a closed surface)
        - degenerate_faces: int
        - duplicate_faces: int
        - bounds: dict with min/max coordinates
        - cfd_ready: bool (overall assessment)
        - issues: list of strings describing problems
    """
    mesh = trimesh.load(str(stl_path), force="mesh")
    issues = []

    # Basic counts
    face_count = mesh.faces.shape[0]
    vertex_count = mesh.vertices.shape[0]

    # Watertightness
    watertight = bool(mesh.is_watertight)
    if not watertight:
        issues.append("Mesh is not watertight (has holes or non-manifold edges)")

    # Euler number (should be 2 for a closed genus-0 surface)
    euler = mesh.euler_number
    if euler != 2:
        issues.append(f"Euler number is {euler} (expected 2 for closed surface)")

    # Volume (only meaningful if watertight)
    volume = float(mesh.volume) if watertight else None

    # Surface area
    surface_area = float(mesh.area)

    # Degenerate faces (zero area)
    face_areas = mesh.area_faces
    degenerate = int(np.sum(face_areas < 1e-10))
    if degenerate > 0:
        issues.append(f"{degenerate} degenerate (zero-area) faces")

    # Duplicate faces
    try:
        unique_faces = set(map(tuple, np.sort(mesh.faces, axis=1).tolist()))
        duplicate = face_count - len(unique_faces)
    except Exception:
        duplicate = 0
    if duplicate > 0:
        issues.append(f"{duplicate} duplicate faces")

    # Face aspect ratios (detect very thin triangles)
    edge_lengths = mesh.edges_unique_length
    if len(edge_lengths) > 0:
        aspect_ratio_max = float(edge_lengths.max() / max(edge_lengths.min(), 1e-10))
        if aspect_ratio_max > 100:
            issues.append(f"Poor aspect ratio (max edge ratio: {aspect_ratio_max:.1f})")
    else:
        aspect_ratio_max = None

    # Bounding box
    bounds_min = mesh.bounds[0].tolist()
    bounds_max = mesh.bounds[1].tolist()
    extents = mesh.extents.tolist()

    # Minimum face count for CFD
    if face_count < 1000:
        issues.append(f"Low face count ({face_count}); may need refinement for CFD")

    cfd_ready = watertight and degenerate == 0 and duplicate == 0

    return {
        "file": str(stl_path),
        "watertight": watertight,
        "face_count": face_count,
        "vertex_count": vertex_count,
        "surface_area_mm2": round(surface_area, 2),
        "volume_mm3": round(volume, 2) if volume is not None else None,
        "euler_number": euler,
        "degenerate_faces": degenerate,
        "duplicate_faces": duplicate,
        "aspect_ratio_max": round(aspect_ratio_max, 1) if aspect_ratio_max else None,
        "bounds_min_mm": [round(v, 2) for v in bounds_min],
        "bounds_max_mm": [round(v, 2) for v in bounds_max],
        "extents_mm": [round(v, 2) for v in extents],
        "cfd_ready": cfd_ready,
        "issues": issues,
    }


def repair_for_cfd(
    stl_path: str,
    output_path: str,
    smoothing_iterations: int = 10,
    smoothing_method: str = "taubin",
    fill_holes: bool = True,
    target_faces: Optional[int] = None,
) -> Dict:
    """Repair and smooth an STL mesh for CFD use.

    Steps:
    1. Remove degenerate and duplicate faces
    2. Fix normals (consistent winding)
    3. Fill holes (if enabled)
    4. Apply pymeshfix repair (if available)
    5. Smooth surface (Taubin or Laplacian)
    6. Optionally decimate to target face count

    Returns quality report of the repaired mesh.
    """
    mesh = trimesh.load(str(stl_path), force="mesh")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Remove degenerate and duplicate faces
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()

    # Step 2: Fix normals
    mesh.fix_normals()

    # Step 3: Fill holes
    if fill_holes:
        trimesh.repair.fill_holes(mesh)

    # Step 4: pymeshfix repair (stronger watertight enforcement)
    try:
        import pymeshfix
        mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        mf.repair()
        mesh = trimesh.Trimesh(vertices=mf.v, faces=mf.f, process=False)
    except ImportError:
        pass
    except Exception:
        pass

    # Step 5: Smooth
    if smoothing_iterations > 0:
        if smoothing_method == "taubin" and hasattr(trimesh.smoothing, "filter_taubin"):
            trimesh.smoothing.filter_taubin(mesh, iterations=smoothing_iterations)
        elif hasattr(trimesh.smoothing, "filter_laplacian"):
            trimesh.smoothing.filter_laplacian(
                mesh, lamb=0.5, iterations=smoothing_iterations
            )

    # Step 6: Decimate if requested
    if target_faces is not None and hasattr(mesh, "simplify_quadratic_decimation"):
        simplified = mesh.simplify_quadratic_decimation(target_faces)
        if isinstance(simplified, trimesh.Trimesh) and simplified.faces.shape[0] > 0:
            mesh = simplified

    # Final cleanup
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    # Export
    mesh.export(str(output_path))

    # Return quality report of repaired mesh
    return check_mesh_quality(str(output_path))


def compute_surface_area(stl_path: str) -> float:
    """Compute the surface area of an STL mesh in mm²."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    return float(mesh.area)


def compute_bounding_box(stl_path: str) -> Dict:
    """Compute bounding box of an STL mesh. Needed for OpenFOAM blockMeshDict."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    return {
        "min": mesh.bounds[0].tolist(),
        "max": mesh.bounds[1].tolist(),
        "extents": mesh.extents.tolist(),
        "center": mesh.centroid.tolist(),
    }


def print_quality_report(report: Dict):
    """Pretty-print a mesh quality report."""
    print(f"\n=== Mesh Quality Report ===")
    print(f"  File: {report['file']}")
    print(f"  Faces: {report['face_count']:,}")
    print(f"  Vertices: {report['vertex_count']:,}")
    print(f"  Surface area: {report['surface_area_mm2']:,.1f} mm²")
    if report["volume_mm3"] is not None:
        print(f"  Volume: {report['volume_mm3']:,.1f} mm³")
    print(f"  Watertight: {report['watertight']}")
    print(f"  Euler number: {report['euler_number']}")
    print(f"  Degenerate faces: {report['degenerate_faces']}")
    print(f"  Duplicate faces: {report['duplicate_faces']}")
    if report["aspect_ratio_max"]:
        print(f"  Max aspect ratio: {report['aspect_ratio_max']}")
    print(f"  Extents (mm): {report['extents_mm']}")
    print(f"  CFD ready: {report['cfd_ready']}")
    if report["issues"]:
        print(f"  Issues:")
        for issue in report["issues"]:
            print(f"    - {issue}")
    else:
        print(f"  No issues found.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Validate and prepare STL meshes for CFD"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check", type=str,
        help="Check quality of a single STL file"
    )
    group.add_argument(
        "--repair", type=str,
        help="Repair a single STL file for CFD"
    )
    group.add_argument(
        "--repair-all", type=str,
        help="Repair all STL files in a directory"
    )
    parser.add_argument(
        "-o", "--output", type=str,
        help="Output path (file for --repair, directory for --repair-all)"
    )
    parser.add_argument(
        "--smoothing", type=int, default=10,
        help="Smoothing iterations (default: 10)"
    )
    parser.add_argument(
        "--method", type=str, default="taubin",
        choices=["taubin", "laplacian"],
        help="Smoothing method (default: taubin)"
    )
    parser.add_argument(
        "--target-faces", type=int, default=None,
        help="Target face count after decimation"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output quality report as JSON"
    )

    args = parser.parse_args()

    if args.check:
        report = check_mesh_quality(args.check)
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_quality_report(report)

    elif args.repair:
        if not args.output:
            print("Error: --output is required with --repair")
            return
        print(f"Repairing {args.repair}...")
        report = repair_for_cfd(
            stl_path=args.repair,
            output_path=args.output,
            smoothing_iterations=args.smoothing,
            smoothing_method=args.method,
            target_faces=args.target_faces,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print_quality_report(report)

    elif args.repair_all:
        input_dir = Path(args.repair_all)
        if not args.output:
            print("Error: --output is required with --repair-all")
            return
        output_dir = Path(args.output)
        stl_files = sorted(input_dir.glob("*.stl"))
        if not stl_files:
            print(f"No STL files found in {input_dir}")
            return

        print(f"Found {len(stl_files)} STL files in {input_dir}")
        reports = []
        for stl_file in stl_files:
            out_path = output_dir / stl_file.name
            print(f"Repairing {stl_file.name}...")
            report = repair_for_cfd(
                stl_path=str(stl_file),
                output_path=str(out_path),
                smoothing_iterations=args.smoothing,
                smoothing_method=args.method,
                target_faces=args.target_faces,
            )
            reports.append(report)
            if not args.json:
                print_quality_report(report)

        if args.json:
            print(json.dumps(reports, indent=2))

        ready = sum(1 for r in reports if r["cfd_ready"])
        print(f"\n{ready}/{len(reports)} meshes are CFD-ready.")


if __name__ == "__main__":
    main()
