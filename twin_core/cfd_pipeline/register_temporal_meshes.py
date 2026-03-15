"""
Temporal mesh registration for moving-mesh cardiac CFD.

Creates topologically consistent meshes across cardiac frames by:
1. Selecting a reference frame (ED) as the template mesh
2. Using Coherent Point Drift (CPD) to find a smooth deformation that
   maps the template surface vertices to each target frame's surface
3. Applying the deformation to produce meshes with identical topology

This produces N meshes with IDENTICAL vertex count and connectivity
but DIFFERENT vertex positions — exactly what OpenFOAM's
dynamicMeshDict / displacementInterpolation needs.

Method:
    Coherent Point Drift (CPD) — non-rigid point set registration
    - Operates directly on mesh surface vertices (no volumetric registration)
    - Finds smooth, coherent deformation field between point sets
    - Regularized to prevent folding/tearing
    - Well-suited for cardiac motion tracking

References:
    - Myronenko A, Song X. "Point Set Registration: Coherent Point Drift."
      IEEE TPAMI 32(12):2262-2275, 2010. doi:10.1109/TPAMI.2010.46
    - Sotiras A, Davatzikos C, Paragios N. "Deformable Medical Image
      Registration: A Survey." IEEE TMI 32(7):1153-1190, 2013.

Usage:
    python -m twin_core.cfd_pipeline.register_temporal_meshes \
        --predictions-meshes ~/digital_twin_project/test_3d/meshes \
        --template-mesh ~/digital_twin_project/test_3d/meshes/patient006_frame01/LV.stl \
        --template-frame patient006_frame01 \
        --output ~/digital_twin_project/test_3d/registered_meshes \
        --structure LV

    # Verify consistency:
    python -m twin_core.cfd_pipeline.register_temporal_meshes \
        --verify ~/digital_twin_project/test_3d/registered_meshes

Dependencies:
    pip install pycpd trimesh numpy scipy
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from pycpd import DeformableRegistration
except ImportError:
    DeformableRegistration = None
    print("WARNING: pycpd not installed. Install with: pip install pycpd")

try:
    import trimesh
except ImportError:
    trimesh = None
    print("WARNING: trimesh not installed. Install with: pip install trimesh")


def subsample_points(vertices: np.ndarray, n_points: int = 2000) -> np.ndarray:
    """Subsample vertices for faster CPD registration.

    CPD is O(N*M) so we register on a subset, then interpolate.

    Returns indices of selected vertices.
    """
    if len(vertices) <= n_points:
        return np.arange(len(vertices))
    indices = np.random.choice(len(vertices), n_points, replace=False)
    return np.sort(indices)


def register_cpd(
    template_vertices: np.ndarray,
    target_vertices: np.ndarray,
    alpha: float = 2.0,
    beta: float = 2.0,
    max_iterations: int = 150,
    tolerance: float = 1e-5,
) -> np.ndarray:
    """Register template vertices to target using non-rigid CPD.

    Parameters
    ----------
    template_vertices : (N, 3) source point set
    target_vertices : (M, 3) target point set
    alpha : trade-off between fit and regularization (higher = smoother)
    beta : width of Gaussian kernel (higher = more global/rigid deformation)
    max_iterations : maximum CPD iterations
    tolerance : convergence tolerance

    Returns
    -------
    (N, 3) deformed template vertices
    """
    template_vertices = np.ascontiguousarray(template_vertices, dtype=np.float64)
    target_vertices = np.ascontiguousarray(target_vertices, dtype=np.float64)
    reg = DeformableRegistration(
        X=target_vertices,    # target (what we want to match)
        Y=template_vertices,  # source (what gets deformed)
        alpha=alpha,
        beta=beta,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    deformed, _ = reg.register()
    return deformed


def register_with_interpolation(
    template_vertices: np.ndarray,
    target_vertices: np.ndarray,
    n_subsample: int = 2000,
    alpha: float = 2.0,
    beta: float = 2.0,
    max_iterations: int = 150,
) -> np.ndarray:
    """Register using CPD on subsampled points, then interpolate to full mesh.

    For large meshes, running CPD on all vertices is slow. Instead:
    1. Subsample both template and target
    2. Run CPD on subsampled sets to get displacement field
    3. Interpolate displacements to all template vertices using RBF

    Parameters
    ----------
    template_vertices : (N, 3) all template vertices
    target_vertices : (M, 3) all target surface vertices
    n_subsample : number of points for CPD registration
    alpha, beta : CPD parameters

    Returns
    -------
    (N, 3) deformed positions for ALL template vertices
    """
    from scipy.interpolate import RBFInterpolator

    n_template = len(template_vertices)
    n_target = len(target_vertices)

    # If small enough, just run CPD directly
    if n_template <= n_subsample:
        return register_cpd(template_vertices, target_vertices,
                            alpha=alpha, beta=beta, max_iterations=max_iterations)

    # Subsample template
    sub_idx = subsample_points(template_vertices, n_subsample)
    sub_template = template_vertices[sub_idx]

    # Subsample target (for speed)
    target_sub_idx = subsample_points(target_vertices, n_subsample)
    sub_target = target_vertices[target_sub_idx]

    # Run CPD on subsampled points
    sub_deformed = register_cpd(sub_template, sub_target,
                                alpha=alpha, beta=beta,
                                max_iterations=max_iterations)

    # Compute displacements at subsampled points
    sub_displacements = sub_deformed - sub_template

    # Interpolate displacements to all template vertices using RBF
    # Using thin-plate spline kernel for smooth interpolation
    rbf_x = RBFInterpolator(sub_template, sub_displacements[:, 0],
                            kernel='thin_plate_spline', smoothing=1.0)
    rbf_y = RBFInterpolator(sub_template, sub_displacements[:, 1],
                            kernel='thin_plate_spline', smoothing=1.0)
    rbf_z = RBFInterpolator(sub_template, sub_displacements[:, 2],
                            kernel='thin_plate_spline', smoothing=1.0)

    full_displacements = np.column_stack([
        rbf_x(template_vertices),
        rbf_y(template_vertices),
        rbf_z(template_vertices),
    ])

    return template_vertices + full_displacements


def register_all_frames(
    meshes_dir: str,
    template_mesh_path: str,
    template_frame: str,
    output_dir: str,
    structure_name: str = "LV",
    n_subsample: int = 2000,
    alpha: float = 2.0,
    beta: float = 2.0,
):
    """Register all frames to the template using CPD.

    Parameters
    ----------
    meshes_dir : directory containing per-frame mesh subdirectories
    template_mesh_path : path to template STL mesh
    template_frame : name of template frame (e.g., 'patient006_frame01')
    output_dir : directory for output registered meshes
    structure_name : STL filename stem (e.g., 'LV')
    n_subsample : number of points for CPD (more = slower but more accurate)
    alpha : CPD regularization (higher = smoother deformation)
    beta : CPD kernel width (higher = more rigid/global deformation)
    """
    meshes_dir = Path(meshes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Temporal Mesh Registration (CPD) ===")
    print(f"  Template frame: {template_frame}")
    print(f"  Template mesh: {template_mesh_path}")
    print(f"  Structure: {structure_name}")
    print(f"  CPD params: alpha={alpha}, beta={beta}, subsample={n_subsample}")
    print(f"  Output: {output_dir}")

    # Load template mesh
    print(f"\n  Loading template mesh...")
    template_mesh = trimesh.load(str(template_mesh_path), force="mesh")
    template_vertices = template_mesh.vertices.copy()
    template_faces = template_mesh.faces.copy()

    n_verts = len(template_vertices)
    n_faces = len(template_faces)
    print(f"  Template: {n_verts:,} vertices, {n_faces:,} faces")
    print(f"  Template volume: {template_mesh.volume:.0f} mm3")

    # Find all frame directories
    frame_dirs = sorted(meshes_dir.glob("patient006_frame*"))
    print(f"  Found {len(frame_dirs)} frames to process\n")

    metadata = {
        "template_frame": template_frame,
        "template_mesh": str(template_mesh_path),
        "structure": structure_name,
        "method": "CPD (non-rigid)",
        "cpd_alpha": alpha,
        "cpd_beta": beta,
        "n_subsample": n_subsample,
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "frames": {},
    }

    for frame_dir in frame_dirs:
        frame_name = frame_dir.name
        target_stl = frame_dir / f"{structure_name}.stl"

        if not target_stl.exists():
            print(f"  {frame_name}: SKIPPED (no {structure_name}.stl)")
            continue

        out_dir = output_dir / frame_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_stl = out_dir / f"{structure_name}.stl"

        if frame_name == template_frame:
            print(f"  {frame_name}: template (copy)")
            template_mesh.export(str(out_stl))
            metadata["frames"][frame_name] = {
                "type": "template",
                "max_displacement_mm": 0.0,
                "volume_mm3": float(template_mesh.volume),
            }
            continue

        print(f"  {frame_name}: ", end="", flush=True)

        try:
            # Load target mesh
            target_mesh = trimesh.load(str(target_stl), force="mesh")
            target_vertices = target_mesh.vertices.copy()

            print(f"registering ({len(target_vertices)} target pts)...", end=" ", flush=True)

            # Run CPD registration
            deformed_vertices = register_with_interpolation(
                template_vertices,
                target_vertices,
                n_subsample=n_subsample,
                alpha=alpha,
                beta=beta,
            )

            # Statistics
            displacements = np.linalg.norm(deformed_vertices - template_vertices, axis=1)
            max_disp = float(displacements.max())
            mean_disp = float(displacements.mean())

            # Create output mesh with same topology
            deformed_mesh = trimesh.Trimesh(
                vertices=deformed_vertices,
                faces=template_faces,
                process=False,
            )
            deformed_mesh.export(str(out_stl))

            # Volume comparison
            target_vol = float(target_mesh.volume)
            reg_vol = float(deformed_mesh.volume)
            vol_err = abs(reg_vol - target_vol) / target_vol * 100

            print(f"done | max={max_disp:.1f}mm mean={mean_disp:.1f}mm | "
                  f"vol: target={target_vol:.0f} reg={reg_vol:.0f} err={vol_err:.1f}%")

            metadata["frames"][frame_name] = {
                "type": "registered",
                "max_displacement_mm": max_disp,
                "mean_displacement_mm": mean_disp,
                "target_volume_mm3": target_vol,
                "registered_volume_mm3": reg_vol,
                "volume_error_pct": vol_err,
            }

        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
            metadata["frames"][frame_name] = {
                "type": "failed",
                "error": str(e),
            }

    # Save metadata
    meta_path = output_dir / "registration_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n=== Registration Summary ===")
    registered = {k: v for k, v in metadata["frames"].items()
                  if v.get("type") == "registered"}
    if registered:
        disps = [v["max_displacement_mm"] for v in registered.values()]
        vol_errs = [v["volume_error_pct"] for v in registered.values()]
        print(f"  Max displacement range: {min(disps):.1f} — {max(disps):.1f} mm")
        print(f"  Mean max displacement: {np.mean(disps):.1f} mm")
        print(f"  Volume error: mean={np.mean(vol_errs):.1f}%, max={np.max(vol_errs):.1f}%")

    failed = sum(1 for v in metadata["frames"].values() if v.get("type") == "failed")
    if failed:
        print(f"  WARNING: {failed} frames failed")

    print(f"  Metadata saved: {meta_path}")
    print(f"  Done.")

    return metadata


def verify_consistency(registered_dir: str):
    """Verify that all registered meshes have consistent topology."""
    registered_dir = Path(registered_dir)

    print(f"\n=== Verifying Mesh Consistency ===")
    print(f"  Directory: {registered_dir}")

    stl_files = sorted(registered_dir.rglob("LV.stl"))
    if not stl_files:
        print("  No LV.stl files found!")
        return

    reference = trimesh.load(str(stl_files[0]), force="mesh")
    ref_nverts = len(reference.vertices)
    ref_nfaces = len(reference.faces)
    ref_faces = reference.faces.copy()

    print(f"  Reference: {stl_files[0].parent.name}")
    print(f"    Vertices: {ref_nverts:,}")
    print(f"    Faces: {ref_nfaces:,}")
    print()

    all_ok = True
    for stl_file in stl_files:
        frame_name = stl_file.parent.name
        mesh = trimesh.load(str(stl_file), force="mesh")

        nverts = len(mesh.vertices)
        nfaces = len(mesh.faces)
        faces_match = np.array_equal(mesh.faces, ref_faces)

        areas = mesh.area_faces
        n_degenerate = int((areas < 1e-12).sum())

        status = "OK" if (nverts == ref_nverts and nfaces == ref_nfaces and faces_match) else "MISMATCH"
        if n_degenerate > 0:
            status = "DEGENERATE"
        if status != "OK":
            all_ok = False

        print(f"  {frame_name}: V={nverts:,} F={nfaces:,} "
              f"faces_match={faces_match} degenerate={n_degenerate} [{status}]")

    print()
    if all_ok:
        print("  ALL FRAMES CONSISTENT — ready for dynamic mesh CFD")
    else:
        print("  WARNING: some frames have inconsistent topology")


def main():
    parser = argparse.ArgumentParser(
        description="Register cardiac meshes across temporal frames for dynamic CFD"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--predictions-meshes", type=str,
        help="Directory containing per-frame mesh subdirectories (each with LV.stl)"
    )
    group.add_argument(
        "--verify", type=str,
        help="Verify consistency of registered meshes in this directory"
    )

    parser.add_argument(
        "--template-mesh", type=str,
        help="Path to template STL mesh (required with --predictions-meshes)"
    )
    parser.add_argument(
        "--template-frame", type=str, default="patient006_frame01",
        help="Template frame name (default: patient006_frame01)"
    )
    parser.add_argument(
        "-o", "--output", type=str,
        help="Output directory for registered meshes"
    )
    parser.add_argument(
        "--structure", type=str, default="LV",
        help="Structure name / STL filename (default: LV)"
    )
    parser.add_argument(
        "--subsample", type=int, default=2000,
        help="Number of points for CPD registration (default: 2000)"
    )
    parser.add_argument(
        "--alpha", type=float, default=2.0,
        help="CPD regularization parameter (default: 2.0)"
    )
    parser.add_argument(
        "--beta", type=float, default=2.0,
        help="CPD kernel width parameter (default: 2.0)"
    )

    args = parser.parse_args()

    if args.verify:
        verify_consistency(args.verify)
        return

    if not args.template_mesh or not args.output:
        parser.error("--template-mesh and --output are required with --predictions-meshes")

    register_all_frames(
        meshes_dir=args.predictions_meshes,
        template_mesh_path=args.template_mesh,
        template_frame=args.template_frame,
        output_dir=args.output,
        structure_name=args.structure,
        n_subsample=args.subsample,
        alpha=args.alpha,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
