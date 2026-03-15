"""
Temporal mesh registration for moving-mesh cardiac CFD.

Creates topologically consistent meshes across cardiac frames by:
1. Selecting a reference frame (ED) as the template
2. Registering each frame's segmentation volume to the reference using
   diffeomorphic (B-spline SyN) image registration
3. Warping the template surface mesh vertices using the resulting
   displacement fields

This produces N meshes with IDENTICAL vertex count and connectivity
but DIFFERENT vertex positions — exactly what OpenFOAM's
dynamicMeshDict / displacementInterpolation needs.

Method:
    Diffeomorphic image registration (SimpleITK)
    - Uses the segmentation label maps directly
    - B-spline or demons registration to find displacement field
    - Displacement field applied to template mesh vertices via interpolation

References:
    - Sotiras A, Davatzikos C, Paragios N. "Deformable Medical Image
      Registration: A Survey." IEEE TMI 32(7):1153-1190, 2013.
    - Tobon-Gomez C et al. "Benchmarking framework for myocardial tracking
      and deformation algorithms." Med Image Anal 17(6):632-648, 2013.

Usage:
    python register_temporal_meshes.py \
        --predictions ~/digital_twin_project/test_3d/predictions \
        --template-mesh ~/digital_twin_project/test_3d/meshes/patient006_frame01/LV.stl \
        --template-frame patient006_frame01 \
        --output ~/digital_twin_project/test_3d/registered_meshes \
        --label 3

    # Verify consistency:
    python register_temporal_meshes.py \
        --verify ~/digital_twin_project/test_3d/registered_meshes

Dependencies (install on VM):
    pip install SimpleITK nibabel trimesh numpy scipy
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None
    print("WARNING: SimpleITK not installed. Install with: pip install SimpleITK")

try:
    import nibabel as nib
except ImportError:
    nib = None
    print("WARNING: nibabel not installed. Install with: pip install nibabel")

try:
    import trimesh
except ImportError:
    trimesh = None
    print("WARNING: trimesh not installed. Install with: pip install trimesh")

from scipy.ndimage import map_coordinates


def load_segmentation_as_sitk(nifti_path: str, label: int) -> "sitk.Image":
    """Load a NIfTI segmentation and create a binary mask for the given label.

    Parameters
    ----------
    nifti_path : path to .nii.gz segmentation file
    label : integer label to extract (e.g., 3 for LV in ACDC)

    Returns
    -------
    SimpleITK Image with binary mask (float32, values 0.0/1.0)
    """
    img = sitk.ReadImage(str(nifti_path))
    # Extract single label as binary mask
    binary = sitk.Equal(img, label)
    # Convert to float for registration (smoother gradients)
    binary_float = sitk.Cast(binary, sitk.sitkFloat32)
    # Light Gaussian smoothing to help registration converge
    smoothed = sitk.SmoothingRecursiveGaussian(binary_float, sigma=0.5)
    return smoothed


def register_to_template(
    fixed_image: "sitk.Image",
    moving_image: "sitk.Image",
    method: str = "bspline",
) -> "sitk.DisplacementFieldTransform":
    """Register moving_image to fixed_image and return displacement field.

    Parameters
    ----------
    fixed_image : reference (template) segmentation
    moving_image : target frame segmentation
    method : 'bspline' (faster) or 'demons' (more flexible)

    Returns
    -------
    SimpleITK DisplacementFieldTransform mapping fixed→moving space
    """
    if method == "demons":
        # Demons registration — good for label maps
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(200)
        demons.SetStandardDeviations(1.5)
        demons.SetSmoothDisplacementField(True)
        demons.SetSmoothUpdateField(True)

        displacement_field = demons.Execute(fixed_image, moving_image)
        return sitk.DisplacementFieldTransform(displacement_field)

    elif method == "bspline":
        # B-spline registration via registration framework
        registration = sitk.ImageRegistrationMethod()

        # Similarity metric — mean squares works well for binary/near-binary
        registration.SetMetricAsMeanSquares()

        # Optimizer
        registration.SetOptimizerAsLBFGSB(
            gradientConvergenceTolerance=1e-5,
            numberOfIterations=200,
            maximumNumberOfCorrections=5,
            maximumNumberOfFunctionEvaluations=1000,
            costFunctionConvergenceFactor=1e7,
        )

        # Multi-resolution strategy
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # B-spline transform
        grid_physical_spacing = [8.0, 8.0, 8.0]  # mm — control point spacing
        image_size = fixed_image.GetSize()
        image_spacing = fixed_image.GetSpacing()
        mesh_size = [
            int(round(sz * sp / gsp))
            for sz, sp, gsp in zip(image_size, image_spacing, grid_physical_spacing)
        ]
        # Ensure at least 4 control points per dimension
        mesh_size = [max(4, m) for m in mesh_size]

        tx = sitk.BSplineTransformInitializer(
            fixed_image, transformDomainMeshSize=mesh_size, order=3
        )
        registration.SetInitialTransform(tx, inPlace=True)

        # Interpolator
        registration.SetInterpolator(sitk.sitkLinear)

        # Execute
        final_transform = registration.Execute(fixed_image, moving_image)

        # Convert to displacement field for uniform handling
        displacement_filter = sitk.TransformToDisplacementFieldFilter()
        displacement_filter.SetReferenceImage(fixed_image)
        displacement_field = displacement_filter.Execute(final_transform)

        return sitk.DisplacementFieldTransform(displacement_field)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'bspline' or 'demons'.")


def warp_mesh_vertices(
    vertices_mm: np.ndarray,
    displacement_field: "sitk.Image",
) -> np.ndarray:
    """Warp mesh vertices using a SimpleITK displacement field.

    The displacement field maps template space → target space.
    Vertices are in mm (physical coordinates matching the NIfTI).

    Parameters
    ----------
    vertices_mm : (N, 3) array of vertex positions in mm
    displacement_field : SimpleITK displacement field image

    Returns
    -------
    (N, 3) array of warped vertex positions in mm
    """
    # Convert displacement field to numpy
    # Shape: (Z, Y, X, 3) — SimpleITK stores as ZYX
    disp_np = sitk.GetArrayFromImage(displacement_field)  # (Z, Y, X, 3)

    # Get image geometry
    origin = np.array(displacement_field.GetOrigin())      # (X, Y, Z)
    spacing = np.array(displacement_field.GetSpacing())     # (X, Y, Z)
    direction = np.array(displacement_field.GetDirection()).reshape(3, 3)

    # Convert physical coordinates (mm) to voxel coordinates
    # physical = origin + direction @ (voxel * spacing)
    # voxel = inv(direction) @ ((physical - origin) / spacing)
    # But for axis-aligned images (identity direction), simplifies to:
    # voxel = (physical - origin) / spacing
    inv_direction = np.linalg.inv(direction)

    warped = np.zeros_like(vertices_mm)

    for i in range(len(vertices_mm)):
        # Physical to continuous voxel index
        phys = vertices_mm[i]  # (X, Y, Z)
        voxel_xyz = inv_direction @ ((phys - origin) / spacing)

        # Interpolate displacement at this voxel location
        # disp_np is (Z, Y, X, 3), so we need ZYX indexing
        voxel_zyx = voxel_xyz[::-1]

        # Trilinear interpolation for each displacement component
        dx = map_coordinates(disp_np[..., 0], voxel_zyx.reshape(3, 1),
                             order=1, mode='nearest')[0]
        dy = map_coordinates(disp_np[..., 1], voxel_zyx.reshape(3, 1),
                             order=1, mode='nearest')[0]
        dz = map_coordinates(disp_np[..., 2], voxel_zyx.reshape(3, 1),
                             order=1, mode='nearest')[0]

        # Apply displacement (in physical coordinates)
        warped[i] = phys + np.array([dx, dy, dz])

    return warped


def register_all_frames(
    predictions_dir: str,
    template_mesh_path: str,
    template_frame: str,
    output_dir: str,
    label: int = 3,
    method: str = "demons",
    structure_name: str = "LV",
):
    """Register all frames to the template and warp the template mesh.

    Parameters
    ----------
    predictions_dir : directory containing frame_XX.nii.gz predictions
    template_mesh_path : path to template STL (from template frame)
    template_frame : name of template frame (e.g., 'patient006_frame01')
    output_dir : directory for output registered meshes
    label : segmentation label for the structure (3=LV for ACDC)
    method : registration method ('bspline' or 'demons')
    structure_name : name for output files (e.g., 'LV')
    """
    predictions_dir = Path(predictions_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template
    template_nifti = predictions_dir / f"{template_frame}.nii.gz"
    if not template_nifti.exists():
        raise FileNotFoundError(f"Template prediction not found: {template_nifti}")

    print(f"\n=== Temporal Mesh Registration ===")
    print(f"  Template frame: {template_frame}")
    print(f"  Template mesh: {template_mesh_path}")
    print(f"  Label: {label} ({structure_name})")
    print(f"  Method: {method}")
    print(f"  Output: {output_dir}")

    # Load template image and mesh
    print(f"\n  Loading template segmentation...")
    fixed_image = load_segmentation_as_sitk(str(template_nifti), label)

    print(f"  Loading template mesh...")
    template_mesh = trimesh.load(str(template_mesh_path), force="mesh")
    template_vertices = template_mesh.vertices.copy()  # in mm
    template_faces = template_mesh.faces.copy()

    n_verts = len(template_vertices)
    n_faces = len(template_faces)
    print(f"  Template: {n_verts:,} vertices, {n_faces:,} faces")

    # Find all prediction frames
    pred_files = sorted(predictions_dir.glob("patient006_frame*.nii.gz"))
    print(f"  Found {len(pred_files)} frames to process\n")

    # Store metadata
    metadata = {
        "template_frame": template_frame,
        "template_mesh": str(template_mesh_path),
        "label": label,
        "structure": structure_name,
        "method": method,
        "n_vertices": n_verts,
        "n_faces": n_faces,
        "frames": {},
    }

    for pred_file in pred_files:
        frame_name = pred_file.stem.replace(".nii", "")
        frame_dir = output_dir / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        out_stl = frame_dir / f"{structure_name}.stl"

        if frame_name == template_frame:
            # Template frame — just copy the mesh
            print(f"  {frame_name}: template (copy)")
            template_mesh.export(str(out_stl))
            metadata["frames"][frame_name] = {
                "type": "template",
                "max_displacement_mm": 0.0,
            }
            continue

        print(f"  {frame_name}: registering...", end=" ", flush=True)

        # Load target frame
        moving_image = load_segmentation_as_sitk(str(pred_file), label)

        # Register
        try:
            disp_transform = register_to_template(fixed_image, moving_image, method)

            # Get displacement field as image
            if isinstance(disp_transform, sitk.DisplacementFieldTransform):
                disp_field = disp_transform.GetDisplacementField()
            else:
                # Convert transform to displacement field
                disp_filter = sitk.TransformToDisplacementFieldFilter()
                disp_filter.SetReferenceImage(fixed_image)
                disp_field = disp_filter.Execute(disp_transform)

            # Warp template vertices
            warped_vertices = warp_mesh_vertices(template_vertices, disp_field)

            # Compute displacement statistics
            displacements = np.linalg.norm(warped_vertices - template_vertices, axis=1)
            max_disp = float(displacements.max())
            mean_disp = float(displacements.mean())

            # Create warped mesh with same topology
            warped_mesh = trimesh.Trimesh(
                vertices=warped_vertices,
                faces=template_faces,
                process=False,  # Don't modify connectivity!
            )
            warped_mesh.export(str(out_stl))

            print(f"done (max disp: {max_disp:.2f} mm, mean: {mean_disp:.2f} mm)")

            metadata["frames"][frame_name] = {
                "type": "registered",
                "max_displacement_mm": max_disp,
                "mean_displacement_mm": mean_disp,
            }

            # Optionally save displacement field for debugging
            # sitk.WriteImage(disp_field, str(frame_dir / "displacement_field.nii.gz"))

        except Exception as e:
            print(f"FAILED: {e}")
            metadata["frames"][frame_name] = {
                "type": "failed",
                "error": str(e),
            }

    # Save metadata
    meta_path = output_dir / "registration_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Metadata saved: {meta_path}")
    print(f"  Done. {len(metadata['frames'])} frames processed.")

    return metadata


def verify_consistency(registered_dir: str):
    """Verify that all registered meshes have consistent topology.

    Checks:
    - All meshes have the same vertex count
    - All meshes have the same face count
    - All meshes have identical face connectivity
    - No degenerate faces
    """
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

        # Check for degenerate faces
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
        "--predictions", type=str,
        help="Directory containing frame prediction NIfTI files"
    )
    group.add_argument(
        "--verify", type=str,
        help="Verify consistency of registered meshes in this directory"
    )

    parser.add_argument(
        "--template-mesh", type=str,
        help="Path to template STL mesh (required with --predictions)"
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
        "--label", type=int, default=3,
        help="Segmentation label for structure (default: 3 = LV for ACDC)"
    )
    parser.add_argument(
        "--method", type=str, default="demons", choices=["bspline", "demons"],
        help="Registration method (default: demons)"
    )
    parser.add_argument(
        "--structure", type=str, default="LV",
        help="Structure name for output files (default: LV)"
    )

    args = parser.parse_args()

    if args.verify:
        verify_consistency(args.verify)
        return

    if not args.template_mesh or not args.output:
        parser.error("--template-mesh and --output are required with --predictions")

    register_all_frames(
        predictions_dir=args.predictions,
        template_mesh_path=args.template_mesh,
        template_frame=args.template_frame,
        output_dir=args.output,
        label=args.label,
        method=args.method,
        structure_name=args.structure,
    )


if __name__ == "__main__":
    main()
