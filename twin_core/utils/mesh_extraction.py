"""
mesh_extraction.py

Robust mask -> mesh utilities for cardiac segmentation pipelines.

Features
- Accepts either a labeled volume (integer labels) or a binary mask.
- Optional `class_id` to extract a specific label from a labeled volume.
- 3D cleaning: hole filling, largest-component filtering, small-object removal.
- Marching cubes with correct spacing handling.
- Postprocessing: normal fixing, Laplacian/Taubin smoothing (if available), optional decimation.
- Best-effort mesh repair using trimesh and optional pymeshfix if installed.
- Backwards-compatible API: mask_to_mesh(mask, spacing, out_path, ...)
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from skimage import measure
import scipy.ndimage as ndi
import trimesh
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _ensure_binary(mask: np.ndarray) -> np.ndarray:
    """Return a binary (0/1) uint8 array from mask-like input."""
    return (mask > 0).astype(np.uint8)


def _extract_class(mask: np.ndarray, class_id: Optional[int]) -> np.ndarray:
    """
    If mask is labeled and class_id is provided, return binary mask for that class.
    If class_id is None, assume mask is already binary or labeled and convert to binary.
    """
    if class_id is None:
        # If mask has multiple labels, treat non-zero as foreground
        return _ensure_binary(mask)
    else:
        return (mask == class_id).astype(np.uint8)


def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a 3D binary mask."""
    labeled, n = ndi.label(binary)
    if n == 0:
        return binary
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # background
    largest_label = counts.argmax()
    return (labeled == largest_label).astype(np.uint8)


def _remove_small_objects(binary: np.ndarray, min_size: int = 64) -> np.ndarray:
    """Remove connected components smaller than min_size voxels."""
    labeled, n = ndi.label(binary)
    if n == 0:
        return binary
    counts = np.bincount(labeled.ravel())
    mask = np.zeros_like(binary, dtype=np.uint8)
    for lab in range(1, len(counts)):
        if counts[lab] >= min_size:
            mask[labeled == lab] = 1
    return mask


def _fill_holes_3d(binary: np.ndarray) -> np.ndarray:
    """Fill holes in 3D binary mask using binary_fill_holes slice-wise then 3D closing."""
    # 3D hole fill
    filled = ndi.binary_fill_holes(binary).astype(np.uint8)
    # small morphological closing to smooth small cavities
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    closed = ndi.binary_closing(filled, structure=structure)
    return closed.astype(np.uint8)


def _marching_cubes_to_trimesh(binary: np.ndarray, spacing: Tuple[float, float, float]) -> trimesh.Trimesh:
    """
    Run marching_cubes on a binary volume and return a trimesh.Trimesh.
    - binary: (Z, Y, X) boolean/uint8
    - spacing: (dz, dy, dx) in physical units (e.g., mm)
    """
    # skimage.measure.marching_cubes expects input in (Z, Y, X) and spacing per axis
    verts, faces, normals, _ = measure.marching_cubes(binary.astype(np.float32), level=0.5, spacing=spacing)
    if verts.size == 0 or faces.size == 0:
        raise ValueError("Marching cubes produced empty mesh (no surface found).")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False)
    return mesh


def _repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Best-effort mesh repair:
    - remove degenerate faces
    - fix normals
    - fill small holes (trimesh)
    - attempt pymeshfix if available for more robust repair
    """
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # Ensure consistent winding and normals
    try:
        mesh.fix_normals()
    except Exception:
        pass

    # Try trimesh's fill holes (best-effort)
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    # Optional: use pymeshfix if installed for stronger repair
    try:
        import pymeshfix
        mf = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        mf.repair()
        verts_fixed, faces_fixed = mf.v, mf.f
        mesh = trimesh.Trimesh(vertices=verts_fixed, faces=faces_fixed, process=False)
    except Exception:
        # pymeshfix not available or repair failed; continue with trimesh mesh
        pass

    # Final cleanup
    try:
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    return mesh


def _smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 5, method: str = "taubin") -> trimesh.Trimesh:
    """
    Apply smoothing to the mesh. Prefer Taubin if available, otherwise Laplacian.
    - iterations: number of smoothing iterations
    - method: 'taubin' or 'laplacian'
    """
    try:
        if method == "taubin" and hasattr(trimesh.smoothing, "filter_taubin"):
            trimesh.smoothing.filter_taubin(mesh, iterations=iterations)
        elif method == "laplacian" and hasattr(trimesh.smoothing, "filter_laplacian"):
            trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)
        else:
            # fallback to laplacian if taubin not available
            if hasattr(trimesh.smoothing, "filter_laplacian"):
                trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)
    except Exception:
        # smoothing failed; return original mesh
        pass
    return mesh


def _decimate_mesh(mesh: trimesh.Trimesh, target_faces: Optional[int] = None, reduction_ratio: Optional[float] = None) -> trimesh.Trimesh:
    """
    Attempt to decimate the mesh to reduce face count.
    - target_faces: desired number of faces (preferred)
    - reduction_ratio: fraction to keep (0.0-1.0)
    Uses trimesh.simplify_quadratic_decimation if available (requires external dependencies).
    """
    try:
        if target_faces is None and reduction_ratio is not None:
            target_faces = int(mesh.faces.shape[0] * float(reduction_ratio))
        if target_faces is None:
            return mesh

        if hasattr(mesh, "simplify_quadratic_decimation"):
            simplified = mesh.simplify_quadratic_decimation(target_faces)
            if isinstance(simplified, trimesh.Trimesh) and simplified.faces.shape[0] > 0:
                return simplified
    except Exception:
        pass
    return mesh


def mask_to_mesh(
    mask: Union[np.ndarray, 'np.ndarray'],
    spacing: Tuple[float, float, float],
    out_path: Union[str, Path],
    class_id: Optional[int] = None,
    postprocess: bool = True,
    min_component_size: int = 64,
    smoothing_iterations: int = 5,
    smoothing_method: str = "taubin",
    decimate_target_faces: Optional[int] = None,
    decimate_ratio: Optional[float] = None,
    overwrite: bool = True,
) -> Optional[trimesh.Trimesh]:
    """
    Convert a labeled volume or binary mask to an STL mesh and save it.

    Parameters
    - mask: np.ndarray, shape (Z, Y, X) or (Y, X) for single-slice. Can be integer-labeled.
    - spacing: tuple (dz, dy, dx) in physical units (e.g., mm). Must match mask axes.
    - out_path: path to write the mesh (STL/PLY/OBJ supported by trimesh).
    - class_id: if provided, extract mask == class_id. If None, treat non-zero as foreground.
    - postprocess: apply cleaning (hole fill, keep largest component, remove small objects).
    - min_component_size: minimum voxel count to keep a connected component.
    - smoothing_iterations: number of smoothing iterations to apply.
    - smoothing_method: 'taubin' or 'laplacian'.
    - decimate_target_faces: target face count after decimation (preferred).
    - decimate_ratio: alternative to target_faces; fraction of faces to keep.
    - overwrite: if False and out_path exists, skip writing.

    Returns:
    - trimesh.Trimesh object on success, None if mesh could not be produced.
    """
    out_path = Path(out_path)

    # Prepare binary mask for the requested class
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 2:
        # single slice -> convert to a 3D volume with Z=1 for marching cubes compatibility
        mask_vol = mask_arr[np.newaxis, ...]
    elif mask_arr.ndim == 3:
        mask_vol = mask_arr
    else:
        raise ValueError(f"Unsupported mask ndim {mask_arr.ndim}; expected 2 or 3")

    binary = _extract_class(mask_vol, class_id)

    if postprocess:
        # Fill holes and remove tiny objects
        binary = _fill_holes_3d(binary)
        binary = _remove_small_objects(binary, min_size=min_component_size)
        binary = _keep_largest_component(binary)

    # If after cleaning nothing remains, abort gracefully
    if binary.sum() == 0:
        logger.warning("mask_to_mesh: empty binary mask after postprocessing; skipping mesh export.")
        return None

    # Run marching cubes -> trimesh
    try:
        mesh = _marching_cubes_to_trimesh(binary, spacing)
    except ValueError as e:
        logger.warning(f"mask_to_mesh: marching cubes failed: {e}")
        return None
    except Exception as e:
        logger.exception("mask_to_mesh: unexpected error during marching cubes")
        return None

    # Repair mesh
    mesh = _repair_mesh(mesh)

    # Smooth mesh
    if smoothing_iterations and smoothing_iterations > 0:
        mesh = _smooth_mesh(mesh, iterations=smoothing_iterations, method=smoothing_method)

    # Decimate if requested
    if decimate_target_faces is not None or decimate_ratio is not None:
        mesh = _decimate_mesh(mesh, target_faces=decimate_target_faces, reduction_ratio=decimate_ratio)

    # Final repair/cleanup
    mesh = _repair_mesh(mesh)

    # Export
    try:
        if out_path.exists() and not overwrite:
            logger.info(f"mask_to_mesh: {out_path} exists and overwrite=False; skipping write.")
        else:
            mesh.export(str(out_path))
            logger.info(f"mask_to_mesh: exported mesh to {out_path}")
    except Exception:
        logger.exception(f"mask_to_mesh: failed to export mesh to {out_path}")
        # still return mesh object for inspection
    return mesh
