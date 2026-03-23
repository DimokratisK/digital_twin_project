"""
Interpolate coarse STL boundary displacement data onto fine OpenFOAM mesh face centres.

Problem: generate_dynamic_case.py writes 4,808 STL vertices as boundary data points,
but OpenFOAM mesh has ~320k wall faces. With mapMethod=nearest, each face snaps to
the closest STL vertex, creating blocks of ~66 faces with identical displacement and
sharp jumps at block edges. This causes Courant number spikes and solver divergence.

Fix: Read OpenFOAM mesh face centres for each patch, then interpolate displacements
from the coarse STL points onto the fine mesh using scipy KDTree inverse-distance
weighting. This produces a smooth displacement field.

Run on VM32:
    python3 interpolate_boundary_data.py ~/cfd_runs/dynamic_mesh/case

After running, re-copy boundaryData to processor directories:
    for i in $(seq 0 31); do
        rm -rf processor$i/constant/boundaryData
        cp -r constant/boundaryData processor$i/constant/
    done
"""

import sys
import re
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree


def parse_openfoam_points(filepath: str) -> np.ndarray:
    """Parse an OpenFOAM points file into (N, 3) array."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find the line with just the point count (a bare integer before the data block)
    n = None
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Skip empty lines, comments, FoamFile header
        if stripped.isdigit() and int(stripped) > 100:
            # Check next non-empty line is "("
            for j in range(i + 1, min(i + 3, len(lines))):
                if lines[j].strip() == "(":
                    n = int(stripped)
                    data_start = j + 1
                    break
            if n is not None:
                break

    if n is None:
        raise ValueError(f"Cannot find point count in: {filepath}")

    # Parse (x y z) lines
    points = []
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line == ")":
            break
        # Remove parentheses and parse three floats
        line = line.replace("(", "").replace(")", "")
        parts = line.split()
        if len(parts) == 3:
            points.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if len(points) != n:
        raise ValueError(f"Expected {n} points, parsed {len(points)} in {filepath}")

    return np.array(points)


def parse_boundary_data_points(filepath: str) -> np.ndarray:
    """Parse a boundaryData/patch/points file (simpler format, no header)."""
    with open(filepath, "r") as f:
        text = f.read()
    # Format: N\n(\n(x y z)\n...\n)
    lines = text.strip().split("\n")
    n = int(lines[0].strip())
    points = []
    for line in lines[2:]:  # skip N and (
        line = line.strip()
        if line == ")":
            break
        coords = line.strip("()").split()
        points.append([float(coords[0]), float(coords[1]), float(coords[2])])
    assert len(points) == n, f"Expected {n} points, got {len(points)}"
    return np.array(points)


def parse_displacement_file(filepath: str) -> np.ndarray:
    """Parse a boundaryData/patch/time/pointDisplacement file."""
    return parse_boundary_data_points(filepath)  # Same format


def get_patch_face_centres(case_dir: Path, patch_name: str) -> np.ndarray:
    """Extract face centres for a patch from OpenFOAM mesh.

    Reads constant/polyMesh/{points,faces,boundary} to compute face centres
    for the named patch.
    """
    polymesh = case_dir / "constant" / "polyMesh"

    # Parse points
    points = parse_openfoam_points(str(polymesh / "points"))

    # Parse faces
    with open(polymesh / "faces", "r") as f:
        faces_text = f.read()
    # Find N faces
    match = re.search(r"^(\d+)\s*\n\(", faces_text, re.MULTILINE)
    n_faces = int(match.group(1))
    # Parse face definitions: N(v0 v1 v2 ...)
    face_pattern = re.compile(r"(\d+)\(([^)]+)\)")
    faces = []
    for m in face_pattern.finditer(faces_text[match.start():]):
        n_verts = int(m.group(1))
        vert_ids = list(map(int, m.group(2).split()))
        assert len(vert_ids) == n_verts
        faces.append(vert_ids)
        if len(faces) == n_faces:
            break

    # Parse boundary to find patch start/size
    with open(polymesh / "boundary", "r") as f:
        boundary_text = f.read()

    # Find the patch block
    patch_pattern = re.compile(
        rf"{patch_name}\s*\{{[^}}]*nFaces\s+(\d+);\s*startFace\s+(\d+);",
        re.DOTALL,
    )
    match = patch_pattern.search(boundary_text)
    if not match:
        raise ValueError(f"Patch '{patch_name}' not found in boundary file")

    n_patch_faces = int(match.group(1))
    start_face = int(match.group(2))

    print(f"  Patch '{patch_name}': {n_patch_faces} faces starting at {start_face}")

    # Compute face centres
    centres = np.zeros((n_patch_faces, 3))
    for i in range(n_patch_faces):
        face_idx = start_face + i
        vert_ids = faces[face_idx]
        face_points = points[vert_ids]
        centres[i] = face_points.mean(axis=0)

    return centres


def idw_interpolate(
    src_points: np.ndarray,
    src_values: np.ndarray,
    dst_points: np.ndarray,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    """Inverse-distance weighted interpolation using k nearest neighbours.

    Parameters
    ----------
    src_points : (M, 3) source point coordinates
    src_values : (M, 3) source displacement vectors
    dst_points : (N, 3) destination point coordinates
    k : number of nearest neighbours
    power : distance weighting power (2 = inverse square)

    Returns
    -------
    dst_values : (N, 3) interpolated displacement vectors
    """
    tree = cKDTree(src_points)
    distances, indices = tree.query(dst_points, k=k)

    # Handle exact matches (distance = 0)
    dst_values = np.zeros((len(dst_points), 3))

    for i in range(len(dst_points)):
        dists = distances[i]
        idxs = indices[i]

        # If any distance is zero, use that point exactly
        zero_mask = dists < 1e-15
        if zero_mask.any():
            # Average of all zero-distance points
            dst_values[i] = src_values[idxs[zero_mask]].mean(axis=0)
        else:
            weights = 1.0 / dists**power
            weights /= weights.sum()
            dst_values[i] = (weights[:, np.newaxis] * src_values[idxs]).sum(axis=0)

    return dst_values


def write_boundary_points(filepath: str, points: np.ndarray):
    """Write points file in OpenFOAM boundaryData format."""
    with open(filepath, "w") as f:
        f.write(f"{len(points)}\n(\n")
        for p in points:
            f.write(f"({p[0]:.8f} {p[1]:.8f} {p[2]:.8f})\n")
        f.write(")\n")


def write_displacement(filepath: str, displacements: np.ndarray):
    """Write pointDisplacement file in OpenFOAM boundaryData format."""
    write_boundary_points(filepath, displacements)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <case_dir>")
        sys.exit(1)

    case_dir = Path(sys.argv[1])
    if not (case_dir / "constant" / "polyMesh").exists():
        print(f"Error: {case_dir}/constant/polyMesh not found")
        sys.exit(1)

    patches = ["wall", "inlet", "outlet"]

    # Step 1: Read mesh face centres for each patch
    print("Reading OpenFOAM mesh face centres...")
    patch_centres = {}
    for patch in patches:
        patch_centres[patch] = get_patch_face_centres(case_dir, patch)

    # Step 2: For each patch, read coarse data, interpolate, overwrite
    for patch in patches:
        bd_dir = case_dir / "constant" / "boundaryData" / patch
        if not bd_dir.exists():
            print(f"Warning: {bd_dir} not found, skipping")
            continue

        # Read coarse source points
        src_points = parse_boundary_data_points(str(bd_dir / "points"))
        dst_points = patch_centres[patch]
        print(f"\nPatch '{patch}': {len(src_points)} source → {len(dst_points)} destination points")

        # Build KDTree once for this patch
        tree = cKDTree(src_points)
        k = min(8, len(src_points))

        # Find all time directories
        time_dirs = sorted([
            d for d in bd_dir.iterdir()
            if d.is_dir() and (d / "pointDisplacement").exists()
        ], key=lambda d: float(d.name))

        print(f"  Interpolating {len(time_dirs)} time steps...")

        for td in time_dirs:
            src_disp = parse_displacement_file(str(td / "pointDisplacement"))
            dst_disp = idw_interpolate(src_points, src_disp, dst_points, k=k)
            write_displacement(str(td / "pointDisplacement"), dst_disp)

        # Overwrite points file with mesh face centres
        write_boundary_points(str(bd_dir / "points"), dst_points)
        print(f"  Done. Updated points ({len(dst_points)}) and {len(time_dirs)} displacement files.")

    print(f"\nInterpolation complete.")
    print(f"Now copy to processor directories:")
    print(f"  for i in $(seq 0 31); do")
    print(f"    rm -rf processor$i/constant/boundaryData")
    print(f"    cp -r constant/boundaryData processor$i/constant/")
    print(f"  done")


if __name__ == "__main__":
    main()
