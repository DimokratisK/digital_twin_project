"""
Replace coarse STL-vertex boundary data with mesh-vertex-density boundary data.

Background
----------
generate_dynamic_case.py writes constant/boundaryData/{wall,inlet,outlet}/ using
the ~4800 vertices of the registered template STL. After snappyHexMesh, the wall
patch has ~95,000 mesh vertices. The timeVaryingMappedFixedValue BC then maps
4800 -> 95000 via mapMethod nearest, which produces piecewise-constant
displacement across mesh vertices and stretches the few skew faces into ill-
conditioned cells. Result: localized Co spikes and solver FPE.

This script regenerates wall boundary data so the sample point count matches
the mesh vertex count exactly (1:1 mapping under nearest), with displacements
barycentrically interpolated from the registered STL frames.

It also rewrites 0/pointDisplacement to keep inlet/outlet cut-plane patches
fixed (those don't physically deform with the cardiac wall).

Usage
-----
    python densify_boundary_data.py \\
        --case ~/cfd_runs/dynamic_mesh/case_v2 \\
        --stl-dir ~/digital_twin_project/test_3d/cfd_dynamic/stl_series \\
        --frames 28 --cardiac-cycle 1.0 --num-cycles 5

The script preserves the existing time-directory naming (e.g. 0.02857143/) so
controlDict and the BC continue to work without modification.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# polyMesh ASCII parsing
# ---------------------------------------------------------------------------

_VEC_RE = re.compile(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)")
# Body restricted to digits/whitespace so we never accidentally match across a face boundary.
_FACE_RE = re.compile(r"(\d+)\s*\(\s*([\d\s]*?)\s*\)")
_PATCH_RE = re.compile(
    r"(\w+)\s*\{[^}]*?type\s+(\w+)\s*;\s*[^}]*?nFaces\s+(\d+)\s*;\s*[^}]*?startFace\s+(\d+)\s*;[^}]*?\}",
    re.DOTALL,
)


def _strip_foam_header(text: str) -> str:
    """Strip the FoamFile { ... } header and the boilerplate comment block."""
    idx = text.find("FoamFile")
    if idx >= 0:
        depth = 0
        i = text.find("{", idx)
        if i >= 0:
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[j + 1 :]
                        break
    return text


def _detect_binary(path: Path) -> bool:
    head = path.read_bytes()[:4096].decode("utf-8", errors="replace")
    if "format" in head:
        m = re.search(r"format\s+(\w+)\s*;", head)
        if m and m.group(1).lower() == "binary":
            return True
    return False


def parse_points(polymesh_dir: Path) -> np.ndarray:
    path = polymesh_dir / "points"
    if _detect_binary(path):
        raise RuntimeError(
            f"{path} is binary. Convert with `foamFormatConvert -ascii` "
            "before running this script."
        )
    text = _strip_foam_header(path.read_text())
    matches = _VEC_RE.findall(text)
    if not matches:
        raise RuntimeError(f"No (x y z) tuples parsed from {path}")
    pts = np.array(matches, dtype=np.float64)
    print(f"  Parsed {len(pts)} mesh points from {path.name}")
    return pts


def parse_faces(polymesh_dir: Path) -> list[list[int]]:
    """Return faces as a list of vertex-index lists.

    Expected ASCII layout (OF11 default):
        NFACES
        (
        N0(v0 v1 ... vN0-1)
        N1(v0 v1 ... vN1-1)
        ...
        )
    """
    path = polymesh_dir / "faces"
    if _detect_binary(path):
        raise RuntimeError(
            f"{path} is binary. Convert with `foamFormatConvert -ascii` "
            "before running this script."
        )
    text = _strip_foam_header(path.read_text())
    # Drop comment lines so the outer-list-opener search is unambiguous.
    text = re.sub(r"//.*?(?:\n|$)", "\n", text)
    # Strip the outer list opener "NFACES\n(" so its NFACES doesn't get parsed as a face header.
    m = re.search(r"(\d+)\s*\(", text)
    if not m:
        raise RuntimeError(f"Could not locate outer face list opener in {path}")
    expected_n = int(m.group(1))
    inner = text[m.end():]
    # Cut off the trailing closing paren of the outer list, if present.
    last_paren = inner.rfind(")")
    if last_paren >= 0:
        inner = inner[:last_paren]
    faces: list[list[int]] = []
    for n_str, body in _FACE_RE.findall(inner):
        idx = [int(x) for x in body.split()]
        if len(idx) != int(n_str):
            raise RuntimeError(
                f"Face vertex count mismatch in {path}: header={n_str} body={idx}"
            )
        faces.append(idx)
    print(f"  Parsed {len(faces)} mesh faces from {path.name} "
          f"(expected {expected_n})")
    if len(faces) != expected_n:
        raise RuntimeError(
            f"Face count mismatch: parsed {len(faces)} but file declares {expected_n}. "
            f"polyMesh/faces may be in a format this script doesn't handle."
        )
    return faces


def parse_boundary(polymesh_dir: Path) -> dict[str, tuple[int, int]]:
    """Return {patch_name: (startFace, nFaces)}."""
    path = polymesh_dir / "boundary"
    text = _strip_foam_header(path.read_text())
    out: dict[str, tuple[int, int]] = {}
    for name, _type, n_faces, start in _PATCH_RE.findall(text):
        out[name] = (int(start), int(n_faces))
    if not out:
        raise RuntimeError(f"No patches parsed from {path}")
    print(f"  Parsed {len(out)} boundary patches: {list(out.keys())}")
    return out


# ---------------------------------------------------------------------------
# Patch vertex extraction
# ---------------------------------------------------------------------------

def patch_unique_vertex_indices(
    faces: list[list[int]], start: int, n_faces: int
) -> np.ndarray:
    seen: dict[int, None] = {}
    for f in faces[start : start + n_faces]:
        for v in f:
            seen.setdefault(v, None)
    return np.array(list(seen.keys()), dtype=np.int64)


def shared_vertex_mask(
    wall_vert_idx: np.ndarray,
    faces: list[list[int]],
    boundary: dict[str, tuple[int, int]],
    wall_patch_name: str,
) -> np.ndarray:
    """Mask over wall_vert_idx: True for vertices that ALSO appear in any non-wall patch.

    These are the cut-ring vertices on the seam between wall and inlet/outlet.
    The inlet/outlet pointDisplacement BC pins them to zero; if the wall BC
    sends them nonzero displacement we get a BC conflict at every shared point
    -> mesh tearing -> Co explosion. Pinning them to zero in the wall data too
    keeps both BCs consistent.
    """
    other_verts: set[int] = set()
    for name, (start, n) in boundary.items():
        if name == wall_patch_name:
            continue
        for f in faces[start : start + n]:
            other_verts.update(f)
    return np.array([int(v) in other_verts for v in wall_vert_idx], dtype=bool)


# ---------------------------------------------------------------------------
# Displacement interpolation
# ---------------------------------------------------------------------------

def load_stl_frames(stl_dir: Path, n_frames: int) -> list[np.ndarray]:
    """Load registered STL series; all frames must share topology with frame 0."""
    pattern_candidates = [
        "patient006_frame{:02d}_LV.stl",
        "frame{:02d}.stl",
        "frame_{:02d}.stl",
    ]
    chosen = None
    for pat in pattern_candidates:
        if (stl_dir / pat.format(0)).exists():
            chosen = pat
            break
    if chosen is None:
        raise FileNotFoundError(
            f"Could not find STL pattern in {stl_dir}. Tried: {pattern_candidates}. "
            f"Edit pattern_candidates to match your filenames."
        )
    print(f"  STL filename pattern: {chosen}")
    frames = []
    for i in range(n_frames):
        m = trimesh.load(str(stl_dir / chosen.format(i)), force="mesh")
        frames.append(np.asarray(m.vertices, dtype=np.float64))
    n_v = len(frames[0])
    for i, vp in enumerate(frames):
        if len(vp) != n_v:
            raise RuntimeError(
                f"Frame {i} vertex count {len(vp)} != frame 0 count {n_v}"
            )
    print(f"  Loaded {n_frames} frames, {n_v} vertices each")
    return frames, chosen


def precompute_barycentric(
    frame0_vertices: np.ndarray, frame0_faces: np.ndarray, target_pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each target point, find closest face on frame-0 surface + barycentric coords.

    Returns
    -------
    face_ids : (M,) int64
    bary     : (M, 3) float64, rows sum to ~1
    dist     : (M,) float64, distance from each target to its closest frame-0 face
    """
    surface = trimesh.Trimesh(
        vertices=frame0_vertices, faces=frame0_faces, process=False
    )
    closest, dist, face_ids = trimesh.proximity.closest_point(surface, target_pts)
    print(
        f"  Closest-face stats: max dist = {dist.max():.3e} m, "
        f"mean = {dist.mean():.3e} m, "
        f"({(dist > 1e-3).sum()} vertices > 1mm from frame-0 surface)"
    )
    triangles = surface.triangles[face_ids]  # (M, 3, 3)
    bary = trimesh.triangles.points_to_barycentric(triangles, closest)
    return face_ids, bary, dist


def displacement_at_targets(
    frame_vertices: np.ndarray,
    frame0_vertices: np.ndarray,
    faces: np.ndarray,
    face_ids: np.ndarray,
    bary: np.ndarray,
    freeze_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate per-vertex displacement (frame - frame0) onto target points.

    Targets where freeze_mask is True get zero displacement (e.g. cut-plane rim
    vertices that are far from the frame-0 surface and would otherwise receive
    extrapolated nonsense).
    """
    disp_per_vertex = frame_vertices - frame0_vertices  # (V, 3)
    tri_disp = disp_per_vertex[faces[face_ids]]  # (M, 3, 3)
    out = np.einsum("mi,mij->mj", bary, tri_disp)  # (M, 3)
    if freeze_mask is not None:
        out[freeze_mask] = 0.0
    return out


# ---------------------------------------------------------------------------
# OpenFOAM file writing
# ---------------------------------------------------------------------------

def write_vector_list(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(data)
    lines = [f"{n}", "("]
    for v in data:
        lines.append(f"({v[0]:.8f} {v[1]:.8f} {v[2]:.8f})")
    lines.append(")")
    lines.append("")
    path.write_text("\n".join(lines))


def existing_time_dirs(boundary_data_patch: Path) -> list[tuple[float, Path]]:
    out = []
    for d in boundary_data_patch.iterdir():
        if d.is_dir():
            try:
                out.append((float(d.name), d))
            except ValueError:
                pass
    out.sort(key=lambda p: p[0])
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True,
                   help="OpenFOAM case directory (must contain constant/polyMesh)")
    p.add_argument("--stl-dir", type=Path, required=True,
                   help="Directory of registered STL frames")
    p.add_argument("--frames", type=int, default=28,
                   help="Frames per cardiac cycle (default: 28)")
    p.add_argument("--cardiac-cycle", type=float, default=1.0,
                   help="Cardiac cycle period in seconds (default: 1.0)")
    p.add_argument("--num-cycles", type=int, default=5,
                   help="Number of cycles in boundary data (default: 5)")
    p.add_argument("--patch", default="wall",
                   help="Patch to densify (default: wall)")
    p.add_argument("--freeze-cut-planes", action="store_true", default=True,
                   help="Switch inlet/outlet pointDisplacement BC to fixedValue (0 0 0)")
    p.add_argument("--no-freeze-cut-planes", dest="freeze_cut_planes",
                   action="store_false")
    p.add_argument("--freeze-distance", type=float, default=1.0,
                   help="Optional supplementary distance freeze: wall vertices farther "
                        "than this many metres from the frame-0 STL surface get zero "
                        "displacement. Default 1.0 m = effectively disabled. Distance "
                        "freezing was found to create discontinuities (frozen vertex "
                        "next to moving neighbour -> tearing). Use topology freeze "
                        "(automatic, on by default) instead.")
    p.add_argument("--no-topology-freeze", dest="topology_freeze",
                   action="store_false", default=True,
                   help="Disable the topology-based freeze of cut-ring vertices "
                        "shared with inlet/outlet patches. NOT recommended — these "
                        "shared points have conflicting BCs (wall: move, inlet: "
                        "fixed) and tear the mesh if not pinned.")
    args = p.parse_args()

    case_dir: Path = args.case.expanduser().resolve()
    polymesh_dir = case_dir / "constant" / "polyMesh"
    bdata_dir = case_dir / "constant" / "boundaryData" / args.patch

    if not polymesh_dir.exists():
        raise SystemExit(f"polyMesh missing: {polymesh_dir}")
    if not bdata_dir.exists():
        raise SystemExit(f"boundaryData/{args.patch} missing: {bdata_dir}")

    # 1. Parse mesh
    print("Step 1: Parsing polyMesh...")
    points = parse_points(polymesh_dir)
    faces = parse_faces(polymesh_dir)
    boundary = parse_boundary(polymesh_dir)
    if args.patch not in boundary:
        raise SystemExit(
            f"Patch '{args.patch}' not found in mesh boundary. "
            f"Available: {list(boundary.keys())}"
        )
    start, n_faces = boundary[args.patch]
    patch_vert_idx = patch_unique_vertex_indices(faces, start, n_faces)
    patch_pts = points[patch_vert_idx]
    print(f"  {args.patch}: {len(patch_pts)} unique vertices "
          f"({n_faces} faces from index {start})")

    # 2. Load STL frames + frame-0 surface for barycentric setup
    print("\nStep 2: Loading registered STL frames...")
    frames, _pattern = load_stl_frames(args.stl_dir.expanduser(), args.frames)
    template_mesh = trimesh.load(
        str(args.stl_dir.expanduser() / _pattern.format(0)), force="mesh"
    )
    frame_faces = np.asarray(template_mesh.faces, dtype=np.int64)

    print("\nStep 3: Computing closest-face + barycentric for each mesh vertex...")
    face_ids, bary, dist = precompute_barycentric(frames[0], frame_faces, patch_pts)

    if args.topology_freeze:
        topology_mask = shared_vertex_mask(patch_vert_idx, faces, boundary, args.patch)
        print(
            f"  Topology freeze: {topology_mask.sum()} / {len(patch_pts)} wall "
            f"vertices shared with non-wall patches (cut-ring) -> pinned to zero"
        )
    else:
        topology_mask = np.zeros(len(patch_pts), dtype=bool)
        print("  Topology freeze: disabled by --no-topology-freeze")

    distance_mask = dist > args.freeze_distance
    if distance_mask.any():
        print(
            f"  Distance freeze: {distance_mask.sum()} / {len(patch_pts)} vertices "
            f"> {args.freeze_distance:.3e} m -> pinned to zero"
        )
    freeze_mask = topology_mask | distance_mask
    print(f"  Combined freeze mask: {freeze_mask.sum()} / {len(patch_pts)} vertices")

    # 3. Backup existing data, write new points file
    print("\nStep 4: Backing up existing boundaryData and writing new points file...")
    backup = bdata_dir.with_suffix(".bak")
    if backup.exists():
        shutil.rmtree(backup)
    shutil.copytree(bdata_dir, backup)
    print(f"  Backup at {backup}")

    write_vector_list(bdata_dir / "points", patch_pts)
    print(f"  Wrote {len(patch_pts)} points to {bdata_dir}/points")

    # 4. Regenerate each existing time directory's pointDisplacement
    print("\nStep 5: Regenerating displacement per time directory...")
    time_dirs = existing_time_dirs(bdata_dir)
    n_existing = len(time_dirs)
    print(f"  Found {n_existing} existing time directories")
    expected = args.num_cycles * args.frames + args.num_cycles  # frames + cycle-end zeros
    if n_existing not in (expected, expected - 1):
        print(f"  NOTE: existing dir count {n_existing} != expected ~{expected}")

    for t_abs, td in time_dirs:
        t_in_cycle = t_abs % args.cardiac_cycle
        # Detect cycle-end zero snapshots: t mod cycle ~= 0 AND t > 0
        is_cycle_close = (
            abs(t_in_cycle) < 1e-6 or
            abs(t_in_cycle - args.cardiac_cycle) < 1e-6
        ) and t_abs > 1e-6
        if is_cycle_close:
            disp = np.zeros_like(patch_pts)
        else:
            frame_idx = int(round(t_in_cycle / args.cardiac_cycle * args.frames))
            frame_idx %= args.frames
            disp = displacement_at_targets(
                frames[frame_idx], frames[0], frame_faces, face_ids, bary,
                freeze_mask=freeze_mask,
            )
        write_vector_list(td / "pointDisplacement", disp)
    print(f"  Regenerated {n_existing} pointDisplacement files")

    # 5. Optionally rewrite 0/pointDisplacement to fix cut planes
    if args.freeze_cut_planes:
        print("\nStep 6: Freezing inlet/outlet pointDisplacement BC...")
        zero_pd = case_dir / "0" / "pointDisplacement"
        if zero_pd.exists():
            shutil.copy2(zero_pd, zero_pd.with_suffix(".pre_densify"))
            text = zero_pd.read_text()
            for patch in ("inlet", "outlet"):
                pattern = re.compile(
                    rf"({re.escape(patch)}\s*\{{)[^}}]*(\}})",
                    re.DOTALL,
                )
                replacement = (
                    rf"\1\n        type            fixedValue;\n"
                    rf"        value           uniform (0 0 0);\n    \2"
                )
                new_text, n_sub = pattern.subn(replacement, text, count=1)
                if n_sub == 1:
                    text = new_text
                    print(f"  Set {patch} -> fixedValue (0 0 0)")
                else:
                    print(f"  WARNING: could not patch {patch} block")
            zero_pd.write_text(text)
        else:
            print(f"  {zero_pd} missing, skipping")

    print("\nDone.")
    print(f"  Wall boundary samples: {len(patch_pts)} (was the old STL count)")
    print(f"  Time directories regenerated: {n_existing}")
    print("  Next: rm -rf processor*; decomposePar; mpirun foamRun ...")


if __name__ == "__main__":
    main()
