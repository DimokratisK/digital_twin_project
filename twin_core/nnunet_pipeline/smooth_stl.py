"""
Smooth an STL mesh using Taubin (default) or Laplacian filtering.

Works on any STL — per-structure output from predictions_to_stl.py, or
concatenated output from merge_stls.py. For concatenated files, each disjoint
shell is smoothed independently (Taubin operates on local vertex neighborhoods,
and concatenate produces shells with no shared vertices).

Usage:
    python -m twin_core.nnunet_pipeline.smooth_stl \
        -i outputs/bjonze_top5/merged/bjonze_218_merged.stl \
        -o outputs/bjonze_top5/merged/bjonze_218_smoothed.stl \
        --iterations 30
"""

import argparse
from pathlib import Path

import trimesh


def smooth_stl(
    in_path: Path,
    out_path: Path,
    iterations: int,
    method: str,
) -> None:
    mesh = trimesh.load(str(in_path), process=False)

    if method == "taubin":
        trimesh.smoothing.filter_taubin(mesh, iterations=iterations)
    elif method == "laplacian":
        trimesh.smoothing.filter_laplacian(mesh, lamb=0.5, iterations=iterations)
    else:
        raise ValueError(f"Unknown method: {method}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))
    print(
        f"Smoothed ({method}, {iterations} iters): {in_path.name} -> {out_path.name}  "
        f"({len(mesh.vertices)} verts, {len(mesh.faces)} faces)"
    )


def main():
    p = argparse.ArgumentParser(description="Smooth an STL mesh (Taubin or Laplacian)")
    p.add_argument("-i", "--input", type=Path, required=True, help="Input STL path")
    p.add_argument("-o", "--output", type=Path, required=True, help="Output STL path")
    p.add_argument(
        "--iterations", type=int, default=30,
        help="Number of smoothing iterations (default: 30)"
    )
    p.add_argument(
        "--method", type=str, default="taubin", choices=["taubin", "laplacian"],
        help="Smoothing method (default: taubin — preserves volume)"
    )
    args = p.parse_args()

    if not args.input.is_file():
        p.error(f"Input STL not found: {args.input}")

    smooth_stl(args.input, args.output, args.iterations, args.method)


if __name__ == "__main__":
    main()
