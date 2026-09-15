"""
Merge selected per-structure STL files into a single combined STL.

Consumes the output layout of predictions_to_stl.py:
    <input>/<case_name>/<StructName>.stl

The merge is a plain concatenation — each source STL becomes a disjoint shell
inside the output. All STLs share world coordinates (they came from the same
NIfTI), so no alignment is needed.

Usage:
    # Single case dir -> one merged STL
    python -m twin_core.nnunet_pipeline.merge_stls \
        -i outputs/bjonze_top5/stls_pred/bjonze_218 \
        -o outputs/bjonze_top5/merged/bjonze_218_LA_LAA_PV.stl \
        --structures LA LAA PV

    # Parent dir with multiple case dirs -> one merged STL per case
    python -m twin_core.nnunet_pipeline.merge_stls \
        -i outputs/bjonze_top5/stls_pred \
        -o outputs/bjonze_top5/merged \
        --structures LA LAA PV --suffix _LA_LAA_PV
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import trimesh


def _write_multi_solid_ascii_stl(
    meshes_by_name: Dict[str, trimesh.Trimesh],
    out_path: Path,
) -> None:
    """Write ASCII STL with one named `solid <name>` block per input mesh.

    SimVascular reads each named solid as a separate face/region. ParaView's
    STL reader flattens solid names — use Filters > Connectivity there instead.
    """
    with open(out_path, "w") as f:
        for name, mesh in meshes_by_name.items():
            f.write(f"solid {name}\n")
            for face, normal in zip(mesh.faces, mesh.face_normals):
                v0, v1, v2 = mesh.vertices[face]
                f.write(
                    f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n"
                )
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid {name}\n")


def merge_case_stls(
    case_dir: Path,
    structures: List[str],
    out_path: Path,
    multi_region: bool = False,
) -> Optional[trimesh.Trimesh]:
    meshes_by_name: Dict[str, trimesh.Trimesh] = {}
    missing = []
    for name in structures:
        stl = case_dir / f"{name}.stl"
        if not stl.is_file():
            missing.append(name)
            continue
        meshes_by_name[name] = trimesh.load(str(stl), process=False)

    if missing:
        print(f"  WARN: {case_dir.name} — missing STLs: {', '.join(missing)}")
    if not meshes_by_name:
        print(f"  SKIP {case_dir.name} — no requested structures found")
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged = trimesh.util.concatenate(list(meshes_by_name.values()))

    if multi_region:
        _write_multi_solid_ascii_stl(meshes_by_name, out_path)
        fmt = "ASCII multi-solid"
    else:
        merged.export(str(out_path))
        fmt = "binary concatenated"

    print(
        f"  Wrote {out_path}  ({fmt}; "
        f"{len(meshes_by_name)} shells, {len(merged.vertices)} verts, {len(merged.faces)} faces)"
    )
    return merged


def main():
    p = argparse.ArgumentParser(
        description="Concatenate per-structure STLs into one combined STL for SimVascular import"
    )
    p.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Case dir with <StructName>.stl files, OR parent dir containing multiple case dirs"
    )
    p.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output STL file (if -i is a case dir) or output dir (if -i is a parent dir)"
    )
    p.add_argument(
        "--structures", nargs="+", required=True,
        help="Structure names to include (matches STL filename stems, e.g. LA LAA PV)"
    )
    p.add_argument(
        "--suffix", type=str, default="_merged",
        help="Suffix appended to per-case merged filenames when -i is a parent dir (default: _merged)"
    )
    p.add_argument(
        "--multi-region", action="store_true",
        help="Write ASCII STL with one named `solid <StructName>` block per input structure "
             "(for SimVascular multi-face import). Default is binary concatenated STL."
    )
    args = p.parse_args()

    if not args.input.exists():
        p.error(f"Input path does not exist: {args.input}")

    has_stls_directly = any(args.input.glob("*.stl"))
    if has_stls_directly:
        merge_case_stls(args.input, args.structures, args.output, multi_region=args.multi_region)
    else:
        case_dirs = sorted(d for d in args.input.iterdir() if d.is_dir())
        if not case_dirs:
            p.error(f"No STLs and no subdirectories in {args.input}")
        for cd in case_dirs:
            out = args.output / f"{cd.name}{args.suffix}.stl"
            merge_case_stls(cd, args.structures, out, multi_region=args.multi_region)


if __name__ == "__main__":
    main()
