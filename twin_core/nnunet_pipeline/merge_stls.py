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
from typing import List, Optional

import trimesh


def merge_case_stls(
    case_dir: Path,
    structures: List[str],
    out_path: Path,
) -> Optional[trimesh.Trimesh]:
    meshes = []
    missing = []
    for name in structures:
        stl = case_dir / f"{name}.stl"
        if not stl.is_file():
            missing.append(name)
            continue
        meshes.append(trimesh.load(str(stl), process=False))

    if missing:
        print(f"  WARN: {case_dir.name} — missing STLs: {', '.join(missing)}")
    if not meshes:
        print(f"  SKIP {case_dir.name} — no requested structures found")
        return None

    merged = trimesh.util.concatenate(meshes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.export(str(out_path))
    print(
        f"  Wrote {out_path}  "
        f"({len(meshes)} shells, {len(merged.vertices)} verts, {len(merged.faces)} faces)"
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
    args = p.parse_args()

    if not args.input.exists():
        p.error(f"Input path does not exist: {args.input}")

    has_stls_directly = any(args.input.glob("*.stl"))
    if has_stls_directly:
        merge_case_stls(args.input, args.structures, args.output)
    else:
        case_dirs = sorted(d for d in args.input.iterdir() if d.is_dir())
        if not case_dirs:
            p.error(f"No STLs and no subdirectories in {args.input}")
        for cd in case_dirs:
            out = args.output / f"{cd.name}{args.suffix}.stl"
            merge_case_stls(cd, args.structures, out)


if __name__ == "__main__":
    main()
