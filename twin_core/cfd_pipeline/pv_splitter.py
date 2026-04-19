"""
Split lumped PV label from Bjonze 11-class nnUNet prediction into N separate
trunks, and emit a multi-label mask (LA + LAA + PV_1..N) ready for
predictions_to_stl.py.

Algorithm: erosion -> connected components -> distance-transform expansion.
N is adaptive (3-5 accepted). Trunks are sorted by centroid X descending,
so trunk_1 is the most-positive-X CC (verify orientation in ParaView before
assigning anatomical names like RSPV/LIPV).

Usage:
    python -m twin_core.cfd_pipeline.pv_splitter \\
        -i /tmp/bjonze_test_out/bjonze_168.nii.gz \\
        -o /tmp/bjonze_168_cfd_labels.nii.gz \\
        --erosion-iters 2 --min-cc-voxels 500
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi

BJONZE_LA = 2
BJONZE_LAA = 8
BJONZE_PV = 10


def split_pv(pv_mask: np.ndarray, erosion_iters: int = 2, min_cc_voxels: int = 500):
    """Return labeled PV mask (0 bg, 1..N trunks) plus per-CC info.

    N is adaptive: count of eroded CCs with size >= min_cc_voxels.
    Seeds are expanded back to the full pv_mask via nearest-seed EDT.
    """
    eroded = ndi.binary_erosion(pv_mask, iterations=erosion_iters)
    cc_labels, n_cc = ndi.label(eroded)
    if n_cc == 0:
        raise RuntimeError(
            f"0 CCs after {erosion_iters}x erosion. Over-eroded or empty PV mask."
        )

    sizes = ndi.sum(eroded, cc_labels, range(1, n_cc + 1)).astype(int)
    kept = [i + 1 for i, s in enumerate(sizes) if s >= min_cc_voxels]
    if len(kept) == 0:
        raise RuntimeError(
            f"No PV CCs >= {min_cc_voxels} voxels after {erosion_iters}x erosion. "
            f"Largest CCs: {sorted(sizes, reverse=True)[:5]}"
        )

    seeds = np.zeros_like(cc_labels, dtype=np.int16)
    for new_idx, old_idx in enumerate(kept, start=1):
        seeds[cc_labels == old_idx] = new_idx

    _, inds = ndi.distance_transform_edt(seeds == 0, return_indices=True)
    expanded = seeds[tuple(inds)]
    pv_split = np.where(pv_mask, expanded, 0).astype(np.int16)
    return pv_split, sizes


def sort_and_relabel_by_x(pv_split: np.ndarray, n_trunks: int) -> np.ndarray:
    """Relabel trunks 1..N sorted by centroid X descending."""
    centroids = [
        (old, np.array(ndi.center_of_mass(pv_split == old)))
        for old in range(1, n_trunks + 1)
    ]
    centroids.sort(key=lambda t: -t[1][0])
    lut = np.zeros(n_trunks + 1, dtype=np.int16)
    for new_idx, (old_idx, _) in enumerate(centroids, start=1):
        lut[old_idx] = new_idx
    return lut[pv_split], centroids


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", required=True, type=Path,
                    help="Bjonze 11-class prediction NIfTI")
    ap.add_argument("-o", "--output", required=True, type=Path,
                    help="Multi-label output NIfTI (LA=1, LAA=2, PV_1..N=3..)")
    ap.add_argument("--erosion-iters", type=int, default=2)
    ap.add_argument("--min-cc-voxels", type=int, default=500)
    args = ap.parse_args()

    img = nib.load(str(args.input))
    pred = np.asarray(img.dataobj, dtype=np.int16)
    la = pred == BJONZE_LA
    laa = pred == BJONZE_LAA
    pv = pred == BJONZE_PV

    print(f"Input: {args.input}")
    print(f"  LA: {la.sum()}  LAA: {laa.sum()}  PV: {pv.sum()} voxels")

    pv_split, _ = split_pv(pv, args.erosion_iters, args.min_cc_voxels)
    n_trunks = int(pv_split.max())
    print(f"\nPV split: {n_trunks} trunks (erosion {args.erosion_iters}x, "
          f"min {args.min_cc_voxels} voxels)")
    if n_trunks not in (3, 4, 5):
        print(f"  WARNING: expected 3-5 trunks, got {n_trunks}. Review before CFD.")

    pv_split, centroids = sort_and_relabel_by_x(pv_split, n_trunks)
    la_centroid = np.array(ndi.center_of_mass(la))
    print(f"  LA centroid (voxel idx): {la_centroid.round(1)}")
    print("  Trunks (sorted by X descending):")
    for new_idx, (_, c) in enumerate(centroids, start=1):
        dx, dy, dz = c - la_centroid
        size = int((pv_split == new_idx).sum())
        print(f"    trunk_{new_idx}: centroid={c.round(1)}  "
              f"offset=({dx:+.1f},{dy:+.1f},{dz:+.1f})  size={size}")

    out = np.zeros_like(pred, dtype=np.int16)
    out[la] = 1
    out[laa] = 2
    for new_idx in range(1, n_trunks + 1):
        out[pv_split == new_idx] = 2 + new_idx

    print(f"\nOutput labels:")
    for v in range(int(out.max()) + 1):
        cnt = int((out == v).sum())
        if cnt == 0:
            continue
        name = {0: "bg", 1: "LA", 2: "LAA"}.get(v, f"PV_{v - 2}")
        print(f"  {v} {name:<6}: {cnt} voxels")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out, img.affine, img.header), str(args.output))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
