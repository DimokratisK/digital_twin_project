#!/usr/bin/env python3
# inspect_mask_tz.py
from pathlib import Path
import re
import numpy as np
import nibabel as nib
import sys

_FRAME_RE = re.compile(r"(?:[_\-]|^)(?:frame|f)0*([0-9]+)", flags=re.IGNORECASE)

def parse_frame_index(fname: str):
    m = _FRAME_RE.search(fname)
    if not m:
        return None
    try:
        return max(0, int(m.group(1)) - 1)
    except Exception:
        return None

def load_mask(path: Path):
    nii = nib.load(str(path))
    arr = np.asarray(nii.get_fdata())
    return arr, nii

def nonzero_z_for_3d(arr3):
    # arr3 assumed (Z,Y,X)
    return [(int(z), int(np.count_nonzero(arr3[z]))) for z in range(arr3.shape[0]) if np.count_nonzero(arr3[z])>0]

def main(p):
    p = Path(p)
    if not p.exists():
        print("File not found:", p); sys.exit(1)
    arr, nii = load_mask(p)
    print("Loaded mask:", p.name)
    print("Array shape:", arr.shape, "dtype:", arr.dtype)
    t_from_name = parse_frame_index(p.name)
    print("Parsed frame index from filename (t):", t_from_name)
    if arr.ndim == 2:
        print("Mask is 2D. It represents a single slice. You must infer t from filename or dataset metadata.")
    elif arr.ndim == 3:
        # try to detect (Z,Y,X) vs (T,Y,X)
        z_dim = arr.shape[0]
        y_dim = arr.shape[1]
        x_dim = arr.shape[2]
        print("3D mask. Interpreting as (Z,Y,X) per-frame volume by default.")
        print("Nonzero counts per z (z,count):")
        for z,c in nonzero_z_for_3d(arr):
            print(" ", z, c)
        if t_from_name is not None:
            print("=> This file likely corresponds to t =", t_from_name, "in the 4D image. Compare 4D image at t and these z indices.")
    elif arr.ndim == 4:
        # try assume (T,Z,Y,X)
        print("4D mask detected. Assuming (T,Z,Y,X) ordering by default.")
        if t_from_name is not None and t_from_name < arr.shape[0]:
            print("Using parsed t from filename:", t_from_name)
            nonzero = [(z, int(np.count_nonzero(arr[t_from_name, z]))) for z in range(arr.shape[1]) if np.count_nonzero(arr[t_from_name, z])>0]
            print("Nonzero z for that t (z,count):", nonzero)
        else:
            print("Searching all (t,z) for nonzero voxels:")
            found = []
            for t in range(arr.shape[0]):
                for z in range(arr.shape[1]):
                    c = int(np.count_nonzero(arr[t,z]))
                    if c>0:
                        found.append((t,z,c))
            print("Found nonzero (t,z,count):", found)
    else:
        print("Unexpected array ndim:", arr.ndim)
    # print header info that can help align
    try:
        print("NIfTI shape (nibabel):", nii.shape)
        print("NIfTI zooms (pixdim):", nii.header.get_zooms())
    except Exception:
        pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_mask_tz.py path/to/mask.nii.gz")
        sys.exit(1)
    main(sys.argv[1])
