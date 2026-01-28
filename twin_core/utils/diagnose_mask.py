#!/usr/bin/env python3
"""
diagnose_mask.py

Usage:
  python diagnose_mask.py --file "path/to/patient001_frame01_gt.nii.gz" --t 0 --z 12

This script prints detailed diagnostics and saves:
  - slice_colored.png  (discrete labels visualization, rounded to ints)
  - slice_hist.png     (histogram of raw values)
in the same folder as the input file.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys

def load_nifti_array(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # often (T,Z,Y,X) or (Z,Y,X)
        return np.asarray(arr)
    except Exception:
        try:
            import nibabel as nib
            nb = nib.load(str(path))
            arr = nb.get_fdata()
            return np.asarray(arr)
        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI: {e}")

def normalize_to_tzyx(arr: np.ndarray):
    # Try to coerce to (T,Z,Y,X) with simple heuristics
    if arr.ndim == 4:
        # prefer (T,Z,Y,X) if plausible
        if arr.shape[2] >= 8 and arr.shape[3] >= 8:
            return arr, "assumed (T,Z,Y,X)"
        # try common transposes
        candidates = [
            (arr, "as-is"),
            (np.transpose(arr, (3,2,1,0)), "transpose (X,Y,Z,T) -> (T,Z,Y,X)"),
            (np.transpose(arr, (3,0,1,2)), "transpose (X,Y,Z,T) alt -> (T,Z,Y,X)"),
            (np.transpose(arr, (0,3,2,1)), "transpose (T,X,Y,Z) -> (T,Z,Y,X)"),
        ]
        # pick candidate where last two dims look like image dims
        best = candidates[0]
        for cand, desc in candidates:
            T,Z,Y,X = cand.shape
            if Y >= 8 and X >= 8:
                best = (cand, desc)
                break
        return best
    elif arr.ndim == 3:
        return arr[np.newaxis, ...], "3D -> (1,Z,Y,X)"
    elif arr.ndim == 2:
        return arr[np.newaxis, np.newaxis, ...], "2D -> (1,1,Y,X)"
    else:
        raise ValueError(f"Unsupported ndim: {arr.ndim}")

def save_colored_slice(slice2d, out_path: Path, title="slice"):
    img = np.asarray(slice2d)
    # round floats to nearest int for discrete display
    rounded = np.rint(img).astype(np.int64)
    labels = np.unique(rounded)
    labels_sorted = np.sort(labels)
    # build colormap
    base_colors = plt.get_cmap("tab10").colors
    colors = list(base_colors)
    while len(colors) < len(labels_sorted):
        colors += list(base_colors)
    cmap = ListedColormap(colors[:len(labels_sorted)])
    bounds = np.concatenate([labels_sorted - 0.5, [labels_sorted[-1] + 0.5]])
    norm = BoundaryNorm(bounds, cmap.N)
    plt.figure(figsize=(6,6))
    im = plt.imshow(rounded, cmap=cmap, norm=norm, interpolation="nearest")
    cbar = plt.colorbar(im, ticks=labels_sorted)
    cbar.ax.set_yticklabels([str(int(l)) for l in labels_sorted])
    plt.title(title)
    plt.axis("off")
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close()

def save_histogram(arr, out_path: Path, title="hist"):
    flat = np.asarray(arr).ravel()
    plt.figure(figsize=(6,3))
    plt.hist(flat, bins=256, color="gray")
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close()

def main():
    p = Path(args.file)
    if not p.exists():
        print("File not found:", p)
        sys.exit(1)

    arr_raw = load_nifti_array(p)
    print("RAW shape:", arr_raw.shape, "dtype:", arr_raw.dtype)
    arr_norm, desc = normalize_to_tzyx(arr_raw)
    print("Normalized to (T,Z,Y,X):", arr_norm.shape, "method:", desc)

    T,Z,Y,X = arr_norm.shape
    print("\nGLOBAL unique values (raw):")
    uniq, counts = np.unique(arr_raw, return_counts=True)
    for u,c in zip(uniq, counts):
        print(f"  value={u}  count={c}")

    print("\nPer-timeframe unique values (raw):")
    for t in range(T):
        uniq_t = np.unique(arr_norm[t])
        print(f"  t={t}: {uniq_t.tolist()}")

    # Validate t,z
    t = int(args.t)
    z = int(args.z)
    if t < 0 or t >= T:
        print(f"Requested t={t} out of range (0..{T-1}). Using t=0.")
        t = 0
    if z < 0 or z >= Z:
        print(f"Requested z={z} out of range (0..{Z-1}). Using z=0.")
        z = 0

    slice2d = arr_norm[t, z]
    print(f"\nSelected slice t={t} z={z} shape {slice2d.shape}")
    uniq_slice, counts_slice = np.unique(slice2d, return_counts=True)
    print("  raw unique values in slice:", uniq_slice.tolist())
    print("  counts:", counts_slice.tolist())

    # Save outputs
    out_dir = p.parent
    colored_out = out_dir / f"{p.stem}_t{t}_z{z}_slice_colored.png"
    hist_out = out_dir / f"{p.stem}_t{t}_z{z}_slice_hist.png"

    # Save histogram of raw values (helps detect continuous/prob maps)
    save_histogram(slice2d, hist_out, title=f"Histogram t={t} z={z}")
    # Save colored discrete image (rounded)
    save_colored_slice(slice2d, colored_out, title=f"Mask t={t} z={z} (rounded)")

    print("\nSaved:")
    print("  colored slice:", colored_out)
    print("  histogram:", hist_out)

    # Interpret results and print guidance
    uniq_vals = set(np.unique(slice2d))
    if len(uniq_vals) == 1:
        v = next(iter(uniq_vals))
        print(f"\nINTERPRETATION: slice contains a single value = {v}.")
        if float(v).is_integer():
            print("  -> This means the slice is uniform (e.g., all background or one class).")
        else:
            print("  -> Values are constant floats; may be a constant intensity image.")
        print("  Suggestion: check other t/z slices or inspect global unique values above.")
    else:
        # if values are non-integer or many unique values, warn about probability map
        non_int = any(not float(x).is_integer() for x in uniq_vals)
        if non_int:
            print("\nINTERPRETATION: slice contains non-integer values (likely probabilities or floats).")
            print("  -> You should round or argmax across channels before visualizing as labels.")
        else:
            print("\nINTERPRETATION: slice contains multiple integer labels (good).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", required=True, help="Path to mask NIfTI")
    parser.add_argument("--t", type=int, default=0, help="time index (0-based)")
    parser.add_argument("--z", type=int, default=0, help="slice index (0-based)")
    args = parser.parse_args()
    main()
