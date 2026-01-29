#!/usr/bin/env python3
"""
overlay_image_and_mask.py

Load an image (2D/3D/4D NIfTI) and a mask (2D/3D/4D NIfTI) and display/save an overlay:
 - image shown in grayscale
 - mask overlaid with discrete colors (labels) or contour outlines
 - automatic selection of time/frame (t) and slice (z) if possible; user can override via CLI

New:
 - --flip180 flag rotates both image and mask 180 degrees before display/save.

Deterministic color mapping:
 0: BG (transparent)
 1: RV -> yellow (#ffd700)
 2: MYO -> green  (#4daf4a)
 3: LV  -> red    (#ff0000)
"""
from pathlib import Path
import re
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
import matplotlib.patches as mpatches
from typing import Optional, Tuple

# Regex to parse frame index like "frame01" or "f01"
_FRAME_RE = re.compile(r"(?:[_\-]|^)(?:frame|f)0*([0-9]+)", flags=re.IGNORECASE)


def parse_frame_index_from_name(fname: str) -> Optional[int]:
    m = _FRAME_RE.search(fname)
    if not m:
        return None
    try:
        idx = int(m.group(1))
        return max(0, idx - 1)
    except Exception:
        return None


# -------------------------
# Deterministic class labels and colors (EDITED)
# -------------------------
CLASS_LABELS = {
    0: "BG",
    1: "RV",
    2: "MYO",
    3: "LV",
}

# Color mapping: index -> color string or rgba tuple.
# Background kept transparent by default (alpha = 0.0).
CLASS_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),  # BG transparent
    1: "#ffd700",             # RV - yellow
    2: "#4daf4a",             # MYO - green
    3: "#ff0000",             # LV - red
}


# -------------------------
# I/O / axis helpers
# -------------------------
def reorder_to_tzyx(nib_img: nib.Nifti1Image) -> np.ndarray:
    """
    Return a numpy array canonicalized to (T, Z, Y, X).
    Conservative heuristics that work for typical cardiac NIfTIs.
    """
    try:
        img_c = nib.as_closest_canonical(nib_img)
        arr = img_c.get_fdata()
    except Exception:
        arr = nib_img.get_fdata()
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis, :, :]
    if arr.ndim == 3:
        # common case: (X,Y,Z) -> (1,Z,Y,X)
        return np.transpose(arr, (2, 1, 0))[None]
    if arr.ndim == 4:
        shape = arr.shape
        # If last axis looks like time, move last -> first then transpose to (T,Z,Y,X)
        if shape[-1] > 1 and shape[-1] <= max(shape[0], shape[1], shape[2]):
            arr_tfirst = np.moveaxis(arr, -1, 0)
            try:
                return np.transpose(arr_tfirst, (0, 3, 2, 1))
            except Exception:
                return arr_tfirst
        # If first axis looks like time
        if shape[0] > 1 and (shape[1] >= 1):
            try:
                return np.transpose(arr, (0, 3, 2, 1))
            except Exception:
                return np.moveaxis(arr, -1, 0)
        # fallback
        arr_tfirst = np.moveaxis(arr, -1, 0)
        try:
            return np.transpose(arr_tfirst, (0, 3, 2, 1))
        except Exception:
            return arr_tfirst
    raise RuntimeError(f"Unsupported array ndim: {arr.ndim}")


def select_slice_from_arr(arr_tzyx: np.ndarray, t: Optional[int], z: Optional[int]) -> Tuple[np.ndarray, int, int]:
    T, Z, Y, X = arr_tzyx.shape
    chosen_t = 0 if t is None else int(min(max(0, t), T - 1))
    chosen_z = int(min(max(0, Z // 2), Z - 1)) if z is None else int(min(max(0, z), Z - 1))
    return arr_tzyx[chosen_t, chosen_z], chosen_t, chosen_z


def load_and_select(image_path: Path, mask_path: Path, t: Optional[int], z: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray], int, int]:
    img_nib = nib.load(str(image_path))
    img_tzyx = reorder_to_tzyx(img_nib)

    mask_slice = None
    mask_tzyx = None
    if mask_path is not None:
        mask_nib = nib.load(str(mask_path))
        mask_tzyx = reorder_to_tzyx(mask_nib)

    # heuristics to pick t/z if not provided
    if t is None:
        t_candidate = parse_frame_index_from_name(image_path.name) or (parse_frame_index_from_name(mask_path.name) if mask_path is not None else None)
        t_use = t_candidate
    else:
        t_use = t

    if z is None:
        z_re = re.compile(r"(?:[_\-]|^)(?:z)0*([0-9]+)", flags=re.IGNORECASE)
        z_m = z_re.search(image_path.name) or (z_re.search(mask_path.name) if mask_path is not None else None)
        z_use_candidate = int(z_m.group(1)) if (z_m is not None) else None
        z_use = z_use_candidate
    else:
        z_use = z

    img_slice, chosen_t, chosen_z = select_slice_from_arr(img_tzyx, t_use, z_use)

    if mask_tzyx is not None:
        if mask_tzyx.shape[0] > chosen_t and mask_tzyx.shape[1] > chosen_z:
            mask_raw = mask_tzyx[chosen_t, chosen_z]
        else:
            t_m = min(chosen_t, mask_tzyx.shape[0] - 1)
            z_m = min(chosen_z, mask_tzyx.shape[1] - 1)
            mask_raw = mask_tzyx[t_m, z_m]
        mask_slice = np.asarray(mask_raw).astype(np.int64)
        if mask_slice.shape != img_slice.shape:
            mask_slice = _align_mask_to_image(mask_slice, img_slice.shape)

    return img_slice, mask_slice, chosen_t, chosen_z


def _align_mask_to_image(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    mh, mw = mask.shape
    th, tw = target_shape
    if mask.T.shape == target_shape:
        return mask.T
    out = np.zeros(target_shape, dtype=mask.dtype)
    ch = min(mh, th); cw = min(mw, tw)
    out[:ch, :cw] = mask[:ch, :cw]
    return out


# -------------------------
# Colormap creation with fixed mapping index->color
# -------------------------
def build_colormap_fixed(num_classes: Optional[int] = None, user_colors: Optional[dict] = None, labels_present: Optional[np.ndarray] = None):
    """
    Build ListedColormap and BoundaryNorm where index i maps to a deterministic color.
    - num_classes: if provided, produce colors for 0..num_classes-1
    - user_colors: dict {idx: colorstr_or_tuple}
    - labels_present: mask array used to detect max label if num_classes not given
    Returns (cmap, norm, color_list)
    """
    max_label_mask = int(np.max(labels_present)) if (labels_present is not None and labels_present.size > 0) else -1
    max_user = max(user_colors.keys()) if (user_colors is not None and len(user_colors) > 0) else -1
    n = 0
    if num_classes is not None:
        n = int(num_classes)
    else:
        n = max(max_label_mask + 1, max_user + 1, max(CLASS_COLORS.keys()) + 1)
    if n <= 0:
        n = 1

    color_list = []
    for i in range(n):
        # precedence: user_colors -> CLASS_COLORS -> fallback tab10
        col = None
        if user_colors and i in user_colors:
            col = user_colors[i]
        elif i in CLASS_COLORS:
            col = CLASS_COLORS[i]
        if col is None:
            cmap_tab = plt.get_cmap("tab10")
            rgba = cmap_tab(i % 10)
        else:
            try:
                if isinstance(col, str):
                    rgba = to_rgba(col)
                else:
                    colt = tuple(float(x) for x in col)
                    if len(colt) == 3:
                        rgba = (colt[0], colt[1], colt[2], 1.0)
                    else:
                        rgba = colt
            except Exception:
                cmap_tab = plt.get_cmap("tab10")
                rgba = cmap_tab(i % 10)
        color_list.append(rgba)

    cmap = ListedColormap(color_list)
    bounds = np.arange(-0.5, n + 0.5, 1.0)
    norm = BoundaryNorm(boundaries=bounds, ncolors=cmap.N, clip=False)
    return cmap, norm, color_list


# -------------------------
# Overlay + legend (legend shows visible patches even if overlay uses transparent BG)
# -------------------------
def overlay_and_plot(image: np.ndarray,
                     mask: Optional[np.ndarray],
                     label_names: Optional[list] = None,
                     alpha: float = 0.6,
                     outline: bool = False,
                     save_path: Optional[Path] = None,
                     show: bool = True,
                     user_colors: Optional[dict] = None,
                     class_labels: Optional[dict] = None,
                     num_classes: Optional[int] = None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    vmin, vmax = np.percentile(image, (1, 99))
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.axis("off")

    if mask is not None:
        # Build colormap and overlay
        cmap, norm, color_list = build_colormap_fixed(num_classes=num_classes, user_colors=user_colors, labels_present=mask)
        ax.imshow(mask, cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest")

        if outline:
            try:
                import skimage.measure as measure
                max_label = int(np.nanmax(mask))
                for lab in range(1, max_label + 1):
                    lab_mask = (mask == lab).astype(np.uint8)
                    if lab_mask.sum() == 0:
                        continue
                    contours = measure.find_contours(lab_mask, 0.5)
                    for contour in contours:
                        ax.plot(contour[:, 1], contour[:, 0], linewidth=1.0, color='white')
            except Exception:
                max_label = int(np.nanmax(mask))
                for lab in range(1, max_label + 1):
                    cs = (mask == lab).astype(float)
                    ax.contour(cs, levels=[0.5], colors=['white'], linewidths=0.6)

        # Prepare legend
        if class_labels is None:
            class_labels = {i: f"class_{i}" for i in range(len(color_list))}
        if label_names is not None:
            for i, nm in enumerate(label_names):
                class_labels[i] = nm

        # Build legend handles for indices 0..n-1
        handles = []
        n = len(color_list)
        for lab in range(n):
            name = class_labels.get(lab, f"class_{lab}")
            rgba = color_list[lab] if lab < len(color_list) else (0.5, 0.5, 0.5, 1.0)
            legend_rgba = (rgba[0], rgba[1], rgba[2], 1.0)  # force visible alpha for legend
            patch = mpatches.Patch(facecolor=legend_rgba, edgecolor='k', label=f"{lab}: {name}")
            handles.append(patch)

        if handles:
            ax.legend(handles=handles,
                      loc="upper right",
                      bbox_to_anchor=(0.98, 0.98),
                      framealpha=0.9,
                      fontsize="small",
                      borderpad=0.3,
                      handlelength=1.0)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), bbox_inches="tight", dpi=200)
        print(f"Saved overlay to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Overlay a GT mask on top of a cardiac image slice")
    parser.add_argument("--image", required=True, help="Path to image NIfTI (2D/3D/4D)")
    parser.add_argument("--mask", required=True, help="Path to mask NIfTI (2D/3D/4D)")
    parser.add_argument("--t", type=int, default=None, help="Time/frame index to display (0-indexed). If omitted, script will guess or use 0.")
    parser.add_argument("--z", type=int, default=None, help="Slice (z) index to display (0-indexed). If omitted, script will guess or use middle slice.")
    parser.add_argument("--out", type=str, default=None, help="Path to save PNG (optional)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha for masks (0.0-1.0)")
    parser.add_argument("--outline", action="store_true", help="Overlay contour outlines for each label")
    parser.add_argument("--label-names", type=str, default=None, help="Comma-separated label names starting from index 0 (BG). Example: BG,RV,MYO,LV")
    parser.add_argument("--no-show", action="store_true", help="Don't show interactive window (useful for headless runs)")
    parser.add_argument("--flip180", action="store_true", help="Rotate both image and mask by 180 degrees before plotting/saving")
    parser.add_argument("--num-classes", type=int, default=None, help="Force number of classes (makes legend show indices up to this value)")
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_path = Path(args.mask)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    label_names = None
    if args.label_names:
        label_names = [n.strip() for n in args.label_names.split(",")]

    img_slice, mask_slice, chosen_t, chosen_z = load_and_select(image_path, mask_path, args.t, args.z)
    print(f"Using t={chosen_t}, z={chosen_z} (image dims: {img_slice.shape})")

    if args.flip180:
        img_slice = np.rot90(img_slice, 2)
        if mask_slice is not None:
            mask_slice = np.rot90(mask_slice, 2)
        print("Applied 180-degree rotation to image and mask (flip180).")

    overlay_and_plot(img_slice,
                     mask_slice,
                     label_names,
                     alpha=args.alpha,
                     outline=args.outline,
                     save_path=Path(args.out) if args.out else None,
                     show=not args.no_show,
                     user_colors=CLASS_COLORS,
                     class_labels=CLASS_LABELS,
                     num_classes=args.num_classes)

if __name__ == "__main__":
    main()
