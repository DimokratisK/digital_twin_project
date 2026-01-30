#!/usr/bin/env python3
"""
single_slice_inference.py

Single-slice inference (CLI):
 - accepts CLI args for checkpoint, nifti, t, z, out_dir, device, normalization
 - loads a NIfTI (3D or 4D), picks the requested slice (or auto-picks)
 - applies normalization, pads to multiple, runs UNet inference
 - saves .npy and PNG overlay with a per-class legend placed inside the overlay

Legend/colours:
 - Background (class 0) is plotted as transparent.
 - RV, MYO, LV get distinct, high-contrast colors suitable for overlays.
"""
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys
import time

# Default config (overridden by CLI)
DEFAULT_CHECKPOINT = Path("checkpoints/ckpt_best.pt")
DEFAULT_NIFTI = Path("patient_4d.nii.gz")
DEFAULT_MODEL_KWARGS = {"in_channels": 1, "out_channels": 4}
DEFAULT_NORMALIZATION = "minmax"  # "minmax", "zscore", "none"
DEFAULT_T = None
DEFAULT_Z = None
DEFAULT_PAD_MULTIPLE = 16
DEFAULT_DEVICE = "cuda"
DEFAULT_OUT_DIR = Path("outputs/inference")

# Import model
from twin_core.utils.UNET_model import UNet

# Default human-readable class labels (index -> name)
CLASS_LABELS = {
    0: "BG",
    1: "RV",
    2: "MYO",
    3: "LV",
}

timestr = time.strftime("%Y%m%d-%H%M%S")

# Default colors for classes (index -> rgba hex). Background set to fully transparent.
# You can edit these to taste. They map index->color, so label values align to colors.
DEFAULT_CLASS_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),     # BG transparent
    1: "#ffd700",                # RV - yellow
    2: "#4daf4a",                # MYO - green
    3: "#ff0000",                # LV - red
}


def parse_args():
    p = argparse.ArgumentParser(description="Single-slice inference with in-overlay legend")
    p.add_argument("--checkpoint", "-c", type=Path, default=DEFAULT_CHECKPOINT, help="Path to checkpoint (.pt)")
    p.add_argument("--nifti", "-n", type=Path, default=DEFAULT_NIFTI, help="Path to input NIfTI (3D or 4D)")
    p.add_argument("--t", type=int, default=DEFAULT_T, help="Time/frame index (0-based). If omitted, auto-pick.")
    p.add_argument("--z", type=int, default=DEFAULT_Z, help="Slice index (z) (0-based). If omitted, auto-pick.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    p.add_argument("--device", type=str, default=DEFAULT_DEVICE, choices=("cpu", "cuda"), help="Device")
    p.add_argument("--normalization", type=str, default=DEFAULT_NORMALIZATION, choices=("none", "minmax", "zscore"), help="Normalization mode")
    p.add_argument("--clip_min", type=float, default=None, help="Optional clip min for minmax")
    p.add_argument("--clip_max", type=float, default=None, help="Optional clip max for minmax")
    p.add_argument("--in_channels", type=int, default=DEFAULT_MODEL_KWARGS["in_channels"], help="Model in_channels")
    p.add_argument("--out_channels", type=int, default=DEFAULT_MODEL_KWARGS["out_channels"], help="Model out_channels (num classes)")
    p.add_argument("--pad_multiple", type=int, default=DEFAULT_PAD_MULTIPLE, help="Pad H/W to multiples of this for model input")
    return p.parse_args()


def load_nifti_array(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # SimpleITK -> (T?, Z?, Y, X) order
        return np.asarray(arr)
    except Exception:
        try:
            import nibabel as nib
            nb = nib.load(str(path))
            return np.asarray(nb.get_fdata())
        except Exception as e:
            raise RuntimeError(f"Failed to read NIfTI with SimpleITK and nibabel: {e}")


def normalize_to_tzyx(arr: np.ndarray) -> np.ndarray:
    # Convert many common layouts to (T,Z,Y,X)
    if arr.ndim == 4:
        # If arr looks like (T,Z,Y,X) already (reasonable image plane sizes), assume it's correct.
        # Otherwise, move last axis to front then transpose to (T,Z,Y,X).
        if arr.shape[2] >= 8 and arr.shape[3] >= 8:
            return arr
        return np.transpose(arr, (3, 2, 1, 0))
    elif arr.ndim == 3:
        # (Z, Y, X) -> (1, Z, Y, X)
        return arr[np.newaxis, ...]
    elif arr.ndim == 2:
        # (Y, X) -> (1, 1, Y, X)
        return arr[np.newaxis, np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported array ndim: {arr.ndim}")


def pick_most_informative_slice(arr_tzyx: np.ndarray):
    T, Z, _, _ = arr_tzyx.shape
    best = (0, 0)
    best_count = -1
    for t in range(T):
        for z in range(Z):
            cnt = int((arr_tzyx[t, z] != 0).sum())
            if cnt > best_count:
                best_count = cnt
                best = (t, z)
    return best


def pad_to_multiple(img: np.ndarray, multiple: int = 16):
    h, w = img.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pt = pad_h // 2
    pb = pad_h - pt
    pl = pad_w // 2
    pr = pad_w - pl
    padded = np.pad(img, ((pt, pb), (pl, pr)), mode="constant", constant_values=0.0)
    return padded, (pt, pb, pl, pr)


def unpad(img: np.ndarray, pad):
    pt, pb, pl, pr = pad
    h_slice = slice(pt, None) if pb == 0 else slice(pt, -pb)
    w_slice = slice(pl, None) if pr == 0 else slice(pl, -pr)
    return img[h_slice, w_slice]


def apply_normalization(slice2d: np.ndarray, mode: str, clip_min, clip_max) -> np.ndarray:
    s = slice2d.astype(np.float32)
    if mode == "none":
        return s
    if mode == "minmax":
        mn = clip_min if clip_min is not None else float(s.min())
        mx = clip_max if clip_max is not None else float(s.max())
        if mx > mn:
            return (s - mn) / (mx - mn)
        return s - mn
    if mode == "zscore":
        mu = float(s.mean())
        sigma = float(s.std()) if float(s.std()) > 0 else 1.0
        return (s - mu) / sigma
    raise ValueError("Unknown normalization mode")


def extract_state_dict_from_checkpoint(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state", "model_state_dict", "model", "net", "state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        # Lightning / nested patterns
        if state is None:
            if "checkpoint" in ckpt and isinstance(ckpt["checkpoint"], dict):
                for k in ("state_dict", "model_state_dict", "model"):
                    if k in ckpt["checkpoint"]:
                        state = ckpt["checkpoint"][k]
                        break
    if state is None and isinstance(ckpt, dict) and all(isinstance(v, (torch.Tensor, np.ndarray)) for v in ckpt.values()):
        state = ckpt
    if state is None:
        # try find nested dict that looks like param mapping
        if isinstance(ckpt, dict):
            for v in ckpt.values():
                if isinstance(v, dict) and all(isinstance(x, (torch.Tensor, np.ndarray)) for x in v.values()):
                    state = v
                    break
    if state is None:
        raise RuntimeError(f"No model weights found in checkpoint. Top-level keys: {list(ckpt.keys())[:50]}")
    # strip common prefixes
    new = {}
    for k, v in state.items():
        nk = k
        for p in ("module.", "model.", "net."):
            if nk.startswith(p):
                nk = nk[len(p):]
        new[nk] = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
    return new


def build_colormap_for_labels(labels: np.ndarray, user_colors: dict = None):
    """
    Build a ListedColormap and BoundaryNorm mapping label integers to colors.
    - labels: 2D integer labels array
    - user_colors: dict {label: colorstring or rgba tuple}
    Returns (cmap, norm, color_list_for_legend)
    """
    max_label = int(labels.max()) if labels.size > 0 else 0
    n = max_label + 1
    color_list = []
    # Use user-provided colors where available, else fallback to default mapping/Tab10
    for i in range(n):
        if user_colors and i in user_colors:
            color = user_colors[i]
        else:
            color = DEFAULT_CLASS_COLORS.get(i, None)
        if color is None:
            # fallback to tab10
            cmap_tab = plt.get_cmap("tab10")
            color = cmap_tab(i % 10)
        # ensure RGBA tuple for transparency handling
        if isinstance(color, str):
            color_rgba = plt.colors.to_rgba(color) if hasattr(plt, "colors") else None
            if color_rgba is None:
                # fallback simple conversion
                color_rgba = tuple(plt.cm.tab10(i % 10))
        else:
            color_rgba = color
        color_list.append(color_rgba)
    cmap = ListedColormap(color_list)
    # boundaries should be at half-integers so each integer maps to a color bin
    bounds = np.arange(-0.5, n + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, color_list


def visualize_overlay_with_legend(image: np.ndarray, labels: np.ndarray, out_png: Path, class_labels: dict, class_colors: dict):
    """
    Left: grayscale image, Right: overlay with legend placed inside the overlay axes (upper-right).
    Legend uses small colored boxes matching the overlay.
    """
    cmap, norm, color_list = build_colormap_for_labels(labels, user_colors=class_colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_img, ax_overlay = axes

    ax_img.imshow(image, cmap="gray")
    ax_img.set_title("Image")
    ax_img.axis("off")

    ax_overlay.imshow(image, cmap="gray")
    # overlay labels using our listed colormap and boundary norm
    ax_overlay.imshow(labels, cmap=cmap, norm=norm, alpha=0.65, interpolation="nearest")
    ax_overlay.set_title("Prediction overlay")
    ax_overlay.axis("off")

    # Create legend entries in the same order as colors/labels present
    unique_labels = sorted(int(x) for x in np.unique(labels))
    handles = []
    for lab in unique_labels:
        # skip background (0) if transparent and not desired in legend? here we include BG but it's transparent
        name = class_labels.get(lab, f"Class {lab}")
        col = color_list[lab] if lab < len(color_list) else None
        # ensure visible legend patch even if background transparent: give border and small alpha
        patch = mpatches.Patch(color=col, label=f"{lab}: {name}", alpha=1.0)
        handles.append(patch)

    # Place legend inside overlay axes (upper right) with a translucent box
    if handles:
        # inside box: adjust fontsize and number of columns if many classes
        ax_overlay.legend(handles=handles,
                          loc="upper right",
                          bbox_to_anchor=(0.98, 0.98),
                          framealpha=0.9,
                          fontsize="small",
                          borderpad=0.3,
                          handlelength=1.0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    CHECKPOINT = Path(args.checkpoint)
    NIFTI = Path(args.nifti)
    T_INDEX = args.t
    Z_INDEX = args.z
    OUT_DIR = Path(args.out_dir)
    DEVICE = args.device
    NORMALIZATION = args.normalization
    CLIP_MIN = args.clip_min
    CLIP_MAX = args.clip_max
    MODEL_KWARGS = {"in_channels": args.in_channels, "out_channels": args.out_channels}
    PAD_MULTIPLE = int(args.pad_multiple)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.exists():
        print("Checkpoint not found:", CHECKPOINT)
        sys.exit(1)
    if not NIFTI.exists():
        print("NIfTI not found:", NIFTI)
        sys.exit(1)

    arr_raw = load_nifti_array(NIFTI)
    print("raw shape:", arr_raw.shape, "dtype:", arr_raw.dtype)

    arr = normalize_to_tzyx(arr_raw)
    print("interpreted (T,Z,Y,X):", arr.shape)
    T, Z, Y, X = arr.shape

    # choose slice
    t = T_INDEX if T_INDEX is not None else None
    z = Z_INDEX if Z_INDEX is not None else None
    if t is None or z is None:
        t_best, z_best = pick_most_informative_slice(arr)
        print("auto-picked slice:", (t_best, z_best))
        if t is None:
            t = t_best
        if z is None:
            z = z_best

    if t < 0 or t >= T:
        print("t out of range, using 0")
        t = 0
    if z < 0 or z >= Z:
        print(f"z out of range (0..{Z-1}), using 0")
        z = 0

    slice2d = arr[t, z].astype(np.float32)
    print(f"selected t={t}, z={z}, slice shape {slice2d.shape}")

    img_norm = apply_normalization(slice2d, NORMALIZATION, CLIP_MIN, CLIP_MAX)
    padded, pad_info = pad_to_multiple(img_norm, multiple=PAD_MULTIPLE)

    x = torch.from_numpy(padded.astype(np.float32))[None, None, ...]
    device = torch.device("cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    x = x.to(device)

    model = UNet(**MODEL_KWARGS)
    model.to(device)

    state_for_load = extract_state_dict_from_checkpoint(CHECKPOINT)
    try:
        model.load_state_dict(state_for_load)
        print("Loaded checkpoint (strict).")
    except Exception:
        model.load_state_dict(state_for_load, strict=False)
        print("Loaded checkpoint (non-strict).")

    model.eval()
    with torch.no_grad():
        logits = model(x)
        if logits.ndim == 3:
            logits = logits.unsqueeze(0)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_unpadded = unpad(pred, pad_info)
    out_npy = OUT_DIR / f"pred_t{t}_z{z}_{timestr}.npy"
    out_png = OUT_DIR / f"pred_t{t}_z{z}_{timestr}.png"
    np.save(str(out_npy), pred_unpadded)

    # Visualize using in-overlay legend and deterministic colors
    visualize_overlay_with_legend(slice2d, pred_unpadded, out_png, CLASS_LABELS, DEFAULT_CLASS_COLORS)

    print("Saved prediction .npy:", out_npy)
    print("Saved overlay PNG:", out_png)
    print("Unique labels in prediction:", np.unique(pred_unpadded))


if __name__ == "__main__":
    main()
