#!/usr/bin/env python3
"""
merge_gt_and_pred.py

Load a ground-truth image+mask frame (per-frame .nii.gz) and run inference on the
corresponding 4D cine to produce a prediction for the same t,z. Save a side-by-side
PNG: left = GT overlay, right = Prediction overlay. Also saves predicted mask .npy.

Features:
 - Canonicalizes orientation via nibabel.as_closest_canonical() and reorders arrays to (T,Z,Y,X).
 - Robust checkpoint loading (strips prefixes, tries strict/non-strict loads).
 - Centered pad/crop to make GT image, GT mask and predicted mask share the same 2D shape.
 - Optional automatic alignment of predicted mask to GT mask using a small set of 2D transforms
   (rotations + optional horizontal flip) that maximizes Dice with GT.
 - Deterministic colors and legend as requested.

Example:
  python -m twin_core.utils.merge_gt_and_pred \
    --data_root "C:/.../merged_dataset_150" \
    --checkpoint "checkpoints/ckpt_best.pt" \
    --patient 9 --frame 13 \
    --z 5 --out_dir outputs/merge --device cpu

"""
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
import sys
import nibabel as nib
from typing import Optional, Tuple, List
import time

# Import model (must be importable in your environment)
from twin_core.utils.UNET_model import UNet

timestr = time.strftime("%Y%m%dT%H%M%S")

# Deterministic class labels and colors
CLASS_LABELS = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
DEFAULT_CLASS_COLORS = {
    0: (0.0, 0.0, 0.0, 0.0),     # BG transparent
    1: "#ffd700",                # RV - yellow
    2: "#4daf4a",                # MYO - green
    3: "#ff0000",                # LV - red
}


# -------------------------
# NIfTI loading and canonicalization helpers
# -------------------------
def reorder_to_tzyx_from_nib(nib_obj: nib.Nifti1Image) -> np.ndarray:
    """
    Canonicalize via nib.as_closest_canonical and return array shaped (T,Z,Y,X).
    Works for 2D/3D/4D common cases.
    """
    try:
        img_c = nib.as_closest_canonical(nib_obj)
        arr = img_c.get_fdata()
    except Exception:
        arr = nib_obj.get_fdata()
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis, :, :]
    if arr.ndim == 3:
        # common case (X,Y,Z) -> (1,Z,Y,X)
        return np.transpose(arr, (2, 1, 0))[None]
    if arr.ndim == 4:
        # try common (X,Y,Z,T) -> (T,Z,Y,X)
        shape = arr.shape
        if shape[-1] > 1 and shape[-1] <= max(shape[0], shape[1], shape[2]):
            arr_tfirst = np.moveaxis(arr, -1, 0)
            try:
                return np.transpose(arr_tfirst, (0, 3, 2, 1))
            except Exception:
                return arr_tfirst
        # fallback: (T,X,Y,Z) or other -> try to get (T,Z,Y,X)
        try:
            return np.transpose(arr, (0, 3, 2, 1))
        except Exception:
            return np.moveaxis(arr, -1, 0)
    raise RuntimeError(f"Unsupported array ndim: {arr.ndim}")


def load_nifti_as_tzyx(path: Path) -> np.ndarray:
    nb = nib.load(str(path))
    return reorder_to_tzyx_from_nib(nb)


# -------------------------
# Small image utilities
# -------------------------
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


def center_pad_or_crop_to(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Center-crop or center-pad `img` to `target_shape` (H, W).
    Deterministic and symmetric.
    """
    h, w = img.shape
    th, tw = target_shape
    # crop if needed
    if h >= th:
        start_h = (h - th) // 2
        img_cropped_h = img[start_h:start_h + th, :]
    else:
        # pad vertically
        pad_top = (th - h) // 2
        pad_bot = th - h - pad_top
        img = np.pad(img, ((pad_top, pad_bot), (0, 0)), mode="constant", constant_values=0)
        img_cropped_h = img
    if img_cropped_h.shape[1] >= tw:
        start_w = (img_cropped_h.shape[1] - tw) // 2
        out = img_cropped_h[:, start_w:start_w + tw]
    else:
        pad_left = (tw - img_cropped_h.shape[1]) // 2
        pad_right = tw - img_cropped_h.shape[1] - pad_left
        out = np.pad(img_cropped_h, ((0, 0), (pad_left, pad_right)), mode="constant", constant_values=0)
    return out


def apply_minmax_or_zscore(img: np.ndarray, mode: str, clip_min=None, clip_max=None):
    s = img.astype(np.float32)
    if mode == "none":
        return s
    if mode == "minmax":
        mn = clip_min if clip_min is not None else float(s.min())
        mx = clip_max if clip_max is not None else float(s.max())
        return (s - mn) / (mx - mn) if mx > mn else s - mn
    if mode == "zscore":
        mu = float(s.mean()); sigma = float(s.std()) if float(s.std()) > 0 else 1.0
        return (s - mu) / sigma
    raise ValueError("unknown norm")


# -------------------------
# Checkpoint loader
# -------------------------
def extract_state_dict_from_checkpoint(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = None
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state", "model_state_dict", "model", "net", "state"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]; break
        if state is None and "checkpoint" in ckpt and isinstance(ckpt["checkpoint"], dict):
            for k in ("state_dict", "model_state_dict", "model"):
                if k in ckpt["checkpoint"]:
                    state = ckpt["checkpoint"][k]; break
    if state is None and isinstance(ckpt, dict) and all(isinstance(v, (torch.Tensor, np.ndarray)) for v in ckpt.values()):
        state = ckpt
    if state is None:
        if isinstance(ckpt, dict):
            for v in ckpt.values():
                if isinstance(v, dict) and all(isinstance(x, (torch.Tensor, np.ndarray)) for x in v.values()):
                    state = v; break
    if state is None:
        raise RuntimeError("No model weights found in checkpoint.")
    # strip prefixes
    new = {}
    for k, v in state.items():
        nk = k
        for p in ("module.", "model.", "net."):
            if nk.startswith(p):
                nk = nk[len(p):]
        new[nk] = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
    return new


# -------------------------
# Colormap / legend helpers
# -------------------------
def build_colormap_from_colors(colors_map: dict, max_label: int):
    """
    Build ListedColormap and BoundaryNorm given a mapping index->color (strings or rgba tuples)
    for labels 0..max_label inclusive.
    """
    color_list = []
    for i in range(max_label + 1):
        col = colors_map.get(i, None)
        if col is None:
            cmap_tab = plt.get_cmap("tab10")
            rgba = cmap_tab(i % 10)
        else:
            if isinstance(col, str):
                rgba = to_rgba(col)
            else:
                # allow tuple with 3 or 4 floats/ints
                try:
                    tup = tuple(float(x) for x in col)
                    if len(tup) == 3:
                        rgba = (tup[0], tup[1], tup[2], 1.0)
                    else:
                        rgba = tup
                except Exception:
                    cmap_tab = plt.get_cmap("tab10")
                    rgba = cmap_tab(i % 10)
        color_list.append(rgba)
    cmap = ListedColormap(color_list)
    bounds = np.arange(-0.5, max_label + 1 + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm, color_list


# -------------------------
# Simple Dice (binary) for alignment metric - applied per-class and averaged
# -------------------------
def dice_score_binary(a: np.ndarray, b: np.ndarray, eps: float = 1e-8):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = np.sum(a & b)
    denom = np.sum(a) + np.sum(b)
    if denom == 0:
        return 1.0  # both empty -> perfect
    return (2.0 * inter) / (denom + eps)


def multi_class_dice(a: np.ndarray, b: np.ndarray, num_classes: int):
    scores = []
    for c in range(1, num_classes):  # exclude background for alignment
        ac = (a == c).astype(np.uint8)
        bc = (b == c).astype(np.uint8)
        denom = ac.sum() + bc.sum()
        if denom == 0:
            continue
        inter = np.sum(ac & bc)
        scores.append((2.0 * inter) / (denom + 1e-8))
    if not scores:
        return 0.0
    return float(np.mean(scores))


# -------------------------
# Small set of 2D transforms for alignment
# rotations by 0/90/180/270 and optional horizontal flip -> 8 transforms
# -------------------------
def generate_transforms():
    transforms = []
    for k in (0, 1, 2, 3):  # rot90 k times
        transforms.append(("rot", k, False))
        transforms.append(("rot", k, True))  # rotated then hflip
    return transforms


def apply_transform_2d(img: np.ndarray, tr):
    kind, k, do_flip = tr
    out = np.rot90(img, k)
    if do_flip:
        out = np.fliplr(out)
    return out


def pick_best_transform_by_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Tuple[np.ndarray, float, tuple]:
    """
    Try small set of transforms on pred to maximize multi-class Dice vs gt.
    Returns transformed_pred, best_score, best_transform
    """
    best_score = -1.0
    best_pred = pred
    best_tr = None
    for tr in generate_transforms():
        cand = apply_transform_2d(pred, tr)
        # ensure shapes match (they should)
        if cand.shape != gt.shape:
            cand = center_pad_or_crop_to(cand, gt.shape)
        score = multi_class_dice(cand, gt, num_classes)
        if score > best_score:
            best_score = score
            best_pred = cand
            best_tr = tr
    return best_pred, best_score, best_tr


# -------------------------
# Visualization: side-by-side GT / Pred overlays
# -------------------------
def visualize_pair_and_save(image: np.ndarray, gt_labels: Optional[np.ndarray], pred_labels: Optional[np.ndarray],
                            out_path: Path, class_labels: dict = CLASS_LABELS, class_colors: dict = DEFAULT_CLASS_COLORS,
                            alpha: float = 0.6):
    max_label = 0
    if gt_labels is not None:
        max_label = max(max_label, int(gt_labels.max()))
    if pred_labels is not None:
        max_label = max(max_label, int(pred_labels.max()))

    cmap, norm, color_list = build_colormap_from_colors(class_colors, max_label)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 6))
    # left: GT overlay
    ax_l.imshow(image, cmap="gray")
    ax_l.axis("off")
    ax_l.set_title("Ground-truth overlay")
    if gt_labels is not None:
        ax_l.imshow(gt_labels, cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest")

    # right: Pred overlay (same image shown for fair visual comparison)
    ax_r.imshow(image, cmap="gray")
    ax_r.axis("off")
    ax_r.set_title("Prediction overlay")
    if pred_labels is not None:
        ax_r.imshow(pred_labels, cmap=cmap, norm=norm, alpha=alpha, interpolation="nearest")

    # Legend: show labels that are present in either GT or pred (force visible alpha)
    present = set()
    if gt_labels is not None:
        present.update(int(x) for x in np.unique(gt_labels.ravel()))
    if pred_labels is not None:
        present.update(int(x) for x in np.unique(pred_labels.ravel()))
    present = sorted(present)

    handles = []
    for lab in present:
        name = class_labels.get(lab, f"class_{lab}")
        col = color_list[lab] if lab < len(color_list) else (0.5, 0.5, 0.5, 1.0)
        legend_rgba = (col[0], col[1], col[2], 1.0)
        handles.append(mpatches.Patch(facecolor=legend_rgba, edgecolor="k", label=f"{lab}: {name}"))
    if handles:
        ax_r.legend(handles=handles,
                    loc="upper right",
                    bbox_to_anchor=(0.98, 0.98),
                    framealpha=0.9,
                    fontsize="small",
                    borderpad=0.3,
                    handlelength=1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)


# -------------------------
# Main pipeline
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Merge GT overlay and predicted overlay for a given patient/frame/t/z")
    p.add_argument("--data_root", type=str, default=None, help="Optional base dataset root (contains training/). Used when --patient/--frame are given.")
    p.add_argument("--patient", type=int, default=None, help="Patient number (e.g. 1 -> patient001). If given will auto-build paths when image/mask/nifti not provided.")
    p.add_argument("--frame", type=int, default=None, help="Frame number as in file name (1-indexed). E.g. frame12 -> pass 12")
    p.add_argument("--image", type=str, default=None, help="Explicit path to per-frame image (patientXXX_frameYY.nii.gz)")
    p.add_argument("--mask", type=str, default=None, help="Explicit path to per-frame GT mask")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    p.add_argument("--nifti", type=str, default=None, help="Path to 4D cine NIfTI (patientXXX_4d.nii.gz) for inference")
    p.add_argument("--t", type=int, default=None, help="Time index for inference (0-based). If omitted and frame provided, uses frame-1")
    p.add_argument("--z", type=int, required=True, help="Slice z index (0-based) to display and infer for")
    p.add_argument("--out_dir", type=str, default="outputs/merge", help="Output dir")
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"), help="Device for inference")
    p.add_argument("--pad_multiple", type=int, default=16, help="Pad H/W to multiple for model input")
    p.add_argument("--normalization", type=str, default="minmax", choices=("none", "minmax", "zscore"), help="Normalization mode used for model input")
    p.add_argument("--clip_min", type=float, default=None)
    p.add_argument("--clip_max", type=float, default=None)
    p.add_argument("--align_pred_to_gt", action="store_true", default=True, help="Try small 2D transforms on prediction to maximize Dice vs GT (useful if residual orientation mismatch exists).")
    p.add_argument("--no-save-npy", action="store_true", default=False, help="Don't save predicted mask .npy (default saves it).")
    args = p.parse_args()

    # Resolve patient-based paths
    image_path = Path(args.image) if args.image else None
    mask_path = Path(args.mask) if args.mask else None
    nifti_path = Path(args.nifti) if args.nifti else None

    pid = f"patient{int(args.patient):03d}" if (args.patient is not None) else None
    frame_idx = int(args.frame) if (args.frame is not None) else None

    if (image_path is None) or (mask_path is None) or (nifti_path is None):
        if args.data_root is None or pid is None or frame_idx is None:
            missing = []
            if image_path is None: missing.append("--image or --patient+--frame")
            if mask_path is None: missing.append("--mask or --patient+--frame")
            if nifti_path is None: missing.append("--nifti or --patient+--frame")
            raise RuntimeError(f"Missing inputs: provide explicit paths or supply --data_root/--patient/--frame. Missing: {', '.join(missing)}")
        base = Path(args.data_root) / "training" / pid
        if image_path is None:
            image_path = base / f"{pid}_frame{int(frame_idx):02d}.nii.gz"
        if mask_path is None:
            mask_path = base / f"{pid}_frame{int(frame_idx):02d}_gt.nii.gz"
        if nifti_path is None:
            nifti_path = base / f"{pid}_4d.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if not nifti_path.exists():
        raise FileNotFoundError(f"4D nifti for inference not found: {nifti_path}")

    # infer t for inference
    if args.t is not None:
        infer_t = int(args.t)
    elif frame_idx is not None:
        infer_t = frame_idx - 1
    else:
        raise RuntimeError("Either --t or --frame must be provided to determine inference time index.")

    z = int(args.z)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load GT image+mask slices (canonicalized)
    img_tzyx = load_nifti_as_tzyx(image_path)
    # If the per-frame image file is single-frame (T=1) and the filename denotes frame number,
    # select chosen_t inferred from frame or 0 if not provided in file.
    gt_img_slice, chosen_t_img, chosen_z_img = None, None, None
    try:
        gt_img_slice, chosen_t_img, chosen_z_img = (lambda arr, t_arg, z_arg: (arr[t_arg if t_arg is not None else 0, z_arg if z_arg is not None else arr.shape[1] // 2], (t_arg if t_arg is not None else 0), (z_arg if z_arg is not None else arr.shape[1] // 2)))(img_tzyx, None, z)
    except Exception:
        raise RuntimeError("Failed to select GT image slice.")

    mask_tzyx = load_nifti_as_tzyx(mask_path)
    gt_mask_slice, _, _ = (lambda arr, t_arg, z_arg: (arr[t_arg if t_arg is not None else 0, z_arg if z_arg is not None else arr.shape[1] // 2], (t_arg if t_arg is not None else 0), (z_arg if z_arg is not None else arr.shape[1] // 2)))(mask_tzyx, None, z)
    gt_mask_slice = np.asarray(gt_mask_slice).astype(np.int64)

    # Ensure GT image and GT mask share same 2D shape; center-pad/crop deterministically if not
    if gt_img_slice.shape != gt_mask_slice.shape:
        gt_mask_slice = center_pad_or_crop_to(gt_mask_slice, gt_img_slice.shape)

    # --- Run inference on 4D cine to get predicted mask for (infer_t, z)
    cine_tzyx = load_nifti_as_tzyx(nifti_path)
    T_c, Z_c, Y_c, X_c = cine_tzyx.shape
    if infer_t < 0 or infer_t >= T_c:
        raise ValueError(f"inference t={infer_t} out of range for {nifti_path} with T={T_c}")
    if z < 0 or z >= Z_c:
        raise ValueError(f"z={z} out of range for {nifti_path} with Z={Z_c}")

    slice_for_model = cine_tzyx[infer_t, z].astype(np.float32)

    # normalization + pad for model input
    slice_norm = apply_minmax_or_zscore(slice_for_model, args.normalization, args.clip_min, args.clip_max)
    padded, pad_info = pad_to_multiple(slice_norm, multiple=int(args.pad_multiple))
    x = torch.from_numpy(padded.astype(np.float32))[None, None, ...]
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    x = x.to(device)

    # prepare model
    n_classes = max(CLASS_LABELS.keys()) + 1
    model = UNet(in_channels=1, out_channels=int(n_classes))
    model.to(device)
    state_for_load = extract_state_dict_from_checkpoint(Path(args.checkpoint))
    try:
        model.load_state_dict(state_for_load)
    except Exception:
        model.load_state_dict(state_for_load, strict=False)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        if logits.ndim == 3:
            logits = logits.unsqueeze(0)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_unpadded = unpad(pred, pad_info)

    # Align pred mask spatial size to gt image
    if pred_unpadded.shape != gt_img_slice.shape:
        pred_unpadded = center_pad_or_crop_to(pred_unpadded, gt_img_slice.shape)

    # Optional: try transforms on pred mask to maximize Dice vs GT (useful if residual orientation mismatch)
    if args.align_pred_to_gt and (gt_mask_slice is not None):
        try:
            best_pred, best_score, best_tr = pick_best_transform_by_dice(pred_unpadded, gt_mask_slice, num_classes=int(n_classes))
            # Only accept transform if it improves over identity
            identity_score = multi_class_dice(pred_unpadded, gt_mask_slice, num_classes=int(n_classes))
            if best_score >= identity_score:
                pred_unpadded = best_pred
                if best_tr is not None:
                    print(f"Applied best transform {best_tr} to prediction (dice improved {identity_score:.4f} -> {best_score:.4f}).")
            else:
                print(f"No transform improved Dice (identity {identity_score:.4f} >= best {best_score:.4f}); leaving prediction as-is.")
        except Exception as e:
            print(f"Warning: alignment attempt failed: {e}. Proceeding without transform.")

    # Save predicted mask .npy
    # Use patient id resolved from --patient or nifti parent folder
    pid_for_name = f"patient{int(args.patient):03d}" if (args.patient is not None) else nifti_path.parent.name
    pred_npy = Path(args.out_dir) / f"pred_{pid_for_name}_frame{frame_idx if frame_idx is not None else infer_t+1}_t{infer_t}_z{z}_{timestr}.npy"
    if not args.no_save_npy:
        np.save(str(pred_npy), pred_unpadded)
        print("Saved predicted mask (.npy):", pred_npy)

    # Build output PNG filename with patient name, frame (1-indexed), t (frame-1), z and timestamp
    frame_for_name = frame_idx if frame_idx is not None else (infer_t + 1)
    out_png = Path(args.out_dir) / f"merge_{pid_for_name}_frame{int(frame_for_name)}_t{int(infer_t)}_z{int(z)}_{timestr}.png"

    # Visualize with deterministic colors & legend
    visualize_pair_and_save(gt_img_slice, gt_mask_slice, pred_unpadded, out_png,
                            class_labels=CLASS_LABELS, class_colors=DEFAULT_CLASS_COLORS, alpha=0.6)

    print("Saved merged overlay PNG:", out_png)


if __name__ == "__main__":
    main()
