#!/usr/bin/env python3
"""
scripts/compare_slice.py

CLI to run inference for a single slice (or indexed sample) and compare prediction
against a ground-truth mask. Produces:
 - per-class Dice scores (printed and appended to CSV)
 - optional PNG side-by-side visualization (image / GT / prediction / diff)

Usage examples:
  python -m scripts.compare_slice --checkpoint checkpoints/ckpt_best.pt \
    --nifti "data/patient002_4d.nii.gz" --patient patient002 --t 0 --z 12 \
    --out_dir outputs/compare --device cuda --save_png

  python -m scripts.compare_slice --checkpoint checkpoints/ckpt_best.pt \
    --preprocessed "preprocessed/patient002" --idx 5 --out_dir outputs/compare --save_png

Notes:
 - If GT masks are stored in a preprocessed layout, the script will attempt to
   locate them automatically by replacing "data" with "masks" in the image path,
   or by searching the provided --gt_root. You can also pass --gt_root explicitly.
 - The script pads/unpads masks using the same pad_to_target_2d helper used in training.
"""

from pathlib import Path
import argparse
import csv
import sys
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt

# Local imports from your repo
from twin_core.utils.inference_dataset import InferenceDataset
from twin_core.utils.segmentation_model import load_model

# Try to import repo metrics and pad helper; provide safe fallbacks if missing
try:
    from twin_core.utils.metrics import dice_per_class as repo_dice_per_class
except Exception:
    repo_dice_per_class = None

try:
    from twin_core.data_ingestion.dataset import pad_to_target_2d
except Exception:
    # Minimal fallback (keeps behaviour compatible with InferenceDataset fallback)
    def pad_to_target_2d(arr: np.ndarray, target_h: int, target_w: int, mode: str = "constant", cval: float = 0.0) -> np.ndarray:
        h, w = arr.shape
        if h > target_h or w > target_w:
            raise ValueError(f"pad_to_target_2d: array {(h,w)} exceeds target {(target_h,target_w)}")
        pad_h = target_h - h
        pad_w = target_w - w
        pad_top = 0
        pad_bottom = pad_h
        pad_left = 0
        pad_right = pad_w
        if mode == "constant":
            padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=cval)
        else:
            padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode)
        return padded


def dice_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """
    Compute per-class Dice coefficient (not loss) between integer label arrays.
    Returns numpy array shape (num_classes,) with values in [0,1].
    """
    if repo_dice_per_class is not None:
        try:
            return np.asarray(repo_dice_per_class(pred, target, num_classes=num_classes))
        except Exception:
            pass

    scores = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        p = (pred == c).astype(np.uint8)
        t = (target == c).astype(np.uint8)
        inter = (p & t).sum()
        denom = p.sum() + t.sum()
        if denom == 0:
            scores[c] = 1.0  # absent in both -> perfect
        else:
            scores[c] = (2.0 * inter) / float(denom)
    return scores


def find_index_for(ds: InferenceDataset, patient_id=None, t=None, z=None):
    """
    Find the first sample index in dataset matching patient_id, t and z.
    """
    # Prefer scanning ds.samples if available (cheap)
    samples = getattr(ds, "samples", None)
    if samples is not None:
        for i, s in enumerate(samples):
            pid = s.get("patient_id")
            st = s.get("t")
            sz = s.get("z")
            if (patient_id is None or pid == patient_id) and (t is None or st == t) and (z is None or sz == z):
                return i

    # Fallback to iterating __getitem__ (dataset may be small)
    for i in range(len(ds)):
        _, meta = ds[i]
        if (patient_id is None or meta.get("patient_id") == patient_id) and (t is None or meta.get("t") == t) and (z is None or meta.get("z") == z):
            return i
    raise ValueError("No matching sample found for the given patient/t/z")


def _guess_gt_path_from_meta(meta: dict, gt_root: Path = None) -> Path | None:
    """
    Try to guess a ground-truth mask path from dataset meta.
    Strategies:
      - If meta['image_path'] exists and contains 'data', replace with 'masks' and try common extensions (.npy, .nii.gz, .nii).
      - If gt_root provided, search gt_root/<patient_id>/masks for files containing t or z indices.
      - If meta indicates nifti source and gt_root is a nifti file, try to load that nifti and extract slice.
    Returns Path to mask file if found, else None.
    """
    pid = meta.get("patient_id")
    img_path = meta.get("image_path")
    t = meta.get("t")
    z = meta.get("z")

    candidates = []

    if img_path:
        p = Path(img_path)
        # Replace 'data' with 'masks' in path if present
        parts = list(p.parts)
        try:
            idx = parts.index("data")
            parts[idx] = "masks"
            mask_candidate = Path(*parts)
            candidates.append(mask_candidate)
        except ValueError:
            # try sibling folder 'masks' next to 'data' parent
            if p.parent.name == "data":
                candidates.append(p.parent.parent / "masks" / p.name)

        # also try same filename with mask suffix
        stem = p.stem
        for ext in (".npy", ".nii.gz", ".nii"):
            candidates.append(p.with_suffix(ext))
            candidates.append(p.parent / (stem + "_mask" + ext))
            candidates.append(p.parent / (stem + "-mask" + ext))
            candidates.append(p.parent / (stem + ".mask" + ext))

    # If gt_root provided, search for likely files
    if gt_root is not None:
        gt_root = Path(gt_root)
        # patient subfolder
        if (gt_root / pid).exists():
            search_root = gt_root / pid
        else:
            search_root = gt_root
        # look for npy or nifti files in masks subfolder
        for candidate in search_root.rglob("*.npy"):
            name = candidate.name.lower()
            if pid and pid in name:
                candidates.append(candidate)
            if t is not None and f"t{int(t):02d}" in name:
                candidates.append(candidate)
            if z is not None and f"z{int(z):02d}" in name:
                candidates.append(candidate)
        for candidate in search_root.rglob("*.nii*"):
            candidates.append(candidate)

    # Try candidates in order and return first that exists
    for c in candidates:
        if c.exists():
            return c

    return None


def load_mask_from_path(mask_path: Path, meta: dict, target_h: int, target_w: int) -> np.ndarray:
    """
    Load mask from .npy or NIfTI and pad to target size.
    If mask_path is a 3D/4D volume, extract slice using meta['t'] and meta['z'] when available.
    """
    if mask_path.suffix.lower() == ".npy":
        arr = np.load(str(mask_path))
    else:
        # try nibabel or SimpleITK for nifti
        try:
            import SimpleITK as sitk
            img = sitk.ReadImage(str(mask_path))
            arr = sitk.GetArrayFromImage(img)  # likely (T,Z,Y,X) or (Z,Y,X)
        except Exception:
            try:
                import nibabel as nib
                nb = nib.load(str(mask_path))
                arr = nb.get_fdata()
            except Exception:
                raise RuntimeError("Unable to load mask file: " + str(mask_path))

    arr = np.asarray(arr)
    # If arr is 3D or 4D, try to index using meta
    if arr.ndim == 4:
        # assume (T,Z,Y,X) or (Z,Y,X,T) - try common shapes
        # prefer (T,Z,Y,X)
        t = meta.get("t")
        z = meta.get("z")
        if t is not None and z is not None and arr.shape[0] > t and arr.shape[1] > z:
            slice2d = arr[int(t), int(z)]
        elif arr.shape[-1] > t and arr.shape[-2] > z:
            # maybe (Y,X,Z,T) etc - try transpose
            slice2d = np.transpose(arr, (3, 2, 1, 0))[int(t), int(z)]
        else:
            # fallback: take first available 2D
            slice2d = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
    elif arr.ndim == 3:
        # assume (Z,Y,X) or (T,Y,X)
        t = meta.get("t")
        z = meta.get("z")
        if z is not None and arr.shape[0] > z:
            slice2d = arr[int(z)]
        elif t is not None and arr.shape[0] > t:
            slice2d = arr[int(t)]
        else:
            slice2d = arr[0]
    elif arr.ndim == 2:
        slice2d = arr
    else:
        raise ValueError("Unsupported mask array shape: " + str(arr.shape))

    slice2d = np.asarray(slice2d).astype(np.int64)
    # pad to target
    padded = pad_to_target_2d(slice2d, target_h=target_h, target_w=target_w, mode="constant", cval=0)
    return padded


def visualize_and_save(image_orig: np.ndarray, gt: np.ndarray | None, pred: np.ndarray, out_png: Path):
    """
    Save a 1x4 figure: original image, GT overlay, prediction overlay, difference mask.
    image_orig: unpadded original image (H,W) float
    gt/pred: padded arrays (target_h, target_w) integer labels
    """
    # Ensure arrays are same shape
    H, W = pred.shape
    img = image_orig
    # If image_orig is smaller, pad for display
    if img.shape != pred.shape:
        img = pad_to_target_2d(img, target_h=pred.shape[0], target_w=pred.shape[1], mode="constant", cval=0.0)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    if gt is not None:
        axes[1].imshow(img, cmap="gray")
        axes[1].imshow(gt, cmap="tab10", alpha=0.6, vmin=0, vmax=9)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
    else:
        axes[1].text(0.5, 0.5, "GT not found", ha="center", va="center")
        axes[1].axis("off")

    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(pred, cmap="tab10", alpha=0.6, vmin=0, vmax=9)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    if gt is not None:
        diff = (pred != gt).astype(np.uint8)
        axes[3].imshow(diff, cmap="gray")
        axes[3].set_title("Difference (1=mismatch)")
    else:
        axes[3].text(0.5, 0.5, "No GT", ha="center", va="center")
        axes[3].axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare model prediction for a single slice against GT")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (state_dict or Lightning ckpt)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nifti", type=str, help="Path to 4D cine NIfTI (T,Z,Y,X)")
    group.add_argument("--preprocessed", type=str, help="Path to preprocessed patient folder or root")
    parser.add_argument("--idx", type=int, default=None, help="Direct sample index in InferenceDataset")
    parser.add_argument("--patient", type=str, default=None, help="Patient id to match (optional)")
    parser.add_argument("--t", type=int, default=None, help="Time/frame index to match (optional)")
    parser.add_argument("--z", type=int, default=None, help="Slice index to match (optional)")
    parser.add_argument("--gt_root", type=str, default=None, help="Optional root folder to search for GT masks")
    parser.add_argument("--out_dir", type=str, default="outputs/compare", help="Directory to write outputs (CSV, PNG)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu or cuda)")
    parser.add_argument("--save_png", action="store_true", help="Save side-by-side PNG visualization")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively (requires display)")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of segmentation classes")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "compare_metrics.csv"
    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"

    # Build dataset
    if args.nifti:
        ds = InferenceDataset(nifti_path=Path(args.nifti), pad_multiple=16)
    else:
        ds = InferenceDataset(preprocessed_root=Path(args.preprocessed), pad_multiple=16, patient_id=args.patient)

    # Determine sample index
    if args.idx is not None:
        idx = int(args.idx)
        if idx < 0 or idx >= len(ds):
            raise IndexError(f"Index {idx} out of range (0..{len(ds)-1})")
    else:
        idx = find_index_for(ds, patient_id=args.patient, t=args.t, z=args.z)

    # Load image tensor and meta
    image_tensor, meta = ds[idx]  # (1, H_p, W_p)
    # image_tensor is channel-first (1,H,W) as float32
    x = image_tensor.unsqueeze(0).to(device)  # (1,1,H,W)

    # Load model
    model = load_model(checkpoint, device=device, n_classes=args.num_classes)
    model.eval()

    # Inference
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H_p, W_p)

    # Load GT if possible
    gt_path = None
    if args.gt_root:
        gt_path = _guess_gt_path_from_meta(meta, gt_root=Path(args.gt_root))
    else:
        gt_path = _guess_gt_path_from_meta(meta, gt_root=None)

    gt_padded = None
    if gt_path is not None:
        try:
            gt_padded = load_mask_from_path(gt_path, meta, target_h=ds.target_h, target_w=ds.target_w)
        except Exception as e:
            warnings.warn(f"Failed to load GT mask from {gt_path}: {e}")
            gt_padded = None

    # If GT not found and dataset is preprocessed and meta contains image_path,
    # try to derive mask path by replacing 'data' with 'masks' (best-effort)
    if gt_padded is None and meta.get("image_path"):
        try:
            candidate = _guess_gt_path_from_meta(meta, gt_root=None)
            if candidate is not None and candidate.exists():
                gt_padded = load_mask_from_path(candidate, meta, target_h=ds.target_h, target_w=ds.target_w)
                gt_path = candidate
        except Exception:
            pass

    # If still no GT, warn and proceed (we will still save prediction)
    if gt_padded is None:
        warnings.warn("Ground-truth mask not found. Only prediction will be saved and visualized.")

    # Compute per-class Dice (on padded arrays)
    per_class = None
    if gt_padded is not None:
        per_class = dice_per_class(pred, gt_padded, num_classes=args.num_classes)
    else:
        per_class = np.full((args.num_classes,), np.nan, dtype=np.float32)

    # Prepare CSV row
    row = {
        "sample_idx": idx,
        "patient_id": meta.get("patient_id"),
        "t": meta.get("t"),
        "z": meta.get("z"),
        "gt_path": str(gt_path) if gt_path is not None else "",
        "pred_path": "",
    }
    # Append per-class dice
    for ci in range(args.num_classes):
        row[f"dice_class_{ci}"] = float(per_class[ci]) if not np.isnan(per_class[ci]) else ""

    # Save prediction as npy
    pred_out = out_dir / f"pred_idx{idx}_pid{row['patient_id']}_t{row['t']}_z{row['z']}.npy"
    np.save(str(pred_out), pred)
    row["pred_path"] = str(pred_out)

    # Visualization
    if args.save_png:
        # get original unpadded image for display
        orig_img, _ = ds.get_original_slice(idx)
        png_out = out_dir / f"compare_idx{idx}_pid{row['patient_id']}_t{row['t']}_z{row['z']}.png"
        visualize_and_save(orig_img, gt_padded, pred, png_out)
        row["png_path"] = str(png_out)
        if args.show:
            # show inline
            img = plt.imread(str(png_out))
            plt.figure(figsize=(10, 4))
            plt.imshow(img)
            plt.axis("off")
            plt.show()

    # Append to CSV (create header if missing)
    header = ["sample_idx", "patient_id", "t", "z", "gt_path", "pred_path", "png_path"] + [f"dice_class_{i}" for i in range(args.num_classes)]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=header)
        if write_header:
            writer.writeheader()
        # ensure all header keys present in row
        out_row = {k: row.get(k, "") for k in header}
        writer.writerow(out_row)

    # Print summary
    print("Comparison complete")
    print(f"Sample idx: {idx}  patient: {row['patient_id']}  t: {row['t']}  z: {row['z']}")
    if gt_padded is not None:
        for ci, v in enumerate(per_class):
            print(f"  Dice class {ci}: {v:.4f}")
        mean_dice = float(np.nanmean(per_class[1:])) if args.num_classes > 1 else float(np.nanmean(per_class))
        print(f"  Mean Dice (exclude BG): {mean_dice:.4f}")
    else:
        print("  No GT available; prediction saved to:", pred_out)

    print("Outputs written to:", out_dir)


if __name__ == "__main__":
    main()
