#!/usr/bin/env python3
"""
single_slice_inference.py

Single-slice inference that:
 - loads a NIfTI (3D or 4D)
 - picks a slice (auto or user-specified)
 - applies the chosen normalization
 - pads to a multiple (default 16)
 - imports the UNet class directly from twin_core.utils.UNET_model
 - extracts model weights from a training checkpoint (robust to common formats)
 - runs inference and saves integer mask (.npy) and overlay PNG

Edit only the CONFIG block below before running.
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

# -------------------------
# CONFIG (edit these)
# -------------------------
CHECKPOINT_PATH = Path("checkpoints/ckpt_best.pt")
NIFTI_PATH = Path(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\training\patient001\patient001_frame01.nii.gz")

from twin_core.utils.UNET_model import UNet

MODEL_KWARGS = {"in_channels": 1, "out_channels": 4}  # set to match training
NORMALIZATION = "minmax"   # "minmax", "zscore", or "none"
MEAN = 0.0                 # used if NORMALIZATION == "zscore"
STD = 1.0
CLIP_MIN = None            # optional for minmax
CLIP_MAX = None
# Choose slice: set Z_INDEX to an integer (0-based) to force a slice, or None to auto-pick
T_INDEX = 0
Z_INDEX = None
PAD_MULTIPLE = 16
DEVICE = "cuda"           # "cuda" or "cpu"
OUT_DIR = Path("outputs/inference")
# -------------------------

def load_nifti_array(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img)  # often (Z,Y,X) or (T,Z,Y,X)
        return np.asarray(arr)
    except Exception:
        import nibabel as nib
        nb = nib.load(str(path))
        return np.asarray(nb.get_fdata())

def normalize_to_tzyx(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4:
        # assume (T,Z,Y,X) or try common transpose if last dims small
        if arr.shape[2] >= 8 and arr.shape[3] >= 8:
            return arr
        return np.transpose(arr, (3,2,1,0))
    elif arr.ndim == 3:
        return arr[np.newaxis, ...]
    elif arr.ndim == 2:
        return arr[np.newaxis, np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported array ndim: {arr.ndim}")

def pick_most_informative_slice(arr_tzyx: np.ndarray):
    T,Z,_,_ = arr_tzyx.shape
    best = (0,0); best_count = -1
    for t in range(T):
        for z in range(Z):
            cnt = int((arr_tzyx[t,z] != 0).sum())
            if cnt > best_count:
                best_count = cnt; best = (t,z)
    return best

def pad_to_multiple(img: np.ndarray, multiple: int = 16):
    h,w = img.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    pt = pad_h // 2; pb = pad_h - pt
    pl = pad_w // 2; pr = pad_w - pl
    padded = np.pad(img, ((pt,pb),(pl,pr)), mode="constant", constant_values=0.0)
    return padded, (pt,pb,pl,pr)

def unpad(img: np.ndarray, pad):
    pt,pb,pl,pr = pad
    h_slice = slice(pt, None) if pb == 0 else slice(pt, -pb)
    w_slice = slice(pl, None) if pr == 0 else slice(pl, -pr)
    return img[h_slice, w_slice]

def apply_normalization(slice2d: np.ndarray) -> np.ndarray:
    s = slice2d.astype(np.float32)
    if NORMALIZATION == "none":
        return s
    if NORMALIZATION == "minmax":
        mn = CLIP_MIN if CLIP_MIN is not None else float(s.min())
        mx = CLIP_MAX if CLIP_MAX is not None else float(s.max())
        if mx > mn:
            return (s - mn) / (mx - mn)
        return s - mn
    if NORMALIZATION == "zscore":
        return (s - MEAN) / (STD if STD != 0 else 1.0)
    raise ValueError("Unknown NORMALIZATION")

def visualize_overlay(image: np.ndarray, labels: np.ndarray, out_png: Path):
    img = image
    if img.shape != labels.shape:
        padded_img, _ = pad_to_multiple(img, multiple=PAD_MULTIPLE)
        img = padded_img
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("Image"); plt.axis("off")
    plt.subplot(1,2,2); plt.imshow(img, cmap="gray"); plt.imshow(labels, cmap="tab10", alpha=0.6, interpolation="nearest")
    plt.title("Prediction overlay"); plt.axis("off")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close()

def extract_state_dict_from_checkpoint(ckpt_path: Path):
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # Try common keys
    state = None
    if isinstance(ckpt, dict):
        for key in ("state_dict","model_state","model_state_dict","model","net","state"):
            if key in ckpt:
                state = ckpt[key]; break
        # nested model_state
        if state is None and "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
            for sub in ("state_dict","model_state_dict","weights"):
                if sub in ckpt["model_state"]:
                    state = ckpt["model_state"][sub]; break
    # fallback: ckpt itself might be a state_dict
    if state is None and isinstance(ckpt, dict) and all(hasattr(v, "shape") or isinstance(v, torch.Tensor) for v in ckpt.values()):
        state = ckpt
    if state is None:
        # try to find a nested dict that looks like a param mapping
        if isinstance(ckpt, dict):
            for v in ckpt.values():
                if isinstance(v, dict) and all(hasattr(x, "shape") or isinstance(x, torch.Tensor) for x in v.values()):
                    state = v; break
    if state is None:
        raise RuntimeError(f"No model weights found in checkpoint. Top-level keys: {list(ckpt.keys())[:50]}")
    # strip common prefixes
    def strip_prefixes(sd, prefixes=("module.","model.","net.")):
        new = {}
        for k,v in sd.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            new[nk] = v
        return new
    state = strip_prefixes(state)
    # convert arrays to tensors if necessary
    state_for_load = {}
    for k,v in state.items():
        if isinstance(v, torch.Tensor):
            state_for_load[k] = v
        else:
            try:
                state_for_load[k] = torch.as_tensor(v)
            except Exception:
                state_for_load[k] = v
    return state_for_load

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        print("Checkpoint not found:", CHECKPOINT_PATH); sys.exit(1)
    if not NIFTI_PATH.exists():
        print("NIfTI not found:", NIFTI_PATH); sys.exit(1)

    arr_raw = load_nifti_array(NIFTI_PATH)
    print("raw shape:", arr_raw.shape, "dtype:", arr_raw.dtype)
    arr = normalize_to_tzyx(arr_raw)
    print("interpreted (T,Z,Y,X):", arr.shape)
    T,Z,Y,X = arr.shape

    # choose slice
    t = T_INDEX if T_INDEX is not None else 0
    z = Z_INDEX if Z_INDEX is not None else None
    if z is None:
        t_best, z_best = pick_most_informative_slice(arr)
        print("auto-picked slice:", (t_best, z_best))
        t, z = t_best, z_best
    if t < 0 or t >= T:
        print("t out of range, using 0"); t = 0
    if z < 0 or z >= Z:
        print(f"z out of range (0..{Z-1}), using 0"); z = 0

    slice2d = arr[t, z].astype(np.float32)
    print(f"selected t={t}, z={z}, slice shape {slice2d.shape}, unique values (sample): {np.unique(slice2d)[:10]}")

    # preprocess
    img_norm = apply_normalization(slice2d)
    padded, pad_info = pad_to_multiple(img_norm, multiple=PAD_MULTIPLE)
    print("padded shape:", padded.shape, "pad_info:", pad_info)

    # tensor
    x = torch.from_numpy(padded.astype(np.float32))[None, None, ...]
    device = DEVICE if (DEVICE == "cpu" or torch.cuda.is_available()) else "cpu"
    x = x.to(device)

    # instantiate model (direct import used above)
    model = UNet(**MODEL_KWARGS)
    model.to(device)

    # load checkpoint weights robustly
    state_for_load = extract_state_dict_from_checkpoint(CHECKPOINT_PATH)
    try:
        model.load_state_dict(state_for_load)
        print("Loaded checkpoint (strict).")
    except Exception as e:
        print("Strict load failed:", e)
        try:
            model.load_state_dict(state_for_load, strict=False)
            print("Loaded checkpoint (non-strict).")
        except Exception as e2:
            print("Non-strict load failed:", e2)
            print("Sample checkpoint keys:")
            for i,k in enumerate(list(state_for_load.keys())[:80]):
                print(i, k)
            raise

    model.eval()
    with torch.no_grad():
        logits = model(x)
        if logits.ndim == 3:
            logits = logits.unsqueeze(0)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    pred_unpadded = unpad(pred, pad_info)
    out_npy = OUT_DIR / f"pred_t{t}_z{z}.npy"
    out_png = OUT_DIR / f"pred_t{t}_z{z}.png"
    np.save(str(out_npy), pred_unpadded)
    visualize_overlay(slice2d, pred, out_png)

    print("Saved prediction .npy:", out_npy)
    print("Saved overlay PNG:", out_png)
    print("Unique labels in prediction:", np.unique(pred_unpadded))

if __name__ == "__main__":
    main()
