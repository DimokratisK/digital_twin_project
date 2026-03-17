"""
3-panel comparison: Custom UNet vs nnU-Net vs Ground Truth.

For each selected case and slice, produces a side-by-side overlay:
  [UNet prediction] | [nnU-Net prediction] | [Ground Truth]

Usage:
    # Auto-pick cases from nnU-Net fold 0 validation set:
    python -m twin_core.nnunet_pipeline.compare_unet_vs_nnunet \
        --unet-checkpoint "checkpoints/backup checkpoints/ckpt_best.pt" \
        --nnunet-pred-dir nnunet_data/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation \
        --gt-dir nnunet_data/preprocessed/Dataset027_ACDC/gt_segmentations \
        --img-dir nnunet_data/raw/Dataset027_ACDC/imagesTr \
        --output-dir outputs/comparison_unet_vs_nnunet

    # Specific cases:
    python -m twin_core.nnunet_pipeline.compare_unet_vs_nnunet \
        --unet-checkpoint "checkpoints/backup checkpoints/ckpt_best.pt" \
        --nnunet-pred-dir nnunet_data/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation \
        --gt-dir nnunet_data/preprocessed/Dataset027_ACDC/gt_segmentations \
        --img-dir nnunet_data/raw/Dataset027_ACDC/imagesTr \
        --output-dir outputs/comparison_unet_vs_nnunet \
        --cases patient006_frame01 patient006_frame16
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import nibabel as nib

ACDC_LABELS = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
ACDC_COLORS = ["#000000", "#ffd700", "#4daf4a", "#ff0000"]


def make_3panel_figure(
    image_slice: np.ndarray,
    unet_slice: np.ndarray,
    nnunet_slice: np.ndarray,
    gt_slice: np.ndarray,
    case_name: str,
    slice_idx: int,
    labels: Dict[int, str],
    colors: List[str],
) -> plt.Figure:
    """Create a 3-panel overlay: UNet | nnU-Net | Ground Truth."""
    cmap = ListedColormap([(0, 0, 0, 0)] + [c for c in colors[1:]])
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    for ax, mask, title in [
        (ax1, unet_slice, "Custom UNet (epoch 40)"),
        (ax2, nnunet_slice, "nnU-Net 2D"),
        (ax3, gt_slice, "Ground Truth"),
    ]:
        ax.imshow(image_slice.T, cmap="gray", origin="lower")
        ax.imshow(mask.T, cmap=cmap, norm=norm, alpha=0.6, origin="lower")
        ax.set_title(f"{title}", fontsize=12)
        ax.axis("off")

    handles = [
        Patch(facecolor=c, edgecolor="k", label=f"{labels[i]}")
        for i, c in enumerate(colors)
        if i > 0
    ]
    ax3.legend(handles=handles, loc="upper right", fontsize="small")

    fig.suptitle(f"{case_name}  z={slice_idx}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def run_unet_inference(image_3d: np.ndarray, checkpoint_path: str, device: str = "cpu") -> np.ndarray:
    """Run custom UNet inference on a 3D volume (X, Y, Z) -> labels (X, Y, Z)."""
    import torch
    from twin_core.utils.segmentation_model import load_model
    from twin_core.utils.segmentation_inference import segment_volume

    model = load_model(checkpoint_path, device=device, n_classes=4)

    # image_3d is (X, Y, Z) from nibabel — segment_volume expects (Z, H, W)
    volume_zhw = np.transpose(image_3d, (2, 0, 1))  # (Z, X, Y)

    # Normalize each slice to [0, 1]
    volume_norm = np.zeros_like(volume_zhw, dtype=np.float32)
    for z in range(volume_zhw.shape[0]):
        s = volume_zhw[z].astype(np.float32)
        smin, smax = s.min(), s.max()
        if smax > smin:
            s = (s - smin) / (smax - smin)
        volume_norm[z] = s

    labels_zhw = segment_volume(model, volume_norm, device=device, batch_slices=8)

    # Back to (X, Y, Z)
    labels_xyz = np.transpose(labels_zhw, (1, 2, 0))
    return labels_xyz


def pick_best_slices(gt: np.ndarray, n: int = 3) -> List[int]:
    """Pick z-slices with most foreground content."""
    n_slices = gt.shape[2]
    candidates = []
    for z in range(n_slices):
        n_fg = np.sum(gt[:, :, z] > 0)
        if n_fg > 0:
            candidates.append((z, n_fg))
    candidates.sort(key=lambda x: -x[1])
    if len(candidates) >= n:
        # Most content, mid, least
        selected = [candidates[0][0], candidates[len(candidates) // 2][0], candidates[-1][0]]
    elif candidates:
        selected = [c[0] for c in candidates]
    else:
        selected = [n_slices // 2]
    return sorted(set(selected))


def main():
    parser = argparse.ArgumentParser(description="3-panel comparison: UNet vs nnU-Net vs GT")
    parser.add_argument("--unet-checkpoint", type=str, required=True,
                        help="Path to custom UNet checkpoint (.pt)")
    parser.add_argument("--nnunet-pred-dir", type=str, required=True,
                        help="Directory with nnU-Net predicted .nii.gz files")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory with ground-truth .nii.gz files")
    parser.add_argument("--img-dir", type=str, required=True,
                        help="Directory with input images (*_0000.nii.gz)")
    parser.add_argument("--output-dir", type=str, default="outputs/comparison_unet_vs_nnunet")
    parser.add_argument("--cases", type=str, nargs="+", default=None,
                        help="Specific case names (e.g. patient006_frame01)")
    parser.add_argument("--max-cases", type=int, default=10,
                        help="Max cases to process if --cases not specified")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--slices", type=int, nargs="+", default=None,
                        help="Specific z-slices (default: auto-pick best 3)")

    args = parser.parse_args()

    nnunet_pred_dir = Path(args.nnunet_pred_dir)
    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine cases
    if args.cases:
        cases = args.cases
    else:
        cases = sorted([
            f.name.replace(".nii.gz", "")
            for f in nnunet_pred_dir.glob("*.nii.gz")
        ])
        # Sample evenly
        if len(cases) > args.max_cases:
            step = len(cases) // args.max_cases
            cases = cases[::step][:args.max_cases]

    print(f"Processing {len(cases)} cases")
    print(f"UNet checkpoint: {args.unet_checkpoint}")
    print(f"nnU-Net predictions: {nnunet_pred_dir}")
    print()

    # Cache UNet model (load once)
    unet_labels_cache = {}

    for case in cases:
        img_path = img_dir / f"{case}_0000.nii.gz"
        gt_path = gt_dir / f"{case}.nii.gz"
        nnunet_path = nnunet_pred_dir / f"{case}.nii.gz"

        for p, desc in [(img_path, "image"), (gt_path, "GT"), (nnunet_path, "nnU-Net pred")]:
            if not p.exists():
                print(f"  SKIP {case}: {desc} not found at {p}")
                break
        else:
            print(f"Processing {case}...")

            # Load data
            img = nib.load(str(img_path)).get_fdata()
            gt = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.int16)
            nnunet_pred = np.asarray(nib.load(str(nnunet_path)).dataobj).astype(np.int16)

            # Run UNet inference
            print(f"  Running UNet inference...")
            unet_pred = run_unet_inference(img, args.unet_checkpoint, device=args.device)

            # Pick slices
            slices = args.slices if args.slices else pick_best_slices(gt)

            for z in slices:
                if z >= img.shape[2]:
                    print(f"  SKIP z={z} (only {img.shape[2]} slices)")
                    continue

                fig = make_3panel_figure(
                    image_slice=img[:, :, z],
                    unet_slice=unet_pred[:, :, z],
                    nnunet_slice=nnunet_pred[:, :, z],
                    gt_slice=gt[:, :, z],
                    case_name=case,
                    slice_idx=z,
                    labels=ACDC_LABELS,
                    colors=ACDC_COLORS,
                )
                out_png = output_dir / f"{case}_z{z:02d}_comparison.png"
                fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved {out_png.name}")

    print(f"\nDone. Comparisons saved to {output_dir}")


if __name__ == "__main__":
    main()
