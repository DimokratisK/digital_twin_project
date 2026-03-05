"""
Visualize nnU-Net predictions vs ground truth as side-by-side overlay PNGs.

For each case, shows the MRI slice with GT overlay (left) and prediction
overlay (right) using deterministic colors.

Usage:
    # Visualize specific cases:
    python -m twin_core.nnunet_pipeline.visualize_gt_vs_pred \
        --pred_dir .../fold_0/validation \
        --gt_dir .../gt_segmentations \
        --img_dir .../imagesTr \
        --output_dir outputs/overlays \
        --cases patient006_frame01 patient006_frame16

    # Visualize all validation cases:
    python -m twin_core.nnunet_pipeline.visualize_gt_vs_pred \
        --pred_dir .../fold_0/validation \
        --gt_dir .../gt_segmentations \
        --img_dir .../imagesTr \
        --output_dir outputs/overlays \
        --all
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import nibabel as nib


# ACDC class definitions
ACDC_LABELS = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
ACDC_COLORS = ["#000000", "#ffd700", "#4daf4a", "#ff0000"]  # BG, RV, MYO, LV


def make_overlay_figure(
    image_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    case_name: str,
    slice_idx: int,
    labels: Dict[int, str],
    colors: List[str],
) -> plt.Figure:
    """Create a side-by-side GT vs Prediction overlay figure."""
    cmap = ListedColormap([(0, 0, 0, 0)] + [c for c in colors[1:]])
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(image_slice.T, cmap="gray", origin="lower")
    ax1.imshow(gt_slice.T, cmap=cmap, norm=norm, alpha=0.6, origin="lower")
    ax1.set_title(f"{case_name} z={slice_idx} — Ground Truth")
    ax1.axis("off")

    ax2.imshow(image_slice.T, cmap="gray", origin="lower")
    ax2.imshow(pred_slice.T, cmap=cmap, norm=norm, alpha=0.6, origin="lower")
    ax2.set_title(f"{case_name} z={slice_idx} — Prediction")
    ax2.axis("off")

    handles = [
        Patch(facecolor=c, edgecolor="k", label=f"{i}: {labels[i]}")
        for i, c in enumerate(colors)
        if i > 0
    ]
    ax2.legend(handles=handles, loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


def visualize_case(
    case_name: str,
    img_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    output_dir: Path,
    slices: Optional[List[int]] = None,
    labels: Optional[Dict[int, str]] = None,
    colors: Optional[List[str]] = None,
):
    """Generate overlay PNGs for a single case."""
    if labels is None:
        labels = ACDC_LABELS
    if colors is None:
        colors = ACDC_COLORS

    img_path = img_dir / f"{case_name}_0000.nii.gz"
    gt_path = gt_dir / f"{case_name}.nii.gz"
    pred_path = pred_dir / f"{case_name}.nii.gz"

    for p, desc in [(img_path, "image"), (gt_path, "GT"), (pred_path, "prediction")]:
        if not p.exists():
            print(f"  SKIP {case_name}: {desc} not found at {p}")
            return

    img = nib.load(str(img_path)).get_fdata()
    gt = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.int16)
    pred = np.asarray(nib.load(str(pred_path)).dataobj).astype(np.int16)

    n_slices = img.shape[2]

    if slices is None:
        # Pick mid slice and a few others with structure content
        candidates = []
        for z in range(n_slices):
            n_fg = np.sum(gt[:, :, z] > 0)
            if n_fg > 0:
                candidates.append((z, n_fg))
        candidates.sort(key=lambda x: -x[1])
        # Take up to 3 slices: most content, mid-content, least content
        if len(candidates) >= 3:
            slices = [candidates[0][0], candidates[len(candidates) // 2][0], candidates[-1][0]]
        elif candidates:
            slices = [c[0] for c in candidates]
        else:
            slices = [n_slices // 2]

    output_dir.mkdir(parents=True, exist_ok=True)

    for z in slices:
        if z >= n_slices:
            print(f"  SKIP z={z} (only {n_slices} slices)")
            continue

        fig = make_overlay_figure(
            image_slice=img[:, :, z],
            gt_slice=gt[:, :, z],
            pred_slice=pred[:, :, z],
            case_name=case_name,
            slice_idx=z,
            labels=labels,
            colors=colors,
        )
        out_png = output_dir / f"{case_name}_z{z:02d}_gt_vs_pred.png"
        fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_png.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize nnU-Net predictions vs ground truth"
    )
    parser.add_argument(
        "--pred_dir", type=str, required=True,
        help="Directory with predicted .nii.gz files"
    )
    parser.add_argument(
        "--gt_dir", type=str, required=True,
        help="Directory with ground-truth .nii.gz files"
    )
    parser.add_argument(
        "--img_dir", type=str, required=True,
        help="Directory with input images (*_0000.nii.gz)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/overlays",
        help="Output directory for overlay PNGs"
    )
    parser.add_argument(
        "--cases", type=str, nargs="+", default=None,
        help="Specific case names (e.g. patient006_frame01)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Visualize all validation cases"
    )
    parser.add_argument(
        "--slices", type=int, nargs="+", default=None,
        help="Specific z-slice indices to visualize (default: auto-pick)"
    )

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)

    if args.cases:
        cases = args.cases
    elif args.all:
        cases = sorted([
            f.name.replace(".nii.gz", "")
            for f in pred_dir.glob("*.nii.gz")
        ])
    else:
        parser.error("Provide --cases or --all")

    print(f"Visualizing {len(cases)} case(s)\n")

    for case in cases:
        print(f"Processing {case}...")
        visualize_case(
            case_name=case,
            img_dir=img_dir,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=output_dir,
            slices=args.slices,
        )

    print(f"\nDone. Overlays saved to {output_dir}")


if __name__ == "__main__":
    main()
