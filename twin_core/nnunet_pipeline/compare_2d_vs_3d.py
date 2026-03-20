"""
3-panel comparison: nnU-Net 2D vs nnU-Net 3D fullres vs Ground Truth.

For each selected case and slice, produces a side-by-side overlay:
  [nnU-Net 2D prediction] | [nnU-Net 3D prediction] | [Ground Truth]

Usage:
    python -m twin_core.nnunet_pipeline.compare_2d_vs_3d \
        --pred-2d-dir nnunet_data/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation \
        --pred-3d-dir nnunet_data/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation \
        --gt-dir nnunet_data/preprocessed/Dataset027_ACDC/gt_segmentations \
        --img-dir nnunet_data/raw/Dataset027_ACDC/imagesTr \
        --output-dir outputs/comparison_2d_vs_3d

    # Specific cases:
    python -m twin_core.nnunet_pipeline.compare_2d_vs_3d \
        --pred-2d-dir nnunet_data/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation \
        --pred-3d-dir nnunet_data/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/validation \
        --gt-dir nnunet_data/preprocessed/Dataset027_ACDC/gt_segmentations \
        --img-dir nnunet_data/raw/Dataset027_ACDC/imagesTr \
        --output-dir outputs/comparison_2d_vs_3d \
        --cases patient006_frame01 patient006_frame16
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import nibabel as nib

ACDC_LABELS = {0: "BG", 1: "RV", 2: "MYO", 3: "LV"}
ACDC_COLORS = ["#000000", "#ffd700", "#4daf4a", "#ff0000"]


def make_3panel_figure(
    image_slice: np.ndarray,
    pred_2d_slice: np.ndarray,
    pred_3d_slice: np.ndarray,
    gt_slice: np.ndarray,
    case_name: str,
    slice_idx: int,
    labels: Dict[int, str],
    colors: List[str],
) -> plt.Figure:
    """Create a 3-panel overlay: nnU-Net 2D | nnU-Net 3D | Ground Truth."""
    cmap = ListedColormap([(0, 0, 0, 0)] + [c for c in colors[1:]])
    bounds = np.arange(-0.5, len(colors) + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    for ax, mask, title in [
        (ax1, pred_2d_slice, "nnU-Net 2D"),
        (ax2, pred_3d_slice, "nnU-Net 3D fullres"),
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
        selected = [candidates[0][0], candidates[len(candidates) // 2][0], candidates[-1][0]]
    elif candidates:
        selected = [c[0] for c in candidates]
    else:
        selected = [n_slices // 2]
    return sorted(set(selected))


def main():
    parser = argparse.ArgumentParser(description="3-panel comparison: nnU-Net 2D vs 3D vs GT")
    parser.add_argument("--pred-2d-dir", type=str, required=True,
                        help="Directory with nnU-Net 2D predicted .nii.gz files")
    parser.add_argument("--pred-3d-dir", type=str, required=True,
                        help="Directory with nnU-Net 3D fullres predicted .nii.gz files")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory with ground-truth .nii.gz files")
    parser.add_argument("--img-dir", type=str, required=True,
                        help="Directory with input images (*_0000.nii.gz)")
    parser.add_argument("--output-dir", type=str, default="outputs/comparison_2d_vs_3d")
    parser.add_argument("--cases", type=str, nargs="+", default=None,
                        help="Specific case names (e.g. patient006_frame01)")
    parser.add_argument("--max-cases", type=int, default=10,
                        help="Max cases to process if --cases not specified")
    parser.add_argument("--slices", type=int, nargs="+", default=None,
                        help="Specific z-slices (default: auto-pick best 3)")

    args = parser.parse_args()

    pred_2d_dir = Path(args.pred_2d_dir)
    pred_3d_dir = Path(args.pred_3d_dir)
    gt_dir = Path(args.gt_dir)
    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine cases: intersection of 2D and 3D predictions
    cases_2d = {f.name.replace(".nii.gz", "") for f in pred_2d_dir.glob("*.nii.gz")}
    cases_3d = {f.name.replace(".nii.gz", "") for f in pred_3d_dir.glob("*.nii.gz")}
    common_cases = sorted(cases_2d & cases_3d)

    if args.cases:
        cases = [c for c in args.cases if c in common_cases]
        missing = [c for c in args.cases if c not in common_cases]
        if missing:
            print("WARNING: cases not found in both 2D and 3D predictions: {}".format(
                ", ".join(missing)))
    else:
        cases = common_cases
        if len(cases) > args.max_cases:
            step = len(cases) // args.max_cases
            cases = cases[::step][:args.max_cases]

    print("Processing {} cases".format(len(cases)))
    print("2D predictions: {}".format(pred_2d_dir))
    print("3D predictions: {}".format(pred_3d_dir))
    print("Common cases between 2D and 3D: {}".format(len(common_cases)))
    print()

    for case in cases:
        img_path = img_dir / "{}_0000.nii.gz".format(case)
        gt_path = gt_dir / "{}.nii.gz".format(case)
        pred_2d_path = pred_2d_dir / "{}.nii.gz".format(case)
        pred_3d_path = pred_3d_dir / "{}.nii.gz".format(case)

        paths_ok = True
        for p, desc in [(img_path, "image"), (gt_path, "GT"),
                        (pred_2d_path, "2D pred"), (pred_3d_path, "3D pred")]:
            if not p.exists():
                print("  SKIP {}: {} not found at {}".format(case, desc, p))
                paths_ok = False
                break

        if not paths_ok:
            continue

        print("Processing {}...".format(case))

        img = nib.load(str(img_path)).get_fdata()
        gt = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.int16)
        pred_2d = np.asarray(nib.load(str(pred_2d_path)).dataobj).astype(np.int16)
        pred_3d = np.asarray(nib.load(str(pred_3d_path)).dataobj).astype(np.int16)

        slices = args.slices if args.slices else pick_best_slices(gt)

        for z in slices:
            if z >= img.shape[2]:
                print("  SKIP z={} (only {} slices)".format(z, img.shape[2]))
                continue

            fig = make_3panel_figure(
                image_slice=img[:, :, z],
                pred_2d_slice=pred_2d[:, :, z],
                pred_3d_slice=pred_3d[:, :, z],
                gt_slice=gt[:, :, z],
                case_name=case,
                slice_idx=z,
                labels=ACDC_LABELS,
                colors=ACDC_COLORS,
            )
            out_png = output_dir / "{}_z{:02d}_2d_vs_3d.png".format(case, z)
            fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  Saved {}".format(out_png.name))

    print("\nDone. Comparisons saved to {}".format(output_dir))


if __name__ == "__main__":
    main()
