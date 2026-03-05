"""
Compute a voxel-level confusion matrix from nnU-Net validation predictions.

Loads predicted and ground-truth NIfTI files, accumulates a confusion matrix
across all validation cases, saves it as .npy, and optionally plots it.

Usage:
    python -m twin_core.nnunet_pipeline.compute_confusion_matrix \
        --pred_dir ~/nnunet_data/results/Dataset027_ACDC/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation \
        --gt_dir ~/nnunet_data/preprocessed/Dataset027_ACDC/gt_segmentations \
        --output ~/digital_twin_project/outputs/confusion_matrix_fold0.npy \
        --class_names BG,RV,MYO,LV \
        --plot
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def compute_confusion_matrix(
    pred_dir: Path,
    gt_dir: Path,
    num_classes: int = 4,
) -> np.ndarray:
    """Compute voxel-level confusion matrix (rows=GT, cols=Pred)."""
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))

    if not pred_files:
        raise FileNotFoundError(f"No .nii.gz files found in {pred_dir}")

    print(f"Found {len(pred_files)} prediction files")
    matched = 0

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue

        pred = np.asarray(nib.load(str(pred_path)).dataobj).flatten().astype(np.int64)
        gt = np.asarray(nib.load(str(gt_path)).dataobj).flatten().astype(np.int64)

        # Clip to valid range
        pred = np.clip(pred, 0, num_classes - 1)
        gt = np.clip(gt, 0, num_classes - 1)

        for g_cls in range(num_classes):
            for p_cls in range(num_classes):
                cm[g_cls, p_cls] += np.sum((gt == g_cls) & (pred == p_cls))

        matched += 1

    print(f"Matched {matched} prediction-GT pairs")
    return cm


def main():
    parser = argparse.ArgumentParser(
        description="Compute confusion matrix from nnU-Net validation predictions"
    )
    parser.add_argument(
        "--pred_dir", type=str, required=True,
        help="Directory containing predicted .nii.gz files"
    )
    parser.add_argument(
        "--gt_dir", type=str, required=True,
        help="Directory containing ground-truth .nii.gz files"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for the .npy confusion matrix"
    )
    parser.add_argument(
        "--num_classes", type=int, default=4,
        help="Number of classes (default: 4)"
    )
    parser.add_argument(
        "--class_names", type=str, default=None,
        help="Comma-separated class names (e.g. BG,RV,MYO,LV)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Also save PNG heatmaps next to the .npy file"
    )

    args = parser.parse_args()

    cm = compute_confusion_matrix(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        num_classes=args.num_classes,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), cm)

    print(f"\nConfusion matrix (rows=GT, cols=Pred):")
    print(cm)
    print(f"\nSaved to {output_path}")

    if args.plot:
        from twin_core.utils.plot_confusions import (
            plot_confusion_matrix,
            plot_normalized_confusion,
        )

        class_names = None
        if args.class_names:
            class_names = [s.strip() for s in args.class_names.split(",")]

        png_path = output_path.with_suffix(".png")
        plot_confusion_matrix(cm, png_path, class_names=class_names)
        print(f"Saved plot: {png_path}")

        png_norm = output_path.with_name(output_path.stem + "_norm.png")
        plot_normalized_confusion(cm, png_norm, class_names=class_names)
        print(f"Saved normalized plot: {png_norm}")


if __name__ == "__main__":
    main()
