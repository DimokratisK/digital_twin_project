"""
Evaluation of MM-WHS segmentation: confusion matrices, per-class metrics,
and overlay visualizations for MRI (Dataset028) and CT (Dataset029).

Supports all nnU-Net configurations (2d, 3d_fullres) and fold 0.

Usage:
    # Evaluate all available configs for both modalities:
    python -m twin_core.nnunet_pipeline.evaluate_mmwhs

    # Only MRI:
    python -m twin_core.nnunet_pipeline.evaluate_mmwhs --modality mri

    # Only CT, specific config:
    python -m twin_core.nnunet_pipeline.evaluate_mmwhs --modality ct --configs 3d_fullres

    # Skip overlays (just metrics and confusion matrices):
    python -m twin_core.nnunet_pipeline.evaluate_mmwhs --no-overlays
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories

set_env_vars()
create_directories()

import os

RESULTS_BASE = Path(os.environ["nnUNet_results"])
PREPROCESSED_BASE = Path(os.environ["nnUNet_preprocessed"])
RAW_BASE = Path(os.environ["nnUNet_raw"])

# MM-WHS class definitions (8 classes including background)
NUM_CLASSES = 8
CLASS_NAMES = ["BG", "LV_Myo", "LA", "LV", "RA", "RV", "Aorta", "PA"]
MMWHS_LABELS = {i: name for i, name in enumerate(CLASS_NAMES)}
MMWHS_COLORS = [
    "#000000",  # 0: BG
    "#e41a1c",  # 1: LV_Myo (red)
    "#4daf4a",  # 2: LA (green)
    "#ff7f00",  # 3: LV (orange)
    "#377eb8",  # 4: RA (blue)
    "#ffd700",  # 5: RV (gold)
    "#984ea3",  # 6: Aorta (purple)
    "#a65628",  # 7: PA (brown)
]

MODALITY_MAP = {
    "mri": {"dataset_id": 28, "dataset_name": "Dataset028_MMWHS"},
    "ct":  {"dataset_id": 29, "dataset_name": "Dataset029_MMWHS_CT"},
}


def get_trainer_dir(dataset_name: str, config: str) -> Path:
    return RESULTS_BASE / dataset_name / f"nnUNetTrainer__nnUNetPlans__{config}"


def get_fold_validation_dir(dataset_name: str, config: str, fold: int) -> Path:
    return get_trainer_dir(dataset_name, config) / f"fold_{fold}" / "validation"


def get_gt_dir(dataset_name: str) -> Path:
    return PREPROCESSED_BASE / dataset_name / "gt_segmentations"


def get_images_dir(dataset_name: str) -> Path:
    return RAW_BASE / dataset_name / "imagesTr"


def available_folds(dataset_name: str, config: str) -> list:
    folds = []
    for fold in range(5):
        val_dir = get_fold_validation_dir(dataset_name, config, fold)
        if val_dir.exists() and list(val_dir.glob("*.nii.gz")):
            folds.append(fold)
    return folds


def compute_confusion_matrix(pred_dir: Path, gt_dir: Path) -> tuple:
    """Compute voxel-level confusion matrix (rows=GT, cols=Pred)."""
    import nibabel as nib

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    matched = 0

    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue

        pred = np.asarray(nib.load(str(pred_path)).dataobj).flatten().astype(np.int64)
        gt = np.asarray(nib.load(str(gt_path)).dataobj).flatten().astype(np.int64)
        pred = np.clip(pred, 0, NUM_CLASSES - 1)
        gt = np.clip(gt, 0, NUM_CLASSES - 1)

        for g in range(NUM_CLASSES):
            for p in range(NUM_CLASSES):
                cm[g, p] += np.sum((gt == g) & (pred == p))
        matched += 1

    return cm, matched


def compute_dice_from_cm(cm: np.ndarray) -> dict:
    """Compute per-class Dice and IoU from confusion matrix."""
    metrics = {}
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[CLASS_NAMES[c]] = {
            "Dice": round(dice, 4),
            "IoU": round(iou, 4),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
        }

    fg_dice = np.mean([metrics[c]["Dice"] for c in CLASS_NAMES[1:]])
    fg_iou = np.mean([metrics[c]["IoU"] for c in CLASS_NAMES[1:]])
    metrics["foreground_mean"] = {"Dice": round(fg_dice, 4), "IoU": round(fg_iou, 4)}
    return metrics


def evaluate_fold(
    dataset_name: str, modality: str, config: str, fold: int, output_dir: Path
):
    """Compute confusion matrix, metrics, and plots for one fold."""
    pred_dir = get_fold_validation_dir(dataset_name, config, fold)
    gt_dir = get_gt_dir(dataset_name)

    if not pred_dir.exists() or not list(pred_dir.glob("*.nii.gz")):
        print(f"  No predictions found at {pred_dir}, skipping")
        return None

    print(f"\n  Computing confusion matrix for {modality.upper()} {config} fold {fold}...")
    cm, matched = compute_confusion_matrix(pred_dir, gt_dir)
    print(f"  Matched {matched} prediction-GT pairs")

    fold_out = output_dir / f"{modality}_{config}" / f"fold_{fold}"
    fold_out.mkdir(parents=True, exist_ok=True)

    np.save(str(fold_out / "confusion_matrix.npy"), cm)

    metrics = compute_dice_from_cm(cm)
    with open(fold_out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print metrics
    print(f"\n  === {modality.upper()} {config} fold {fold} ===")
    for cls_name in CLASS_NAMES[1:]:
        m = metrics[cls_name]
        print(f"    {cls_name:>7s}: Dice={m['Dice']:.4f}  IoU={m['IoU']:.4f}")
    fg = metrics["foreground_mean"]
    print(f"    {'FG':>7s}: Dice={fg['Dice']:.4f}  IoU={fg['IoU']:.4f}")

    # Plot confusion matrices
    from twin_core.utils.plot_confusions import (
        plot_confusion_matrix,
        plot_normalized_confusion,
    )

    title = f"{modality.upper()} {config} fold {fold}"
    plot_confusion_matrix(
        cm, fold_out / "confusion_matrix.png", class_names=CLASS_NAMES, title=title
    )
    plot_normalized_confusion(
        cm, fold_out / "confusion_matrix_norm.png", class_names=CLASS_NAMES, title=title
    )
    print(f"  Saved plots to {fold_out}")

    return metrics


def generate_overlays(
    dataset_name: str,
    modality: str,
    config: str,
    fold: int,
    output_dir: Path,
    max_cases: int = 10,
):
    """Generate GT vs prediction overlay PNGs for a fold."""
    pred_dir = get_fold_validation_dir(dataset_name, config, fold)
    gt_dir = get_gt_dir(dataset_name)
    img_dir = get_images_dir(dataset_name)
    overlay_dir = output_dir / f"{modality}_{config}" / f"fold_{fold}" / "overlays"

    if not pred_dir.exists() or not list(pred_dir.glob("*.nii.gz")):
        print(f"  No predictions for overlays at {pred_dir}")
        return

    from twin_core.nnunet_pipeline.visualize_gt_vs_pred import visualize_case

    cases = sorted(
        [f.name.replace(".nii.gz", "") for f in pred_dir.glob("*.nii.gz")]
    )
    if max_cases and len(cases) > max_cases:
        step = len(cases) // max_cases
        cases = cases[::step][:max_cases]

    print(
        f"\n  Generating overlays for {len(cases)} cases "
        f"({modality.upper()} {config} fold {fold})..."
    )
    for case in cases:
        visualize_case(
            case_name=case,
            img_dir=img_dir,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=overlay_dir,
            labels=MMWHS_LABELS,
            colors=MMWHS_COLORS,
        )
    print(f"  Saved overlays to {overlay_dir}")


def print_comparison_table(all_metrics: dict):
    """Print a summary comparison table across all configs."""
    print("\n" + "=" * 100)
    print("MM-WHS EVALUATION SUMMARY")
    print("=" * 100)
    header = (
        f"{'Modality':<6} {'Config':<12} {'Fold':<5} "
        f"{'LV_Myo':>7} {'LA':>7} {'LV':>7} {'RA':>7} "
        f"{'RV':>7} {'Aorta':>7} {'PA':>7} {'FG Mean':>8}"
    )
    print(header)
    print("-" * 100)

    for key, metrics in sorted(all_metrics.items()):
        modality, config, fold = key
        vals = [metrics.get(c, {}).get("Dice", 0) for c in CLASS_NAMES[1:]]
        fg = metrics.get("foreground_mean", {}).get("Dice", 0)
        row = (
            f"{modality.upper():<6} {config:<12} {fold:<5} "
            + " ".join(f"{v:>7.4f}" for v in vals)
            + f" {fg:>8.4f}"
        )
        print(row)
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MM-WHS segmentation: confusion matrices, metrics, overlays"
    )
    parser.add_argument(
        "--modality",
        choices=["mri", "ct", "both"],
        default="both",
        help="Which modality to evaluate (default: both)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["2d", "3d_fullres"],
        help="Configs to evaluate (default: 2d 3d_fullres)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Folds to evaluate (default: all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation_mmwhs",
        help="Output directory (default: outputs/evaluation_mmwhs)",
    )
    parser.add_argument(
        "--max-overlay-cases",
        type=int,
        default=10,
        help="Max cases for overlay visualization per fold (default: 10)",
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip overlay generation",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["mri", "ct"] if args.modality == "both" else [args.modality]
    all_metrics = {}

    for modality in modalities:
        info = MODALITY_MAP[modality]
        dataset_name = info["dataset_name"]

        print(f"\n{'='*60}")
        print(f"MODALITY: {modality.upper()} ({dataset_name})")
        print(f"{'='*60}")

        for config in args.configs:
            if args.folds is not None:
                folds = args.folds
            else:
                folds = available_folds(dataset_name, config)

            if not folds:
                print(f"  No completed folds found for {config}")
                continue

            print(f"  Available folds for {config}: {folds}")

            for fold in folds:
                metrics = evaluate_fold(
                    dataset_name, modality, config, fold, output_dir
                )
                if metrics:
                    all_metrics[(modality, config, fold)] = metrics

                if not args.no_overlays:
                    generate_overlays(
                        dataset_name,
                        modality,
                        config,
                        fold,
                        output_dir,
                        max_cases=args.max_overlay_cases,
                    )

    if all_metrics:
        print_comparison_table(all_metrics)

        summary = {}
        for (modality, config, fold), metrics in all_metrics.items():
            key = f"{modality}_{config}_fold{fold}"
            summary[key] = metrics
        summary_path = output_dir / "all_metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
