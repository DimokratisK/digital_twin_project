"""
Comprehensive evaluation: confusion matrices, per-class metrics, and overlay
visualizations for nnU-Net 2D and 3D configurations.

Works with existing validation predictions (checkpoint_final) or can re-run
inference with checkpoint_best.

Usage:
    # Evaluate everything available (checkpoint_final predictions):
    python -m twin_core.nnunet_pipeline.evaluate_all

    # Re-run validation with checkpoint_best, then evaluate:
    python -m twin_core.nnunet_pipeline.evaluate_all --use-best

    # Only specific configs/folds:
    python -m twin_core.nnunet_pipeline.evaluate_all --configs 2d --folds 0 1 2 3 4
    python -m twin_core.nnunet_pipeline.evaluate_all --configs 3d_fullres --folds 0 1
"""

import argparse
import json
import subprocess
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

DATASET_ID = 27
DATASET_NAME = "Dataset027_ACDC"
CLASS_NAMES = ["BG", "RV", "MYO", "LV"]
NUM_CLASSES = 4


def get_trainer_dir(config: str) -> Path:
    return RESULTS_BASE / DATASET_NAME / f"nnUNetTrainer__nnUNetPlans__{config}"


def get_fold_validation_dir(config: str, fold: int) -> Path:
    return get_trainer_dir(config) / f"fold_{fold}" / "validation"


def get_gt_dir() -> Path:
    return PREPROCESSED_BASE / DATASET_NAME / "gt_segmentations"


def get_images_dir() -> Path:
    return RAW_BASE / DATASET_NAME / "imagesTr"


def available_folds(config: str) -> list:
    """Return list of folds that have completed validation predictions."""
    folds = []
    for fold in range(5):
        val_dir = get_fold_validation_dir(config, fold)
        if val_dir.exists() and list(val_dir.glob("*.nii.gz")):
            folds.append(fold)
    return folds


def rerun_validation_with_best(config: str, fold: int, device: str = "cpu"):
    """Re-run validation inference using checkpoint_best.pth.

    Uses nnU-Net's splits_final.json to identify validation cases,
    symlinks them to a temp dir, runs predict, then moves results.
    """
    import shutil
    import tempfile

    trainer_dir = get_trainer_dir(config)
    fold_dir = trainer_dir / f"fold_{fold}"
    checkpoint = fold_dir / "checkpoint_best.pth"

    if not checkpoint.exists():
        print(f"  WARNING: {checkpoint} not found, skipping")
        return False

    # Read splits to get validation case names
    splits_file = PREPROCESSED_BASE / DATASET_NAME / "splits_final.json"
    with open(splits_file) as f:
        splits = json.load(f)
    val_cases = splits[fold]["val"]

    print(f"  Fold {fold}: {len(val_cases)} validation cases")

    # Create temp input dir with symlinks to validation images
    tmp_input = Path(tempfile.mkdtemp(prefix=f"nnunet_val_{config}_fold{fold}_"))
    images_dir = get_images_dir()
    for case_name in val_cases:
        src = images_dir / f"{case_name}_0000.nii.gz"
        if src.exists():
            dst = tmp_input / f"{case_name}_0000.nii.gz"
            dst.symlink_to(src)

    # Output dir for best checkpoint predictions
    out_dir = fold_dir / "validation_best"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run prediction via nnUNetv2_predict CLI entry point
    # (NOT python -m, which runs __main__ with hardcoded example paths)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(tmp_input),
        "-o", str(out_dir),
        "-d", str(DATASET_ID),
        "-c", config,
        "-f", str(fold),
        "-chk", "checkpoint_best.pth",
        "-device", device,
    ]
    print(f"  Running: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, env=os.environ.copy())

    # Cleanup temp dir
    shutil.rmtree(tmp_input, ignore_errors=True)

    if result.returncode != 0:
        print(f"  ERROR: Prediction failed for {config} fold {fold}")
        return False

    print(f"  Saved best-checkpoint predictions to {out_dir}")
    return True


def compute_confusion_matrix(pred_dir: Path, gt_dir: Path) -> np.ndarray:
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
        metrics[CLASS_NAMES[c]] = {"Dice": round(dice, 4), "IoU": round(iou, 4), "TP": int(tp), "FP": int(fp), "FN": int(fn)}

    fg_dice = np.mean([metrics[c]["Dice"] for c in CLASS_NAMES[1:]])
    fg_iou = np.mean([metrics[c]["IoU"] for c in CLASS_NAMES[1:]])
    metrics["foreground_mean"] = {"Dice": round(fg_dice, 4), "IoU": round(fg_iou, 4)}
    return metrics


def evaluate_fold(config: str, fold: int, output_dir: Path, use_best: bool = False):
    """Compute confusion matrix, metrics, and plots for one fold."""
    if use_best:
        pred_dir = get_trainer_dir(config) / f"fold_{fold}" / "validation_best"
    else:
        pred_dir = get_fold_validation_dir(config, fold)

    gt_dir = get_gt_dir()
    suffix = "best" if use_best else "final"

    if not pred_dir.exists() or not list(pred_dir.glob("*.nii.gz")):
        print(f"  No predictions found at {pred_dir}, skipping")
        return None

    print(f"\n  Computing confusion matrix for {config} fold {fold} ({suffix})...")
    cm, matched = compute_confusion_matrix(pred_dir, gt_dir)
    print(f"  Matched {matched} prediction-GT pairs")

    # Save confusion matrix
    fold_out = output_dir / config / f"fold_{fold}_{suffix}"
    fold_out.mkdir(parents=True, exist_ok=True)

    cm_path = fold_out / "confusion_matrix.npy"
    np.save(str(cm_path), cm)

    # Compute and save metrics
    metrics = compute_dice_from_cm(cm)
    metrics_path = fold_out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print metrics
    print(f"\n  === {config} fold {fold} ({suffix}) ===")
    for cls_name in CLASS_NAMES[1:]:
        m = metrics[cls_name]
        print(f"    {cls_name:>4s}: Dice={m['Dice']:.4f}  IoU={m['IoU']:.4f}")
    fg = metrics["foreground_mean"]
    print(f"    {'FG':>4s}: Dice={fg['Dice']:.4f}  IoU={fg['IoU']:.4f}")

    # Plot confusion matrices
    from twin_core.utils.plot_confusions import plot_confusion_matrix, plot_normalized_confusion

    title = f"{config} fold {fold} ({suffix})"
    plot_confusion_matrix(cm, fold_out / "confusion_matrix.png",
                         class_names=CLASS_NAMES, title=title)
    plot_normalized_confusion(cm, fold_out / "confusion_matrix_norm.png",
                            class_names=CLASS_NAMES, title=title)
    print(f"  Saved plots to {fold_out}")

    return metrics


def generate_overlays(config: str, fold: int, output_dir: Path,
                      use_best: bool = False, max_cases: int = 10):
    """Generate GT vs prediction overlay PNGs for a fold."""
    if use_best:
        pred_dir = get_trainer_dir(config) / f"fold_{fold}" / "validation_best"
    else:
        pred_dir = get_fold_validation_dir(config, fold)

    gt_dir = get_gt_dir()
    img_dir = get_images_dir()
    suffix = "best" if use_best else "final"
    overlay_dir = output_dir / config / f"fold_{fold}_{suffix}" / "overlays"

    if not pred_dir.exists() or not list(pred_dir.glob("*.nii.gz")):
        print(f"  No predictions for overlays at {pred_dir}")
        return

    from twin_core.nnunet_pipeline.visualize_gt_vs_pred import visualize_case

    cases = sorted([f.name.replace(".nii.gz", "") for f in pred_dir.glob("*.nii.gz")])
    if max_cases and len(cases) > max_cases:
        # Sample evenly
        step = len(cases) // max_cases
        cases = cases[::step][:max_cases]

    print(f"\n  Generating overlays for {len(cases)} cases ({config} fold {fold} {suffix})...")
    for case in cases:
        visualize_case(
            case_name=case,
            img_dir=img_dir,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            output_dir=overlay_dir,
        )
    print(f"  Saved overlays to {overlay_dir}")


def print_comparison_table(all_metrics: dict):
    """Print a summary comparison table across all configs/folds."""
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Config':<15} {'Fold':<6} {'Chk':<6} {'RV':>8} {'MYO':>8} {'LV':>8} {'FG Mean':>8}"
    print(header)
    print("-" * 70)

    for key, metrics in sorted(all_metrics.items()):
        config, fold, suffix = key
        rv = metrics.get("RV", {}).get("Dice", 0)
        myo = metrics.get("MYO", {}).get("Dice", 0)
        lv = metrics.get("LV", {}).get("Dice", 0)
        fg = metrics.get("foreground_mean", {}).get("Dice", 0)
        print(f"{config:<15} {fold:<6} {suffix:<6} {rv:>8.4f} {myo:>8.4f} {lv:>8.4f} {fg:>8.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net models: confusion matrices, metrics, overlays")
    parser.add_argument("--configs", nargs="+", default=["2d", "3d_fullres"],
                        help="Configs to evaluate (default: 2d 3d_fullres)")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Folds to evaluate (default: all available)")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation",
                        help="Output directory (default: outputs/evaluation)")
    parser.add_argument("--use-best", action="store_true",
                        help="Re-run validation with checkpoint_best.pth (slower)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for re-running inference (default: cpu)")
    parser.add_argument("--max-overlay-cases", type=int, default=10,
                        help="Max cases for overlay visualization per fold (default: 10)")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Skip overlay generation")
    parser.add_argument("--also-final", action="store_true",
                        help="When using --use-best, also evaluate checkpoint_final for comparison")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for config in args.configs:
        print(f"\n{'='*50}")
        print(f"CONFIG: {config}")
        print(f"{'='*50}")

        if args.folds is not None:
            folds = args.folds
        else:
            folds = available_folds(config)

        if not folds:
            print(f"  No completed folds found for {config}")
            continue

        print(f"  Available folds: {folds}")

        for fold in folds:
            # Re-run with best checkpoint if requested
            if args.use_best:
                print(f"\n--- Re-running validation with checkpoint_best ({config} fold {fold}) ---")
                success = rerun_validation_with_best(config, fold, device=args.device)
                if success:
                    metrics = evaluate_fold(config, fold, output_dir, use_best=True)
                    if metrics:
                        all_metrics[(config, fold, "best")] = metrics
                    if not args.no_overlays:
                        generate_overlays(config, fold, output_dir, use_best=True,
                                        max_cases=args.max_overlay_cases)

            # Also evaluate checkpoint_final (existing predictions)
            if not args.use_best or args.also_final:
                metrics = evaluate_fold(config, fold, output_dir, use_best=False)
                if metrics:
                    all_metrics[(config, fold, "final")] = metrics
                if not args.no_overlays:
                    generate_overlays(config, fold, output_dir, use_best=False,
                                    max_cases=args.max_overlay_cases)

    # Print comparison table
    if all_metrics:
        print_comparison_table(all_metrics)

        # Save all metrics to a single JSON
        summary = {}
        for (config, fold, suffix), metrics in all_metrics.items():
            key = f"{config}_fold{fold}_{suffix}"
            summary[key] = metrics
        summary_path = output_dir / "all_metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull metrics saved to {summary_path}")


if __name__ == "__main__":
    main()
