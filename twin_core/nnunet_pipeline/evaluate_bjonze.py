"""
Evaluation of Bjonze segmentation (Dataset031_Bjonze, 11 classes):
confusion matrices, per-class metrics, per-case Dice distribution violin plots,
and qualitative overlays (best/median/worst case per class).

Usage:
    # Everything (uses fold 0 by default):
    python -m twin_core.nnunet_pipeline.evaluate_bjonze

    # Skip the expensive voxel-level confusion matrix (metrics still come from summary.json):
    python -m twin_core.nnunet_pipeline.evaluate_bjonze --no-confusion

    # Skip overlays:
    python -m twin_core.nnunet_pipeline.evaluate_bjonze --no-overlays

    # Custom output:
    python -m twin_core.nnunet_pipeline.evaluate_bjonze --output-dir outputs/eval_bjonze_final
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories

set_env_vars()
create_directories()

RESULTS_BASE = Path(os.environ["nnUNet_results"])
PREPROCESSED_BASE = Path(os.environ["nnUNet_preprocessed"])
RAW_BASE = Path(os.environ["nnUNet_raw"])

DATASET_ID = 31
DATASET_NAME = "Dataset031_Bjonze"
NUM_CLASSES = 11
CLASS_NAMES = [
    "BG", "Myocardium", "LA", "LV", "RA", "RV",
    "Aorta", "PA", "LAA", "Coronary", "PV",
]
BJONZE_LABELS = {i: name for i, name in enumerate(CLASS_NAMES)}
BJONZE_COLORS = [
    "#000000",  # 0  BG
    "#e41a1c",  # 1  Myocardium (red)
    "#4daf4a",  # 2  LA (green)
    "#ff7f00",  # 3  LV (orange)
    "#377eb8",  # 4  RA (blue)
    "#ffd700",  # 5  RV (gold)
    "#984ea3",  # 6  Aorta (purple)
    "#a65628",  # 7  PA (brown)
    "#f781bf",  # 8  LAA (pink)
    "#00ced1",  # 9  Coronary (cyan)
    "#999933",  # 10 PV (olive)
]


def trainer_dir(config: str) -> Path:
    return RESULTS_BASE / DATASET_NAME / f"nnUNetTrainer__nnUNetPlans__{config}"


def fold_validation_dir(config: str, fold: int) -> Path:
    return trainer_dir(config) / f"fold_{fold}" / "validation"


def gt_dir() -> Path:
    return PREPROCESSED_BASE / DATASET_NAME / "gt_segmentations"


def images_dir() -> Path:
    return RAW_BASE / DATASET_NAME / "imagesTr"


def compute_confusion_matrix(pred_dir: Path, gt_d: Path) -> tuple:
    """Voxel-level confusion matrix (rows=GT, cols=Pred) across all cases.

    Uses bincount instead of nested loop for ~100x speedup vs the MMWHS pattern —
    matters here because 137 cases x ~69M voxels would be slow otherwise.
    """
    import nibabel as nib

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    pred_files = sorted(pred_dir.glob("*.nii.gz"))
    matched = 0

    for pred_path in pred_files:
        gt_path = gt_d / pred_path.name
        if not gt_path.exists():
            continue

        pred = np.asarray(nib.load(str(pred_path)).dataobj).flatten().astype(np.int64)
        gt = np.asarray(nib.load(str(gt_path)).dataobj).flatten().astype(np.int64)
        pred = np.clip(pred, 0, NUM_CLASSES - 1)
        gt = np.clip(gt, 0, NUM_CLASSES - 1)

        flat = gt * NUM_CLASSES + pred
        counts = np.bincount(flat, minlength=NUM_CLASSES * NUM_CLASSES)
        cm += counts.reshape(NUM_CLASSES, NUM_CLASSES)
        matched += 1
        if matched % 10 == 0:
            print(f"    confusion matrix: {matched} cases processed")

    return cm, matched


def compute_dice_from_cm(cm: np.ndarray) -> dict:
    metrics = {}
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[CLASS_NAMES[c]] = {
            "Dice": round(float(dice), 4),
            "IoU": round(float(iou), 4),
            "TP": int(tp), "FP": int(fp), "FN": int(fn),
        }
    fg_classes = CLASS_NAMES[1:]
    metrics["foreground_mean"] = {
        "Dice": round(float(np.mean([metrics[c]["Dice"] for c in fg_classes])), 4),
        "IoU": round(float(np.mean([metrics[c]["IoU"] for c in fg_classes])), 4),
    }
    return metrics


def parse_per_case_dice(summary_path: Path) -> dict:
    """Return {class_label: [(case_name, dice), ...]} from nnUNet summary.json."""
    with open(summary_path) as f:
        s = json.load(f)

    per_class = {str(i): [] for i in range(1, NUM_CLASSES)}
    for entry in s["metric_per_case"]:
        case_name = Path(entry["prediction_file"]).name.replace(".nii.gz", "")
        for label_str, m in entry["metrics"].items():
            if label_str in per_class:
                per_class[label_str].append((case_name, float(m["Dice"])))
    return per_class


def plot_dice_violin(per_class: dict, out_png: Path, title: str):
    import matplotlib.pyplot as plt

    labels_plot = [BJONZE_LABELS[int(k)] for k in sorted(per_class, key=int)]
    data = [[d for _, d in per_class[k]] for k in sorted(per_class, key=int)]

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(BJONZE_COLORS[int(sorted(per_class, key=int)[i])])
        body.set_alpha(0.6)

    ax.set_xticks(range(1, len(labels_plot) + 1))
    ax.set_xticklabels(labels_plot, rotation=30, ha="right")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)


def pick_best_median_worst(per_class: dict) -> dict:
    """Pick 3 cases per class (best, median, worst) for qualitative overlays."""
    picks = {}
    for label, items in per_class.items():
        if not items:
            continue
        sorted_items = sorted(items, key=lambda x: x[1])
        picks[label] = {
            "worst": sorted_items[0],
            "median": sorted_items[len(sorted_items) // 2],
            "best": sorted_items[-1],
        }
    return picks


def generate_targeted_overlays(picks: dict, pred_dir: Path, out_dir: Path):
    """Render CT + GT + Pred overlays for the best/median/worst case of each class."""
    from twin_core.nnunet_pipeline.visualize_gt_vs_pred import visualize_case

    out_dir.mkdir(parents=True, exist_ok=True)
    gt_d = gt_dir()
    img_d = images_dir()
    seen = set()

    for label, by_rank in picks.items():
        for rank, (case_name, dice) in by_rank.items():
            if (case_name, rank) in seen:
                continue
            seen.add((case_name, rank))
            label_name = BJONZE_LABELS[int(label)]
            tag = f"{label_name}_{rank}_dice{dice:.3f}"
            subdir = out_dir / tag
            print(f"    overlay {tag}: {case_name}")
            visualize_case(
                case_name=case_name,
                img_dir=img_d,
                gt_dir=gt_d,
                pred_dir=pred_dir,
                output_dir=subdir,
                labels=BJONZE_LABELS,
                colors=BJONZE_COLORS,
            )


def write_metrics_tables(metrics: dict, summary_means: dict, out_dir: Path):
    """Emit CSV + markdown per-class table combining CM and summary.json numbers."""
    import csv

    rows = []
    for label_idx in range(1, NUM_CLASSES):
        name = CLASS_NAMES[label_idx]
        cm_dice = metrics.get(name, {}).get("Dice") if metrics else None
        summary_dice = summary_means.get(str(label_idx), {}).get("Dice")
        rows.append({
            "label": label_idx,
            "class": name,
            "summary_dice": round(summary_dice, 4) if summary_dice is not None else None,
            "cm_dice": cm_dice,
        })

    csv_path = out_dir / "per_class_dice.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "class", "summary_dice", "cm_dice"])
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "per_class_dice.md"
    with open(md_path, "w") as f:
        f.write("| Label | Class | summary.json Dice | CM-derived Dice |\n")
        f.write("|-------|-------|-------------------|-----------------|\n")
        for r in rows:
            f.write(
                f"| {r['label']} | {r['class']} | "
                f"{r['summary_dice'] if r['summary_dice'] is not None else '—'} | "
                f"{r['cm_dice'] if r['cm_dice'] is not None else '—'} |\n"
            )
    print(f"  wrote {csv_path}\n  wrote {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dataset031_Bjonze validation predictions")
    parser.add_argument("--config", default="3d_fullres")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/evaluation_bjonze")
    parser.add_argument("--no-confusion", action="store_true",
                        help="Skip voxel-level confusion matrix (slow on 137 cases)")
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / f"{args.config}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = fold_validation_dir(args.config, args.fold)
    if not pred_dir.exists() or not list(pred_dir.glob("*.nii.gz")):
        raise SystemExit(f"No predictions at {pred_dir} — has --val --npz finished?")

    summary_path = pred_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}")

    with open(summary_path) as f:
        summary = json.load(f)
    summary_means = summary["mean"]

    print(f"=== Dataset031_Bjonze {args.config} fold {args.fold} ===")
    print(f"Summary foreground_mean Dice: {summary['foreground_mean']['Dice']:.4f}")
    for idx in sorted(summary_means, key=int):
        print(f"  {int(idx):>2}  {CLASS_NAMES[int(idx)]:<11}  Dice={summary_means[idx]['Dice']:.4f}")

    per_case = parse_per_case_dice(summary_path)
    print(f"\nParsed per-case Dice: {sum(len(v) for v in per_case.values())} (class, case) entries")

    print("\nViolin plot of per-case Dice distribution...")
    violin_path = out_dir / "dice_violin.png"
    plot_dice_violin(per_case, violin_path, title=f"Dataset031_Bjonze {args.config} fold {args.fold}")
    print(f"  wrote {violin_path}")

    metrics = None
    if not args.no_confusion:
        print("\nVoxel-level confusion matrix (137 cases, ~69M voxels each)...")
        cm, matched = compute_confusion_matrix(pred_dir, gt_dir())
        print(f"  matched {matched} pred/GT pairs")

        np.save(str(out_dir / "confusion_matrix.npy"), cm)
        metrics = compute_dice_from_cm(cm)
        with open(out_dir / "metrics_from_cm.json", "w") as f:
            json.dump(metrics, f, indent=2)

        from twin_core.utils.plot_confusions import plot_confusion_matrix, plot_normalized_confusion
        title = f"Dataset031_Bjonze {args.config} fold {args.fold}"
        plot_confusion_matrix(cm, out_dir / "confusion_matrix.png",
                              class_names=CLASS_NAMES, title=title)
        plot_normalized_confusion(cm, out_dir / "confusion_matrix_norm.png",
                                  class_names=CLASS_NAMES, title=title)
        print(f"  wrote confusion matrix PNGs to {out_dir}")

    write_metrics_tables(metrics if metrics else {}, summary_means, out_dir)

    if not args.no_overlays:
        print("\nQualitative overlays (best/median/worst per class)...")
        picks = pick_best_median_worst(per_case)
        for label, by_rank in picks.items():
            name = BJONZE_LABELS[int(label)]
            print(f"  {name:<11}  worst={by_rank['worst'][0]} ({by_rank['worst'][1]:.3f})  "
                  f"median={by_rank['median'][0]} ({by_rank['median'][1]:.3f})  "
                  f"best={by_rank['best'][0]} ({by_rank['best'][1]:.3f})")
        generate_targeted_overlays(picks, pred_dir, out_dir / "overlays")

    print(f"\nDone. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
