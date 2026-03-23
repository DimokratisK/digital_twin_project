"""
Generate comprehensive 2D vs 3D vs Ensemble comparison report for documentation.

Reads the summary.json files from nnUNetv2_find_best_configuration output and
produces:
  1. Per-class Dice/IoU summary table (printed + CSV)
  2. Box plots of per-case Dice by class and configuration
  3. Per-case Dice scatter: 2D vs 3D
  4. Voxel-level confusion matrices (from cross-validation predictions)

Usage:
    python -m twin_core.nnunet_pipeline.generate_comparison_report \
        --summaries-dir Portal/comparison_summaries \
        --output-dir outputs/comparison_report

    # Or point directly to nnUNet results:
    python -m twin_core.nnunet_pipeline.generate_comparison_report \
        --results-dir nnunet_data/results/Dataset027_ACDC \
        --output-dir outputs/comparison_report
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}
CONFIG_LABELS = {
    "2d": "nnU-Net 2D",
    "3d_fullres": "nnU-Net 3D",
    "ensemble": "Ensemble (2D+3D)",
}


def load_summaries(summaries_dir: Path = None, results_dir: Path = None):
    """Load summary.json for each configuration."""
    summaries = {}

    if summaries_dir and summaries_dir.exists():
        for subdir in sorted(summaries_dir.iterdir()):
            if not subdir.is_dir():
                continue
            summary_file = subdir / "summary.json"
            if not summary_file.exists():
                continue
            name = subdir.name
            if "ensemble" in name:
                key = "ensemble"
            elif "2d" in name and "3d" not in name:
                key = "2d"
            elif "3d_fullres" in name:
                key = "3d_fullres"
            else:
                continue
            with open(summary_file) as f:
                summaries[key] = json.load(f)
    elif results_dir and results_dir.exists():
        # Load from nnUNet results directory structure
        for config_key, folder_pattern in [
            ("2d", "nnUNetTrainer__nnUNetPlans__2d"),
            ("3d_fullres", "nnUNetTrainer__nnUNetPlans__3d_fullres"),
        ]:
            cv_dir = results_dir / folder_pattern / "crossval_results_folds_0_1_2_3_4"
            summary_file = cv_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summaries[config_key] = json.load(f)

        # Ensemble
        ensemble_dirs = list(results_dir.glob("ensembles/ensemble_*"))
        if ensemble_dirs:
            summary_file = ensemble_dirs[0] / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summaries["ensemble"] = json.load(f)

    return summaries


def extract_per_case_dice(summary: dict) -> dict:
    """Extract per-case Dice scores from summary, keyed by class."""
    per_class = {cls_id: [] for cls_id in CLASS_NAMES}
    case_names = []

    for case_entry in summary["metric_per_case"]:
        metrics = case_entry["metrics"]
        case_names.append(case_entry.get("prediction_file", ""))
        for cls_id in CLASS_NAMES:
            cls_key = str(cls_id)
            if cls_key in metrics:
                per_class[cls_id].append(metrics[cls_key]["Dice"])
            else:
                per_class[cls_id].append(np.nan)

    return per_class, case_names


def print_summary_table(summaries: dict):
    """Print a formatted comparison table."""
    configs = [k for k in ["2d", "3d_fullres", "ensemble"] if k in summaries]

    print("\n" + "=" * 80)
    print("nnU-Net Configuration Comparison — ACDC Dataset (5-fold CV)")
    print("=" * 80)

    # Header
    header = f"{'Metric':<20s}"
    for cfg in configs:
        header += f"  {CONFIG_LABELS[cfg]:>20s}"
    print(header)
    print("-" * len(header))

    # Foreground mean
    row = f"{'FG Mean Dice':<20s}"
    for cfg in configs:
        val = summaries[cfg]["foreground_mean"]["Dice"]
        row += f"  {val:>20.4f}"
    print(row)

    row = f"{'FG Mean IoU':<20s}"
    for cfg in configs:
        val = summaries[cfg]["foreground_mean"]["IoU"]
        row += f"  {val:>20.4f}"
    print(row)

    print("-" * len(header))

    # Per-class
    for cls_id, cls_name in CLASS_NAMES.items():
        for metric in ["Dice", "IoU"]:
            row = f"{cls_name + ' ' + metric:<20s}"
            for cfg in configs:
                val = summaries[cfg]["mean"][str(cls_id)][metric]
                row += f"  {val:>20.4f}"
            print(row)

    print("=" * len(header))


def save_summary_csv(summaries: dict, output_path: Path):
    """Save summary table as CSV."""
    configs = [k for k in ["2d", "3d_fullres", "ensemble"] if k in summaries]

    rows = []
    rows.append(["Metric"] + [CONFIG_LABELS[c] for c in configs])

    rows.append(["FG Mean Dice"] + [f"{summaries[c]['foreground_mean']['Dice']:.4f}" for c in configs])
    rows.append(["FG Mean IoU"] + [f"{summaries[c]['foreground_mean']['IoU']:.4f}" for c in configs])

    for cls_id, cls_name in CLASS_NAMES.items():
        for metric in ["Dice", "IoU"]:
            row = [f"{cls_name} {metric}"]
            for cfg in configs:
                row.append(f"{summaries[cfg]['mean'][str(cls_id)][metric]:.4f}")
            rows.append(row)

    with open(output_path, "w") as f:
        for row in rows:
            f.write(",".join(row) + "\n")


def plot_dice_boxplots(summaries: dict, output_path: Path):
    """Box plots of per-case Dice by class and configuration."""
    configs = [k for k in ["2d", "3d_fullres", "ensemble"] if k in summaries]
    colors = {"2d": "#1f77b4", "3d_fullres": "#ff7f0e", "ensemble": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax_idx, (cls_id, cls_name) in enumerate(CLASS_NAMES.items()):
        ax = axes[ax_idx]
        data = []
        labels = []
        box_colors = []

        for cfg in configs:
            per_class, _ = extract_per_case_dice(summaries[cfg])
            data.append(per_class[cls_id])
            labels.append(CONFIG_LABELS[cfg].replace(" (2D+3D)", "\n(2D+3D)"))
            box_colors.append(colors[cfg])

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.5))

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(cls_name, fontsize=14, fontweight="bold")
        ax.set_ylabel("Dice Score" if ax_idx == 0 else "")
        ax.set_ylim(0.5, 1.02)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Per-Case Dice Score Distribution — ACDC 5-Fold CV", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved box plots: {output_path}")


def plot_dice_scatter_2d_vs_3d(summaries: dict, output_path: Path):
    """Scatter plot of per-case Dice: 2D (x) vs 3D (y) for each class."""
    if "2d" not in summaries or "3d_fullres" not in summaries:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax_idx, (cls_id, cls_name) in enumerate(CLASS_NAMES.items()):
        ax = axes[ax_idx]
        dice_2d, _ = extract_per_case_dice(summaries["2d"])
        dice_3d, _ = extract_per_case_dice(summaries["3d_fullres"])

        x = np.array(dice_2d[cls_id])
        y = np.array(dice_3d[cls_id])

        # Remove NaN pairs
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]

        ax.scatter(x, y, alpha=0.4, s=20, c="#4c72b0")
        ax.plot([0.5, 1], [0.5, 1], "k--", alpha=0.5, linewidth=1)

        ax.set_xlabel("nnU-Net 2D Dice")
        ax.set_ylabel("nnU-Net 3D Dice" if ax_idx == 0 else "")
        ax.set_title(cls_name, fontsize=14, fontweight="bold")
        ax.set_xlim(0.5, 1.02)
        ax.set_ylim(0.5, 1.02)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        # Count wins
        n_2d_better = np.sum(x > y)
        n_3d_better = np.sum(y > x)
        ax.text(0.05, 0.95, f"2D better: {n_2d_better}\n3D better: {n_3d_better}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle("Per-Case Dice: 2D vs 3D — ACDC 5-Fold CV", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter plot: {output_path}")


def plot_foreground_bar_chart(summaries: dict, output_path: Path):
    """Bar chart comparing foreground Dice and per-class Dice."""
    configs = [k for k in ["2d", "3d_fullres", "ensemble"] if k in summaries]
    colors = {"2d": "#1f77b4", "3d_fullres": "#ff7f0e", "ensemble": "#2ca02c"}

    categories = ["FG Mean"] + list(CLASS_NAMES.values())
    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cfg in enumerate(configs):
        values = [summaries[cfg]["foreground_mean"]["Dice"]]
        for cls_id in CLASS_NAMES:
            values.append(summaries[cfg]["mean"][str(cls_id)]["Dice"])

        bars = ax.bar(x + i * width - width, values, width,
                      label=CONFIG_LABELS[cfg], color=colors[cfg], alpha=0.8)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Dice Score Comparison — ACDC 5-Fold CV", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0.85, 0.97)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar chart: {output_path}")


def compute_and_plot_confusion_matrices(summaries: dict, output_dir: Path):
    """Compute voxel-level confusion matrices from per-case TP/FP/FN/TN."""
    configs = [k for k in ["2d", "3d_fullres", "ensemble"] if k in summaries]

    fig, axes = plt.subplots(1, len(configs), figsize=(7 * len(configs), 6))
    if len(configs) == 1:
        axes = [axes]

    class_labels = ["BG", "RV", "MYO", "LV"]

    for ax_idx, cfg in enumerate(configs):
        ax = axes[ax_idx]
        summary = summaries[cfg]

        # Build approximate confusion matrix from mean TP/FP/FN/TN
        # These are per-class binary metrics, so we reconstruct a per-class view
        n_cases = len(summary["metric_per_case"])
        cm = np.zeros((4, 4), dtype=np.float64)

        # Aggregate over all cases
        for case_entry in summary["metric_per_case"]:
            metrics = case_entry["metrics"]
            for cls_id_str, cls_metrics in metrics.items():
                cls_id = int(cls_id_str)
                cm[cls_id, cls_id] += cls_metrics["TP"]
                # FN: GT=cls_id but predicted as something else
                cm[cls_id, 0] += cls_metrics["FN"]  # approximate: assign FN to BG
                # FP: predicted as cls_id but GT is something else
                cm[0, cls_id] += cls_metrics["FP"]  # approximate: assign FP from BG

        # BG diagonal: TN for all classes (use class 1 as reference)
        for case_entry in summary["metric_per_case"]:
            metrics = case_entry["metrics"]
            # TN is shared across classes approximately
            if "1" in metrics:
                cm[0, 0] += metrics["1"]["TN"]

        # Normalize rows to percentages
        cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)

        for i in range(4):
            for j in range(4):
                val = cm_norm[i, j]
                color = "white" if val > 60 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color=color, fontsize=11)

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(class_labels, fontsize=11)
        ax.set_yticklabels(class_labels, fontsize=11)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Ground Truth" if ax_idx == 0 else "", fontsize=12)
        ax.set_title(CONFIG_LABELS[cfg], fontsize=13, fontweight="bold")

    fig.suptitle("Normalized Confusion Matrices — ACDC 5-Fold CV", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_dir / "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrices: {output_dir / 'confusion_matrices.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate 2D vs 3D vs Ensemble comparison report")
    parser.add_argument("--summaries-dir", type=Path, default=None,
                        help="Directory containing summary subdirectories (e.g. Portal/comparison_summaries)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="nnUNet results directory (e.g. nnunet_data/results/Dataset027_ACDC)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison_report"),
                        help="Output directory for report")
    args = parser.parse_args()

    if args.summaries_dir is None and args.results_dir is None:
        # Try default locations
        if Path("Portal/comparison_summaries").exists():
            args.summaries_dir = Path("Portal/comparison_summaries")
        elif Path("nnunet_data/results/Dataset027_ACDC").exists():
            args.results_dir = Path("nnunet_data/results/Dataset027_ACDC")
        else:
            parser.error("Provide --summaries-dir or --results-dir")

    summaries = load_summaries(args.summaries_dir, args.results_dir)

    if not summaries:
        print("ERROR: No summary files found!")
        return

    print(f"Loaded configurations: {', '.join(CONFIG_LABELS[k] for k in summaries)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Print and save summary table
    print_summary_table(summaries)
    save_summary_csv(summaries, args.output_dir / "summary_table.csv")
    print(f"  Saved CSV: {args.output_dir / 'summary_table.csv'}")

    # 2. Bar chart
    plot_foreground_bar_chart(summaries, args.output_dir / "dice_bar_chart.png")

    # 3. Box plots
    plot_dice_boxplots(summaries, args.output_dir / "dice_boxplots.png")

    # 4. Scatter 2D vs 3D
    plot_dice_scatter_2d_vs_3d(summaries, args.output_dir / "dice_scatter_2d_vs_3d.png")

    # 5. Confusion matrices
    compute_and_plot_confusion_matrices(summaries, args.output_dir)

    print(f"\nReport complete! All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
