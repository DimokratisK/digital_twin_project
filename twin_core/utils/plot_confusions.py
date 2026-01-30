#!/usr/bin/env python3
"""
plot_confusions.py

Load one or more confusion matrix .npy files and export PNG heatmaps into outputs/graphs.
- Accepts individual file paths or a directory (will load all *.npy inside).
- Produces one PNG per .npy file, named exactly after the .npy base name (e.g. confusion_val_epoch040.npy -> confusion_val_epoch040.png)
- Also produces a normalized (row-wise) heatmap with suffix _norm.png
- Optional class names (comma-separated) to annotate axes
- Creates output directory if missing

Usage examples:
  python plot_confusions.py --inputs confusion_val_epoch040.npy confusion_train_epoch040.npy
  python plot_confusions.py --inputs /path/to/confusions_dir --out_dir outputs/graphs --class-names BG,RV,MYO,LV
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import textwrap

def ensure_out_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_npy_files(inputs):
    files = []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted([x for x in p.glob("*.npy")]))
        elif p.is_file():
            files.append(p)
        else:
            # allow glob patterns
            import glob
            matches = glob.glob(str(p))
            files.extend([Path(m) for m in matches])
    # deduplicate and sort
    seen = set()
    out = []
    for f in files:
        if f.exists() and f.suffix == ".npy" and str(f) not in seen:
            out.append(f)
            seen.add(str(f))
    return out

def plot_confusion_matrix(cm: np.ndarray,
                          out_png: Path,
                          class_names=None,
                          cmap="Blues",
                          annotate=True,
                          fmt_counts=True,
                          vmin=None,
                          vmax=None,
                          title=None):
    """
    Plot a confusion matrix (2D numpy array) and save to out_png.
    If class_names provided, they are used for tick labels.
    """
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Confusion matrix must be square 2D array. Got shape {cm.shape} for {out_png.name}")

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, n*0.6), max(4, n*0.6)))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title or out_png.stem)
    # ticks
    if class_names:
        names = list(class_names)
        if len(names) != n:
            # fallback: use numeric labels if mismatch
            names = [str(i) for i in range(n)]
    else:
        names = [str(i) for i in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")

    # annotate counts
    if annotate:
        # choose text color based on background
        maxval = cm.max() if cm.size else 1
        thresh = maxval / 2.0
        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                if fmt_counts:
                    txt = f"{int(val):,}"
                else:
                    txt = f"{val:.2f}"
                color = "white" if val > thresh else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)

def plot_normalized_confusion(cm: np.ndarray, out_png: Path, class_names=None, cmap="Blues", title=None):
    """
    Row-normalize (per-ground-truth) confusion matrix and plot percentages.
    """
    cm = np.asarray(cm).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    # avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = np.divide(cm, row_sums, where=(row_sums!=0))
    cm_norm = np.nan_to_num(cm_norm, nan=0.0)
    n = cm.shape[0]

    fig, ax = plt.subplots(figsize=(max(4, n*0.6), max(4, n*0.6)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title or out_png.stem + " (normalized)")
    if class_names and len(class_names) == n:
        names = class_names
    else:
        names = [str(i) for i in range(n)]
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")

    # annotate percentages
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            txt = f"{val*100:5.1f}%"
            color = "white" if val > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion (row-normalized)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        prog="plot_confusions.py",
        description=textwrap.dedent("""Load confusion matrix .npy files and export heatmap PNGs.
        Provide one or more --inputs (files or directories). If a directory is given, all *.npy files inside are processed.
        Output PNGs are written to --out_dir (default: outputs/graphs) and named after the .npy base names.
        """)
    )
    parser.add_argument("--inputs", "-i", nargs="+", required=True,
                        help="Paths to .npy files or directories containing .npy files (supports globs)")
    parser.add_argument("--out_dir", "-o", type=Path, default=Path("outputs/graphs"),
                        help="Directory to save PNG graphs (created if missing)")
    parser.add_argument("--class-names", type=str, default=None,
                        help="Comma-separated class names in label order (e.g. BG,RV,MYO,LV)")
    parser.add_argument("--no-normalized", action="store_true",
                        help="Do not produce the normalized (percent) heatmap")
    parser.add_argument("--cmap", type=str, default="Blues", help="Matplotlib colormap for heatmaps")
    args = parser.parse_args()

    out_dir = args.out_dir
    ensure_out_dir(out_dir)

    # parse class names
    class_names = None
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]

    files = load_npy_files(args.inputs)
    if not files:
        raise SystemExit("No .npy files found in the provided inputs.")

    for f in files:
        try:
            cm = np.load(str(f))
        except Exception as e:
            print(f"Skipping {f} (failed to load): {e}")
            continue

        # validate shape
        if cm.ndim != 2:
            print(f"Skipping {f} (not 2D array, shape={cm.shape})")
            continue
        if cm.shape[0] != cm.shape[1]:
            print(f"Skipping {f} (not square, shape={cm.shape})")
            continue

        base = f.stem  # e.g. confusion_val_epoch040
        out_png = out_dir / f"{base}.png"
        title = base
        try:
            plot_confusion_matrix(cm, out_png, class_names, cmap=args.cmap, annotate=True, fmt_counts=True, title=title)
            print(f"Wrote {out_png}")
        except Exception as e:
            print(f"Failed plotting {f} -> {out_png}: {e}")

        if not args.no_normalized:
            out_png_norm = out_dir / f"{base}_norm.png"
            try:
                plot_normalized_confusion(cm, out_png_norm, class_names, cmap=args.cmap, title=base + " (normalized)")
                print(f"Wrote {out_png_norm}")
            except Exception as e:
                print(f"Failed plotting normalized {f} -> {out_png_norm}: {e}")

if __name__ == "__main__":
    main()
