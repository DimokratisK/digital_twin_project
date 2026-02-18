"""
Post-training wrapper: find best nnU-Net configuration and determine postprocessing.

Runs nnUNetv2_find_best_configuration after all folds have been trained.
This compares 2d vs 3d_fullres across 5-fold cross-validation, evaluates
ensembles, determines postprocessing, and saves inference instructions.

Usage:
    python -m twin_core.nnunet_pipeline.run_find_best_config
    python -m twin_core.nnunet_pipeline.run_find_best_config --dataset_id 27
"""

import argparse
import os
import subprocess
import sys

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories
set_env_vars()
create_directories()


def main():
    parser = argparse.ArgumentParser(
        description="Find best nnU-Net configuration after full training"
    )
    parser.add_argument(
        "--dataset_id", type=int, default=27,
        help="Dataset ID (default: 27)"
    )
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "nnunetv2.evaluation.find_best_configuration",
        str(args.dataset_id),
        "-c", "2d", "3d_fullres",
    ]

    print(f"\n=== Finding Best Configuration ===")
    print(f"  Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        print(f"\nERROR: find_best_configuration failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print("\n=== Done. Check the output above for inference instructions. ===")


if __name__ == "__main__":
    main()
