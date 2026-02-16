"""
Training wrapper for nnU-Net with smoke test and full training profiles.

Usage:
    # Smoke test on local machine (RTX 3050, limited VRAM):
    python -m twin_core.nnunet_pipeline.run_training --profile smoke

    # Full training on supercomputer:
    python -m twin_core.nnunet_pipeline.run_training --profile full

    # Custom run:
    python -m twin_core.nnunet_pipeline.run_training \
        --dataset_id 27 --config 2d --fold 0 --num_epochs 50
"""

import argparse
import subprocess
import sys
import os

# Set up environment before anything else
from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories
set_env_vars()
create_directories()


PROFILES = {
    "smoke": {
        "description": "Quick local test (RTX 3050 safe)",
        "config": "2d",
        "fold": 0,
        "num_epochs": 5,
        "npz": False,
    },
    "full": {
        "description": "Full training (supercomputer)",
        "config": "2d",
        "fold": "all",  # will be expanded to 0,1,2,3,4
        "num_epochs": 1000,  # nnU-Net default
        "npz": True,
    },
}


def build_train_command(
    dataset_id: int,
    config: str,
    fold: int,
    num_epochs: int = 1000,
    npz: bool = False,
    extra_args: list = None,
) -> list:
    """Build the nnUNetv2_train command."""
    cmd = [
        sys.executable, "-m", "nnunetv2.run.run_training",
        str(dataset_id),
        config,
        str(fold),
    ]
    if num_epochs != 1000:
        cmd.extend(["-num_epochs", str(num_epochs)])
    if npz:
        cmd.append("--npz")
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def run_training(
    dataset_id: int = 27,
    profile: str = None,
    config: str = None,
    fold=None,
    num_epochs: int = None,
    npz: bool = False,
    extra_args: list = None,
):
    """Run nnU-Net training with the specified settings."""
    # Apply profile defaults, then override with explicit args
    if profile and profile in PROFILES:
        p = PROFILES[profile]
        print(f"Profile: {profile} - {p['description']}")
        config = config or p["config"]
        fold = fold if fold is not None else p["fold"]
        num_epochs = num_epochs if num_epochs is not None else p["num_epochs"]
        npz = npz or p["npz"]
    else:
        config = config or "2d"
        fold = fold if fold is not None else 0
        num_epochs = num_epochs if num_epochs is not None else 1000

    # Expand "all" folds
    if fold == "all":
        folds = [0, 1, 2, 3, 4]
    else:
        folds = [int(fold)]

    print(f"\n=== nnU-Net Training ===")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Configuration: {config}")
    print(f"  Folds: {folds}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Save softmax (--npz): {npz}")
    print(f"  nnUNet_results: {os.environ.get('nnUNet_results', 'NOT SET')}")
    print()

    for f in folds:
        cmd = build_train_command(
            dataset_id=dataset_id,
            config=config,
            fold=f,
            num_epochs=num_epochs,
            npz=npz,
            extra_args=extra_args,
        )
        print(f"--- Fold {f} ---")
        print(f"  Command: {' '.join(cmd)}\n")

        result = subprocess.run(cmd, env=os.environ.copy())

        if result.returncode != 0:
            print(f"\nERROR: Training failed for fold {f} (exit code {result.returncode})")
            sys.exit(result.returncode)

        print(f"\n  Fold {f} completed successfully.\n")

    print("=== All training runs completed ===")


def main():
    parser = argparse.ArgumentParser(
        description="Run nnU-Net training with predefined profiles"
    )
    parser.add_argument(
        "--profile", type=str, choices=list(PROFILES.keys()),
        help="Training profile: 'smoke' (local test) or 'full' (supercomputer)"
    )
    parser.add_argument("--dataset_id", type=int, default=27, help="Dataset ID (default: 27)")
    parser.add_argument("--config", type=str, help="U-Net config: 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres")
    parser.add_argument("--fold", type=str, help="Fold number (0-4) or 'all'")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--npz", action="store_true", help="Save softmax outputs for ensembling")

    args, extra = parser.parse_known_args()

    fold = args.fold
    if fold is not None and fold != "all":
        fold = int(fold)

    run_training(
        dataset_id=args.dataset_id,
        profile=args.profile,
        config=args.config,
        fold=fold,
        num_epochs=args.num_epochs,
        npz=args.npz,
        extra_args=extra if extra else None,
    )


if __name__ == "__main__":
    main()
