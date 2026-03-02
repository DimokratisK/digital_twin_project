"""
Inference wrapper for nnU-Net: predict segmentation masks from trained models.

Supports single-config prediction and multi-config ensembling.

Usage:
    # Single config (use best config determined by run_find_best_config):
    python -m twin_core.nnunet_pipeline.run_inference \
        -i /path/to/images -o /path/to/predictions --config 2d

    # Ensemble of 2d + 3d_fullres:
    python -m twin_core.nnunet_pipeline.run_inference \
        -i /path/to/images -o /path/to/predictions --config 2d 3d_fullres

    # With specific folds and trainer (e.g. smoke test model):
    python -m twin_core.nnunet_pipeline.run_inference \
        -i /path/to/images -o /path/to/predictions \
        --config 2d --folds 0 --trainer nnUNetTrainer_5epochs
"""

import argparse
import os
import subprocess
import sys
from typing import Optional

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories
set_env_vars()
create_directories()


def build_predict_command(
    input_folder: str,
    output_folder: str,
    dataset_id: int,
    config: str,
    folds: list,
    trainer: str = "nnUNetTrainer",
    checkpoint: str = "checkpoint_final.pth",
    disable_tta: bool = False,
    save_probabilities: bool = False,
    device: str = "cuda",
) -> list:
    """Build the nnUNetv2_predict command."""
    cmd = [
        sys.executable, "-m", "nnunetv2.inference.predict_from_raw_data",
        "-i", input_folder,
        "-o", output_folder,
        "-d", str(dataset_id),
        "-c", config,
        "-f", *[str(f) for f in folds],
        "-tr", trainer,
        "-chk", checkpoint,
        "-device", device,
    ]
    if disable_tta:
        cmd.append("--disable_tta")
    if save_probabilities:
        cmd.append("--save_probabilities")
    return cmd


def build_ensemble_command(
    input_folders: list,
    output_folder: str,
) -> list:
    """Build the nnUNetv2_ensemble command."""
    return [
        sys.executable, "-m", "nnunetv2.ensembling.ensemble",
        "-i", *input_folders,
        "-o", output_folder,
    ]


def run_subprocess(cmd: list, description: str):
    """Run a subprocess command, print it, and exit on failure."""
    print(f"  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=os.environ.copy())
    if result.returncode != 0:
        print(f"\nERROR: {description} failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def run_inference(
    input_folder: str,
    output_folder: str,
    dataset_id: int = 27,
    configs: Optional[list] = None,
    folds: Optional[list] = None,
    trainer: str = "nnUNetTrainer",
    checkpoint: str = "checkpoint_final.pth",
    disable_tta: bool = False,
    device: str = "cuda",
):
    """Run nnU-Net inference with single config or ensemble."""
    if configs is None:
        configs = ["2d"]
    if folds is None:
        folds = [0, 1, 2, 3, 4]

    print(f"\n=== nnU-Net Inference ===")
    print(f"  Input: {input_folder}")
    print(f"  Output: {output_folder}")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Config(s): {configs}")
    print(f"  Folds: {folds}")
    print(f"  Trainer: {trainer}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Test-time augmentation: {'disabled' if disable_tta else 'enabled'}")
    print(f"  Device: {device}")
    print()

    if len(configs) == 1:
        # Single config — predict directly to output folder
        print(f"--- Predicting with {configs[0]} ---")
        cmd = build_predict_command(
            input_folder=input_folder,
            output_folder=output_folder,
            dataset_id=dataset_id,
            config=configs[0],
            folds=folds,
            trainer=trainer,
            checkpoint=checkpoint,
            disable_tta=disable_tta,
            device=device,
        )
        run_subprocess(cmd, f"Prediction ({configs[0]})")

    else:
        # Multiple configs — predict each to temp folder, then ensemble
        temp_folders = []
        for cfg in configs:
            temp_out = f"{output_folder}_{cfg}"
            temp_folders.append(temp_out)

            print(f"--- Predicting with {cfg} (saving probabilities for ensemble) ---")
            cmd = build_predict_command(
                input_folder=input_folder,
                output_folder=temp_out,
                dataset_id=dataset_id,
                config=cfg,
                folds=folds,
                trainer=trainer,
                checkpoint=checkpoint,
                disable_tta=disable_tta,
                save_probabilities=True,
                device=device,
            )
            run_subprocess(cmd, f"Prediction ({cfg})")
            print(f"  {cfg} predictions saved to {temp_out}\n")

        # Ensemble the predictions
        print(f"--- Ensembling {len(configs)} configs ---")
        cmd = build_ensemble_command(
            input_folders=temp_folders,
            output_folder=output_folder,
        )
        run_subprocess(cmd, "Ensemble")

    print(f"\n=== Inference complete. Results in: {output_folder} ===")


def main():
    parser = argparse.ArgumentParser(
        description="Run nnU-Net inference (single config or ensemble)"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="Input folder containing .nii.gz images"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output folder for predicted segmentation masks"
    )
    parser.add_argument(
        "-d", "--dataset_id", type=int, default=27,
        help="Dataset ID (default: 27)"
    )
    parser.add_argument(
        "-c", "--config", type=str, nargs="+", required=True,
        help="Config(s): '2d', '3d_fullres', or both for ensemble"
    )
    parser.add_argument(
        "-f", "--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Folds to use (default: 0 1 2 3 4)"
    )
    parser.add_argument(
        "-tr", "--trainer", type=str, default="nnUNetTrainer",
        help="Trainer class name (default: nnUNetTrainer)"
    )
    parser.add_argument(
        "-chk", "--checkpoint", type=str, default="checkpoint_final.pth",
        help="Checkpoint filename (default: checkpoint_final.pth)"
    )
    parser.add_argument(
        "--disable_tta", action="store_true",
        help="Disable test-time augmentation (faster, slightly less accurate)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda)"
    )

    args = parser.parse_args()

    run_inference(
        input_folder=args.input,
        output_folder=args.output,
        dataset_id=args.dataset_id,
        configs=args.config,
        folds=args.folds,
        trainer=args.trainer,
        checkpoint=args.checkpoint,
        disable_tta=args.disable_tta,
        device=args.device,
    )


if __name__ == "__main__":
    main()
