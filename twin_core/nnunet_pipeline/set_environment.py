"""
Set up nnU-Net environment variables for the digital twin project.

Usage (from project root, with conda env activated):
    # Check current state and print setup instructions:
    python -m twin_core.nnunet_pipeline.set_environment

    # Create directories and set env vars for the current process:
    python -m twin_core.nnunet_pipeline.set_environment --apply
"""

import argparse
import os
import sys
from pathlib import Path


# Project root: two levels up from this file (twin_core/nnunet_pipeline/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data root: use NNUNET_DATA_ROOT env var if set (e.g. on HPC scratch disk),
# otherwise default to <project_root>/nnunet_data.
# On the lab supercomputer, if data lives on a separate partition:
#   export NNUNET_DATA_ROOT=/scratch/user/nnunet_data
NNUNET_DATA_DIR = Path(os.environ.get("NNUNET_DATA_ROOT", PROJECT_ROOT / "nnunet_data"))

PATH_DEFAULTS = {
    "nnUNet_raw": NNUNET_DATA_DIR / "raw",
    "nnUNet_preprocessed": NNUNET_DATA_DIR / "preprocessed",
    "nnUNet_results": NNUNET_DATA_DIR / "results",
}

# Hardware-specific settings (overridable via actual environment variables).
# These defaults are conservative for low-VRAM GPUs (e.g. RTX 3050 4GB).
# On the lab supercomputer, set nnUNet_compile=true and nnUNet_n_proc_DA=12
# (or higher) via the environment before running.
HARDWARE_DEFAULTS = {
    "nnUNet_compile": "false",   # torch.compile uses extra VRAM; disable on low-VRAM GPUs
    "nnUNet_n_proc_DA": "2",     # data augmentation workers; reduce to save RAM
}

# All defaults that can be overridden by the environment
ALL_DEFAULTS = {**PATH_DEFAULTS, **HARDWARE_DEFAULTS}


def set_env_vars():
    """Set nnU-Net environment variables for the current process.

    All variables are only set if not already present in the environment,
    so the lab's supercomputer can override any of them without touching this file.

    To redirect data to a different disk on HPC:
        export NNUNET_DATA_ROOT=/scratch/user/nnunet_data
    To override individual paths:
        export nnUNet_raw=/custom/path/raw
    To override hardware settings:
        export nnUNet_compile=true
        export nnUNet_n_proc_DA=12
    """
    for var_name, default in ALL_DEFAULTS.items():
        if var_name not in os.environ:
            os.environ[var_name] = str(default)


def create_directories():
    """Create the nnU-Net data directories if they don't exist."""
    for var_name in PATH_DEFAULTS:
        path = Path(os.environ[var_name])
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {var_name}: {path}")


def check_nnunet_installation():
    """Verify that nnU-Net v2 is importable."""
    try:
        import nnunetv2
        print(f"  nnU-Net v2 found: {nnunetv2.__file__}")
        return True
    except ImportError:
        print("  WARNING: nnU-Net v2 is not installed or not importable.")
        print("  Install it with: pip install -e ./nnUNet")
        return False


def check_torch():
    """Verify torch installation and CUDA availability."""
    try:
        import torch
        cuda_status = "available" if torch.cuda.is_available() else "NOT available"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        print(f"  PyTorch {torch.__version__}, CUDA: {cuda_status}, GPU: {gpu_name}")
        return True
    except ImportError:
        print("  WARNING: PyTorch is not installed.")
        return False


def print_status():
    """Print current environment variable status."""
    print("\n--- Current Environment Variables ---")
    for var_name, expected_path in PATH_DEFAULTS.items():
        current = os.environ.get(var_name)
        if current is None:
            print(f"  {var_name}: NOT SET")
        else:
            print(f"  {var_name}: {current}")


def print_setup_instructions():
    """Print commands for permanent setup."""
    print("\n--- To set permanently in your conda environment ---")
    print("Run these commands once in your activated conda env:\n")
    for var_name, path in PATH_DEFAULTS.items():
        print(f'  conda env config vars set {var_name}="{path}"')
    print("\nThen reactivate your environment:")
    print("  conda deactivate && conda activate lab_venv_nnunet")

    print("\n--- Or set permanently via Windows (PowerShell) ---\n")
    for var_name, path in PATH_DEFAULTS.items():
        print(f'  [System.Environment]::SetEnvironmentVariable("{var_name}", "{path}", "User")')


def main():
    parser = argparse.ArgumentParser(description="Set up nnU-Net environment variables")
    parser.add_argument("--apply", action="store_true",
                        help="Create directories and set env vars for this process")
    args = parser.parse_args()

    print("=== nnU-Net Environment Setup ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {NNUNET_DATA_DIR}\n")

    print("--- Dependencies ---")
    check_torch()
    check_nnunet_installation()

    if args.apply:
        print("\n--- Creating Directories ---")
        create_directories()
        print("\n--- Setting Environment Variables (this process) ---")
        set_env_vars()
        for var_name in PATH_DEFAULTS:
            print(f"  {var_name} = {os.environ[var_name]}")
        print("\nDone. Environment configured for this process.")
    else:
        print_status()

    print_setup_instructions()


if __name__ == "__main__":
    main()
