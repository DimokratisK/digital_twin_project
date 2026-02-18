"""Wrapper to run nnU-Net plan_and_preprocess with environment setup."""
import sys

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories
set_env_vars()
create_directories()

from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints,
    plan_experiments,
    preprocess,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, required=True, help="Dataset ID (e.g. 27)")
    parser.add_argument("--verify_dataset_integrity", action="store_true")
    parser.add_argument("--no_pp", action="store_true", help="Skip preprocessing, only fingerprint + plan")
    args = parser.parse_args()

    dataset_id = args.d
    fpe = "DatasetFingerprintExtractor"

    print(f"=== Step 1/3: Extracting fingerprints for Dataset{dataset_id:03d} ===")
    extract_fingerprints(
        dataset_ids=[dataset_id],
        fingerprint_extractor_class_name=fpe,
        check_dataset_integrity=args.verify_dataset_integrity,
        num_processes=2,  # conservative for laptop
    )

    print(f"\n=== Step 2/3: Planning experiments ===")
    plan_experiments(
        dataset_ids=[dataset_id],
        configurations=("2d", "3d_fullres"),
        gpu_memory_target_in_gb=4,  # RTX 3050
    )

    if not args.no_pp:
        print(f"\n=== Step 3/3: Preprocessing ===")
        preprocess(
            dataset_ids=[dataset_id],
            configurations=("2d", "3d_fullres"),
            num_processes=(2, 2),  # conservative for laptop
        )
    else:
        print("\n=== Step 3/3: Skipped (--no_pp) ===")

    print("\nDone! Ready for training.")


if __name__ == "__main__":
    main()
