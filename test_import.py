"""Full dataset: preprocess all 300 cases (no multiprocessing, Windows workaround)."""
import os
import multiprocessing
import time

from twin_core.nnunet_pipeline.set_environment import set_env_vars, create_directories
set_env_vars()
create_directories()


def main():
    from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
    from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed

    dataset_name = maybe_convert_to_dataset_name(27)
    plans = load_json(join(nnUNet_preprocessed, dataset_name, "nnUNetPlans.json"))
    plans_manager = PlansManager(plans)
    dataset_json = load_json(join(nnUNet_raw, dataset_name, "dataset.json"))
    dataset = get_filenames_of_train_images_and_targets(
        join(nnUNet_raw, dataset_name), dataset_json
    )

    total = len(dataset)
    start = time.time()

    for config_name in ("2d", "3d_fullres"):
        print(f"\n=== Preprocessing config: {config_name} ({total} cases) ===", flush=True)
        configuration_manager = plans_manager.get_configuration(config_name)
        preprocessor = DefaultPreprocessor()
        output_directory = join(
            nnUNet_preprocessed, dataset_name,
            configuration_manager.data_identifier
        )
        maybe_mkdir_p(output_directory)

        keys = list(dataset.keys())
        config_start = time.time()
        for i, k in enumerate(keys):
            print(f"  [{i+1}/{total}] {k}...", end=" ", flush=True)
            preprocessor.run_case_save(
                join(output_directory, k),
                dataset[k]["images"],
                dataset[k]["label"],
                plans_manager,
                configuration_manager,
                dataset_json,
            )
            elapsed = time.time() - config_start
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate
            print(f"OK ({remaining:.0f}s left)", flush=True)

    # Copy GT segmentations for validation
    import shutil
    gt_dir = join(nnUNet_preprocessed, dataset_name, "gt_segmentations")
    maybe_mkdir_p(gt_dir)
    labels_dir = join(nnUNet_raw, dataset_name, "labelsTr")
    for f in os.listdir(labels_dir):
        shutil.copy2(join(labels_dir, f), join(gt_dir, f))
    print(f"\nGT segmentations copied to {gt_dir}", flush=True)

    total_time = time.time() - start
    print(f"\nAll done! Total time: {total_time/60:.1f} minutes", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
