"""
CLI wrapper to run the complete OpenFOAM cardiac CFD pipeline.

End-to-end: STL → prepare mesh → generate case → run simulation → extract results.

Usage:
    # Full pipeline from STL to WSS results:
    python -m twin_core.cfd_pipeline.run_openfoam \
        --stl meshes/patient006_frame01/LV.stl \
        --output cfd_results/patient006_frame01 \
        --chamber lv --heart-rate 75 --num-cycles 5

    # Generate case only (don't run solver):
    python -m twin_core.cfd_pipeline.run_openfoam \
        --stl meshes/patient006_frame01/LV.stl \
        --output cfd_results/patient006_frame01 \
        --chamber lv --generate-only

    # Run solver on existing case:
    python -m twin_core.cfd_pipeline.run_openfoam \
        --case cfd_results/patient006_frame01 --run-only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _run_command(cmd: list, cwd: str, description: str) -> bool:
    """Run a shell command and return True on success."""
    print(f"\n--- {description} ---")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Working dir: {cwd}\n")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=os.environ.copy(),
    )

    if result.returncode != 0:
        print(f"\nERROR: {description} failed (exit code {result.returncode})")
        return False
    return True


def check_openfoam_installed() -> bool:
    """Check if OpenFOAM commands are available."""
    try:
        result = subprocess.run(
            ["blockMesh", "-help"],
            capture_output=True,
            text=True,
        )
        return True
    except FileNotFoundError:
        return False


def run_meshing(case_dir: str) -> bool:
    """Run blockMesh and snappyHexMesh."""
    cwd = str(case_dir)

    if not _run_command(["blockMesh"], cwd, "Background mesh (blockMesh)"):
        return False

    if not _run_command(
        ["snappyHexMesh", "-overwrite"], cwd, "Volume mesh from STL (snappyHexMesh)"
    ):
        return False

    return True


def run_solver(case_dir: str, n_processors: int = 4, parallel: bool = True) -> bool:
    """Run pimpleFoam (serial or parallel)."""
    cwd = str(case_dir)

    if parallel and n_processors > 1:
        if not _run_command(
            ["decomposePar"], cwd, "Decompose for parallel run"
        ):
            return False

        if not _run_command(
            ["mpirun", "-np", str(n_processors), "pimpleFoam", "-parallel"],
            cwd,
            f"Solve (pimpleFoam, {n_processors} processors)",
        ):
            return False

        if not _run_command(
            ["reconstructPar"], cwd, "Reconstruct parallel results"
        ):
            return False
    else:
        if not _run_command(["pimpleFoam"], cwd, "Solve (pimpleFoam, serial)"):
            return False

    return True


def run_postprocess(case_dir: str) -> bool:
    """Run OpenFOAM post-processing for WSS."""
    cwd = str(case_dir)

    if not _run_command(
        ["postProcess", "-func", "wallShearStress"],
        cwd,
        "Compute wall shear stress",
    ):
        return False

    return True


def run_full_pipeline(
    stl_path: str,
    output_dir: str,
    chamber: str = "lv",
    heart_rate_bpm: float = 75.0,
    num_cycles: int = 5,
    dt: float = 1e-4,
    n_processors: int = 4,
    generate_only: bool = False,
    skip_mesh_prep: bool = False,
):
    """Run the complete CFD pipeline from STL to results.

    Steps:
    1. Prepare STL mesh for CFD (repair, smooth)
    2. Cut valve openings (create multi-region STL with inlet/outlet/wall)
    3. Generate OpenFOAM case directory
    4. Run meshing (blockMesh + snappyHexMesh)
    5. Run solver (pimpleFoam)
    6. Post-process (WSS extraction)
    """
    from twin_core.cfd_pipeline.prepare_cfd_mesh import (
        check_mesh_quality,
        repair_for_cfd,
        print_quality_report,
    )
    from twin_core.cfd_pipeline.cut_valve_openings import prepare_valve_stl
    from twin_core.cfd_pipeline.openfoam_case import create_openfoam_case

    stl_path = Path(stl_path)
    output_dir = Path(output_dir)

    print("=" * 60)
    print("  OpenFOAM Cardiac CFD Pipeline")
    print("=" * 60)

    # Step 1: Check/prepare mesh
    if not skip_mesh_prep:
        print(f"\n=== Step 1/6: Preparing mesh for CFD ===")
        report = check_mesh_quality(str(stl_path))
        print_quality_report(report)

        if not report["cfd_ready"]:
            print("Mesh needs repair. Repairing...")
            prepared_stl = output_dir / "prepared_mesh" / stl_path.name
            report = repair_for_cfd(
                str(stl_path), str(prepared_stl), smoothing_iterations=10
            )
            print_quality_report(report)
            stl_path = prepared_stl
            if not report["cfd_ready"]:
                print("WARNING: Mesh still has issues after repair. Proceeding anyway.")
        else:
            print("Mesh is CFD-ready.")
    else:
        print(f"\n=== Step 1/6: Skipping mesh preparation ===")

    # Step 2: Cut valve openings (create multi-region STL)
    print(f"\n=== Step 2/6: Identifying valve regions ===")
    valve_stl = output_dir / "valve_mesh" / (stl_path.stem + "_valves.stl")
    prepare_valve_stl(
        stl_path=str(stl_path),
        output_path=str(valve_stl),
        scale_to_metres=True,
    )
    stl_path = valve_stl

    # Step 3: Generate OpenFOAM case
    print(f"\n=== Step 3/6: Generating OpenFOAM case ===")
    case_dir = output_dir / "case"
    create_openfoam_case(
        stl_path=str(stl_path),
        output_dir=str(case_dir),
        chamber=chamber,
        heart_rate_bpm=heart_rate_bpm,
        num_cycles=num_cycles,
        dt=dt,
        n_processors=n_processors,
    )

    if generate_only:
        print(f"\n=== Case generated. Stopping (--generate-only). ===")
        print(f"  To run manually: cd {case_dir} && bash run.sh")
        return

    # Check OpenFOAM is installed
    if not check_openfoam_installed():
        print("\nERROR: OpenFOAM is not installed or not in PATH.")
        print("Install with: sudo apt install openfoam-dev")
        print(f"\nCase files generated at: {case_dir}")
        print("You can run manually after installing OpenFOAM.")
        sys.exit(1)

    # Step 4: Meshing
    print(f"\n=== Step 4/6: Meshing ===")
    if not run_meshing(str(case_dir)):
        sys.exit(1)

    # Step 5: Solver
    print(f"\n=== Step 5/6: Running solver ===")
    if not run_solver(str(case_dir), n_processors):
        sys.exit(1)

    # Step 6: Post-processing
    print(f"\n=== Step 6/6: Post-processing ===")
    if not run_postprocess(str(case_dir)):
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete. Results in: {case_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenFOAM cardiac CFD pipeline"
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--stl", type=str,
        help="Path to input STL file (runs full pipeline)"
    )
    input_group.add_argument(
        "--case", type=str,
        help="Path to existing OpenFOAM case directory (--run-only)"
    )

    parser.add_argument(
        "-o", "--output", type=str,
        help="Output directory (required with --stl)"
    )
    parser.add_argument(
        "--chamber", type=str, default="lv", choices=["la", "lv"],
        help="Cardiac chamber type (default: lv)"
    )
    parser.add_argument(
        "--heart-rate", type=float, default=75.0,
        help="Heart rate in bpm (default: 75)"
    )
    parser.add_argument(
        "--num-cycles", type=int, default=5,
        help="Number of cardiac cycles (default: 5)"
    )
    parser.add_argument(
        "--dt", type=float, default=1e-4,
        help="Initial time step in seconds (default: 1e-4)"
    )
    parser.add_argument(
        "--processors", type=int, default=4,
        help="Number of parallel processors (default: 4)"
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="Only generate case files, don't run solver"
    )
    parser.add_argument(
        "--run-only", action="store_true",
        help="Run solver on existing case (use with --case)"
    )
    parser.add_argument(
        "--skip-mesh-prep", action="store_true",
        help="Skip STL repair step"
    )

    args = parser.parse_args()

    if args.stl and not args.output:
        parser.error("--output is required when using --stl")

    if args.run_only:
        if not args.case:
            parser.error("--case is required with --run-only")
        case_dir = args.case
        if not check_openfoam_installed():
            print("ERROR: OpenFOAM is not installed or not in PATH.")
            sys.exit(1)
        print("=== Running solver on existing case ===")
        if not run_meshing(case_dir):
            sys.exit(1)
        if not run_solver(case_dir, args.processors):
            sys.exit(1)
        if not run_postprocess(case_dir):
            sys.exit(1)
        print("=== Done ===")
    else:
        run_full_pipeline(
            stl_path=args.stl,
            output_dir=args.output,
            chamber=args.chamber,
            heart_rate_bpm=args.heart_rate,
            num_cycles=args.num_cycles,
            dt=args.dt,
            n_processors=args.processors,
            generate_only=args.generate_only,
            skip_mesh_prep=args.skip_mesh_prep,
        )


if __name__ == "__main__":
    main()
