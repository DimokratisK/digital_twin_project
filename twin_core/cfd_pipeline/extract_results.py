"""
Extract hemodynamic metrics from OpenFOAM simulation results.

Computes time-averaged WSS (TAWSS), oscillatory shear index (OSI),
and generates summary reports.

Usage:
    # Extract WSS from completed simulation:
    python -m twin_core.cfd_pipeline.extract_results \
        --case cfd_results/patient006_frame01/case \
        --output cfd_results/patient006_frame01/report

    # Quick summary only:
    python -m twin_core.cfd_pipeline.extract_results \
        --case cfd_results/patient006_frame01/case --summary
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def find_time_directories(case_dir: str) -> List[Path]:
    """Find all time step directories in an OpenFOAM case.

    Returns sorted list of paths to time directories (excluding 0/).
    """
    case_path = Path(case_dir)
    time_dirs = []
    for d in case_path.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    time_dirs.append(d)
            except ValueError:
                continue
    return sorted(time_dirs, key=lambda p: float(p.name))


def parse_openfoam_vector_field(filepath: str, patch: str = "wall") -> Optional[np.ndarray]:
    """Parse an OpenFOAM vector field file into a numpy array.

    For boundary fields (like wallShearStress), reads data from the
    specified patch under boundaryField. For volume fields, reads
    internalField.

    Returns array of shape (N, 3) for N face/cell values.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        content = f.read()

    # First try: look for data under boundaryField -> patch -> value
    # This is where wallShearStress stores its per-face data
    patch_pos = content.find(f"    {patch}\n")
    if patch_pos == -1:
        patch_pos = content.find(f"    {patch}\r\n")

    if patch_pos != -1:
        # Find "nonuniform List<vector>" after the patch name
        list_pos = content.find("nonuniform List<vector>", patch_pos)
        if list_pos != -1:
            # Find the opening ( after the count
            start = content.find("(", list_pos)
            if start != -1:
                # Find matching closing ) — skip nested ()
                depth = 0
                end = start
                for i in range(start, len(content)):
                    if content[i] == "(":
                        depth += 1
                    elif content[i] == ")":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break

                data_str = content[start + 1:end].strip()
                vectors = _parse_vector_block(data_str)
                if vectors is not None:
                    return vectors

    # Fallback: try internalField nonuniform
    internal_pos = content.find("internalField   nonuniform")
    if internal_pos != -1:
        start = content.find("(", internal_pos)
        if start != -1:
            depth = 0
            end = start
            for i in range(start, len(content)):
                if content[i] == "(":
                    depth += 1
                elif content[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            data_str = content[start + 1:end].strip()
            vectors = _parse_vector_block(data_str)
            if vectors is not None:
                return vectors

    return None


def _parse_vector_block(data_str: str) -> Optional[np.ndarray]:
    """Parse a block of OpenFOAM vector data into numpy array."""
    if not data_str:
        return None

    vectors = []
    for line in data_str.split("\n"):
        line = line.strip().strip("()")
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                vectors.append([float(x) for x in parts])
            except ValueError:
                continue

    if not vectors:
        return None

    return np.array(vectors)


def compute_wss_magnitude(wss_vectors: np.ndarray) -> np.ndarray:
    """Compute WSS magnitude from vector field. Returns 1D array."""
    return np.linalg.norm(wss_vectors, axis=1)


def compute_tawss(wss_time_series: List[np.ndarray], dt_values: List[float]) -> np.ndarray:
    """Compute Time-Averaged Wall Shear Stress (TAWSS).

    TAWSS = (1/T) * integral(|WSS(t)|) dt

    Parameters
    ----------
    wss_time_series : list of WSS vector arrays, each shape (N, 3)
    dt_values : list of time intervals between consecutive snapshots

    Returns
    -------
    TAWSS array of shape (N,) in Pa
    """
    total_time = sum(dt_values)
    if total_time == 0:
        return np.zeros(wss_time_series[0].shape[0])

    tawss = np.zeros(wss_time_series[0].shape[0])
    for i, (wss, dt) in enumerate(zip(wss_time_series, dt_values)):
        tawss += compute_wss_magnitude(wss) * dt

    return tawss / total_time


def compute_osi(wss_time_series: List[np.ndarray], dt_values: List[float]) -> np.ndarray:
    """Compute Oscillatory Shear Index (OSI).

    OSI = 0.5 * (1 - |integral(WSS dt)| / integral(|WSS| dt))

    OSI ranges from 0 (unidirectional) to 0.5 (fully oscillatory).
    High OSI correlates with endothelial dysfunction and thrombus risk.

    Parameters
    ----------
    wss_time_series : list of WSS vector arrays, each shape (N, 3)
    dt_values : list of time intervals

    Returns
    -------
    OSI array of shape (N,)
    """
    total_time = sum(dt_values)
    if total_time == 0:
        return np.zeros(wss_time_series[0].shape[0])

    # Numerator: magnitude of time-integrated WSS vector
    integrated_wss = np.zeros_like(wss_time_series[0])
    for wss, dt in zip(wss_time_series, dt_values):
        integrated_wss += wss * dt

    mag_integrated = np.linalg.norm(integrated_wss, axis=1)

    # Denominator: time-integrated WSS magnitude
    integrated_mag = np.zeros(wss_time_series[0].shape[0])
    for wss, dt in zip(wss_time_series, dt_values):
        integrated_mag += compute_wss_magnitude(wss) * dt

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        osi = 0.5 * (1.0 - mag_integrated / integrated_mag)
        osi = np.where(np.isfinite(osi), osi, 0.0)

    return np.clip(osi, 0.0, 0.5)


def extract_wss_from_case(case_dir: str, cardiac_cycle: float = 0.8) -> Dict:
    """Extract WSS metrics from a completed OpenFOAM case.

    Parameters
    ----------
    case_dir : path to OpenFOAM case directory
    cardiac_cycle : duration of one cardiac cycle in seconds

    Returns
    -------
    dict with TAWSS, OSI statistics and arrays
    """
    time_dirs = find_time_directories(case_dir)

    if not time_dirs:
        return {"error": "No time directories found"}

    # Collect WSS data from each time step
    wss_series = []
    times = []

    for td in time_dirs:
        wss_file = td / "wallShearStress"
        if not wss_file.exists():
            continue

        wss = parse_openfoam_vector_field(str(wss_file))
        if wss is not None and wss.shape[0] > 0:
            wss_series.append(wss)
            times.append(float(td.name))

    if len(wss_series) < 2:
        return {"error": f"Insufficient WSS data (found {len(wss_series)} time steps)"}

    # Compute time intervals
    dt_values = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    # Use only interior intervals (skip last WSS entry to match dt count)
    wss_for_avg = wss_series[:-1]

    # Skip initial transient (first cardiac cycle)
    skip_time = cardiac_cycle
    start_idx = 0
    cumulative_time = 0.0
    for i, dt in enumerate(dt_values):
        cumulative_time += dt
        if cumulative_time >= skip_time:
            start_idx = i + 1
            break

    if start_idx >= len(wss_for_avg):
        start_idx = 0  # Fall back to using all data

    wss_subset = wss_for_avg[start_idx:]
    dt_subset = dt_values[start_idx:]

    if len(wss_subset) < 2:
        wss_subset = wss_for_avg
        dt_subset = dt_values

    # Compute metrics
    tawss = compute_tawss(wss_subset, dt_subset)
    osi = compute_osi(wss_subset, dt_subset)

    # Convert to physical units (Pa → dyne/cm² for clinical convention)
    from twin_core.cfd_pipeline.boundary_conditions import BLOOD_DENSITY
    tawss_pa = tawss * BLOOD_DENSITY  # kinematic → physical
    tawss_dyne_cm2 = tawss_pa * 10  # Pa → dyne/cm²

    return {
        "n_time_steps": len(wss_series),
        "n_wall_faces": wss_series[0].shape[0],
        "time_range_s": [times[0], times[-1]],
        "analysis_start_s": times[start_idx] if start_idx < len(times) else times[0],
        "tawss": {
            "mean_Pa": float(np.mean(tawss_pa)),
            "max_Pa": float(np.max(tawss_pa)),
            "min_Pa": float(np.min(tawss_pa)),
            "std_Pa": float(np.std(tawss_pa)),
            "mean_dyne_cm2": float(np.mean(tawss_dyne_cm2)),
            "max_dyne_cm2": float(np.max(tawss_dyne_cm2)),
        },
        "osi": {
            "mean": float(np.mean(osi)),
            "max": float(np.max(osi)),
            "min": float(np.min(osi)),
            "std": float(np.std(osi)),
            "high_osi_fraction": float(np.mean(osi > 0.3)),
        },
        # Raw arrays for visualization
        "_tawss_array": tawss_pa.tolist(),
        "_osi_array": osi.tolist(),
    }


def generate_report(case_dir: str, output_dir: str, cardiac_cycle: float = 0.8):
    """Generate a comprehensive hemodynamic report.

    Saves:
    - summary.json: key metrics
    - tawss.npy: TAWSS array for visualization
    - osi.npy: OSI array for visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Extracting hemodynamic metrics ===")
    print(f"  Case: {case_dir}")
    print(f"  Output: {output_dir}")

    results = extract_wss_from_case(case_dir, cardiac_cycle)

    if "error" in results:
        print(f"\n  ERROR: {results['error']}")
        return results

    # Save arrays separately
    tawss_array = np.array(results.pop("_tawss_array"))
    osi_array = np.array(results.pop("_osi_array"))

    np.save(str(output_path / "tawss.npy"), tawss_array)
    np.save(str(output_path / "osi.npy"), osi_array)

    # Save summary
    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n  Time steps analyzed: {results['n_time_steps']}")
    print(f"  Wall faces: {results['n_wall_faces']:,}")
    print(f"\n  TAWSS:")
    print(f"    Mean: {results['tawss']['mean_Pa']:.4f} Pa ({results['tawss']['mean_dyne_cm2']:.2f} dyne/cm²)")
    print(f"    Max:  {results['tawss']['max_Pa']:.4f} Pa")
    print(f"    Min:  {results['tawss']['min_Pa']:.4f} Pa")
    print(f"\n  Oscillatory Shear Index (OSI):")
    print(f"    Mean: {results['osi']['mean']:.4f}")
    print(f"    Max:  {results['osi']['max']:.4f}")
    print(f"    High OSI fraction (>0.3): {results['osi']['high_osi_fraction']:.1%}")
    print(f"\n  Files saved:")
    print(f"    {summary_path}")
    print(f"    {output_path / 'tawss.npy'}")
    print(f"    {output_path / 'osi.npy'}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract hemodynamic metrics from OpenFOAM results"
    )
    parser.add_argument(
        "--case", type=str, required=True,
        help="Path to OpenFOAM case directory"
    )
    parser.add_argument(
        "-o", "--output", type=str,
        help="Output directory for report (default: <case>/report)"
    )
    parser.add_argument(
        "--cardiac-cycle", type=float, default=0.8,
        help="Cardiac cycle duration in seconds (default: 0.8)"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary only (no file output)"
    )

    args = parser.parse_args()

    if args.summary:
        results = extract_wss_from_case(args.case, args.cardiac_cycle)
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            # Remove raw arrays for clean output
            results.pop("_tawss_array", None)
            results.pop("_osi_array", None)
            print(json.dumps(results, indent=2))
    else:
        output_dir = args.output or str(Path(args.case) / "report")
        generate_report(args.case, output_dir, args.cardiac_cycle)


if __name__ == "__main__":
    main()
