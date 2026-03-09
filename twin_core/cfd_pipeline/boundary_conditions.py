"""
Literature-based boundary conditions for cardiac CFD simulations.

Provides pulsatile waveforms for left atrium (LA) and left ventricle (LV)
CFD simulations, derived from published literature values.

Usage:
    from twin_core.cfd_pipeline.boundary_conditions import (
        get_la_boundary_conditions,
        get_lv_boundary_conditions,
        scale_waveform_to_heart_rate,
        write_openfoam_time_series,
    )

References
----------
[1] Otani T, Al-Issa A, Pourmorteza A, McVeigh ER, Wada S, Ashikaga H.
    "A Computational Framework for Personalized Blood Flow Analysis in the
    Human Left Atrium." Ann Biomed Eng 44(11):3284-3294, 2016.
    doi:10.1007/s10439-016-1590-x

[2] Koizumi R, Funamoto K, Hayase T, Kanke Y, Shibata M, Shiraishi Y,
    Yambe T. "Numerical analysis of hemodynamic changes in the left atrium
    due to atrial fibrillation." J Biomech 48(3):472-478, 2015.
    doi:10.1016/S0021-9290(14)00679-4

[3] Masci A, Alessandrini M, Forti D, Menghini F, Dede L, Tommasi C,
    Quarteroni A, Corsi C. "A Patient-Specific Computational Fluid Dynamics
    Model of the Left Atrium in Atrial Fibrillation: Development and Initial
    Evaluation." FIMH 2017, LNCS 10263, pp. 392-400, 2017.
    doi:10.1007/978-3-319-59448-4_37

[4] Balzotti C, Siena P, Girfoglio M, Stabile G, Duenas-Pamplona J,
    Sierra-Pallares J, Amat-Santos I, Rozza G. "A reduced order model
    formulation for left atrium flow: an atrial fibrillation case."
    Biomech Model Mechanobiol 23:1411-1429, 2024.
    doi:10.1007/s10237-024-01847-1

Blood properties consensus (from [1], [2], [4]):
    - Density: 1050 kg/m3
    - Dynamic viscosity: 0.0035 Pa.s (Newtonian)
    - Kinematic viscosity: 3.333e-6 m2/s
    - Laminar flow assumption (Re ~ 1100-2400)
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# =============================================================================
# Blood properties — consensus from Otani [1], Koizumi [2], Balzotti [4]
# =============================================================================

BLOOD_DENSITY = 1050.0        # kg/m³ [1],[2],[4]
BLOOD_DYNAMIC_VISCOSITY = 0.0035  # Pa·s [1],[2],[4]
BLOOD_KINEMATIC_VISCOSITY = BLOOD_DYNAMIC_VISCOSITY / BLOOD_DENSITY  # ~3.333e-6 m²/s


# =============================================================================
# Left Atrium boundary conditions
# =============================================================================

# Normalized pulsatile waveform for pulmonary vein (PV) inflow.
# Time is normalized to one cardiac cycle [0, 1].
# Velocity is normalized to peak velocity (multiply by patient-specific peak).
# Based on Koizumi et al. (2015) and Masci et al. (2019).
#
# Phases:
#   0.0 - 0.35: Systolic filling (S-wave, blood enters LA from PVs)
#   0.35 - 0.50: Diastasis (low flow)
#   0.50 - 0.70: Early diastolic emptying (E-wave, blood exits LA through MV)
#   0.70 - 0.85: Diastasis
#   0.85 - 1.00: Late diastolic (A-wave, atrial contraction)

_PV_INFLOW_NORMALIZED = np.array([
    # (normalized_time, normalized_velocity)
    [0.00, 0.20],
    [0.05, 0.40],
    [0.10, 0.65],
    [0.15, 0.85],
    [0.20, 1.00],   # Peak S-wave
    [0.25, 0.85],
    [0.30, 0.55],
    [0.35, 0.30],
    [0.40, 0.15],
    [0.45, 0.10],   # Diastasis minimum
    [0.50, 0.25],
    [0.55, 0.50],
    [0.60, 0.70],   # D-wave (diastolic PV filling)
    [0.65, 0.55],
    [0.70, 0.30],
    [0.75, 0.15],
    [0.80, 0.10],   # Diastasis
    [0.85, 0.20],
    [0.90, 0.35],   # A-wave reversal (atrial contraction)
    [0.95, 0.25],
    [1.00, 0.20],
])

# Typical peak velocity at pulmonary veins (m/s)
PV_PEAK_VELOCITY = 0.5  # m/s (healthy adult)

# Pulmonary vein flow distribution
PV_FLOW_SPLIT = {
    "right_superior": 0.28,   # ~28% of total inflow
    "right_inferior": 0.27,   # ~27%
    "left_superior": 0.24,    # ~24%
    "left_inferior": 0.21,    # ~21%
}

# Mitral valve outflow — normalized waveform
# Represents flow exiting LA through mitral valve during diastole.
# Positive = flow out of LA (into LV). Zero during systole (valve closed).
_MV_OUTFLOW_NORMALIZED = np.array([
    [0.00, 0.00],   # Systole — valve closed
    [0.05, 0.00],
    [0.10, 0.00],
    [0.15, 0.00],
    [0.20, 0.00],
    [0.25, 0.00],
    [0.30, 0.00],
    [0.35, 0.00],   # End systole
    [0.40, 0.20],   # Valve opens
    [0.45, 0.60],
    [0.50, 1.00],   # Peak E-wave
    [0.55, 0.80],
    [0.60, 0.45],
    [0.65, 0.20],
    [0.70, 0.10],   # Diastasis
    [0.75, 0.08],
    [0.80, 0.15],
    [0.85, 0.50],
    [0.90, 0.75],   # Peak A-wave (atrial contraction)
    [0.95, 0.40],
    [1.00, 0.00],   # Valve closes
])

MV_PEAK_VELOCITY = 0.8  # m/s (healthy adult, E-wave peak)


def scale_waveform_to_heart_rate(
    normalized_waveform: np.ndarray,
    peak_velocity: float,
    heart_rate_bpm: float = 75.0,
    start_time: float = 0.0,
) -> np.ndarray:
    """Scale a normalized waveform to physical time and velocity.

    Parameters
    ----------
    normalized_waveform : array of shape (N, 2) with columns [norm_time, norm_velocity]
    peak_velocity : peak velocity in m/s
    heart_rate_bpm : heart rate in beats per minute
    start_time : offset time in seconds

    Returns
    -------
    array of shape (N, 2) with columns [time_seconds, velocity_m_s]
    """
    cardiac_cycle = 60.0 / heart_rate_bpm  # seconds
    scaled = normalized_waveform.copy()
    scaled[:, 0] = normalized_waveform[:, 0] * cardiac_cycle + start_time
    scaled[:, 1] = normalized_waveform[:, 1] * peak_velocity
    return scaled


def get_la_boundary_conditions(
    heart_rate_bpm: float = 75.0,
    pv_peak_velocity: float = PV_PEAK_VELOCITY,
    mv_peak_velocity: float = MV_PEAK_VELOCITY,
    num_cycles: int = 1,
) -> Dict:
    """Get complete LA boundary conditions scaled to patient parameters.

    Returns dict with:
        - pv_inflow: array (N, 2) of [time, velocity] for pulmonary vein inlet
        - mv_outflow: array (N, 2) of [time, velocity] for mitral valve outlet
        - pv_flow_split: dict of PV name → fraction
        - blood_properties: dict with density, viscosity
        - cardiac_cycle_s: float
    """
    cardiac_cycle = 60.0 / heart_rate_bpm
    all_pv = []
    all_mv = []

    for cycle in range(num_cycles):
        start = cycle * cardiac_cycle
        pv = scale_waveform_to_heart_rate(
            _PV_INFLOW_NORMALIZED, pv_peak_velocity, heart_rate_bpm, start
        )
        mv = scale_waveform_to_heart_rate(
            _MV_OUTFLOW_NORMALIZED, mv_peak_velocity, heart_rate_bpm, start
        )
        # Skip first point of subsequent cycles to avoid duplicate times
        if cycle > 0:
            pv = pv[1:]
            mv = mv[1:]
        all_pv.append(pv)
        all_mv.append(mv)

    return {
        "pv_inflow": np.vstack(all_pv),
        "mv_outflow": np.vstack(all_mv),
        "pv_flow_split": PV_FLOW_SPLIT,
        "blood_properties": {
            "density_kg_m3": BLOOD_DENSITY,
            "dynamic_viscosity_Pa_s": BLOOD_DYNAMIC_VISCOSITY,
            "kinematic_viscosity_m2_s": BLOOD_KINEMATIC_VISCOSITY,
        },
        "cardiac_cycle_s": cardiac_cycle,
        "heart_rate_bpm": heart_rate_bpm,
    }


# =============================================================================
# Left Ventricle boundary conditions (for testing with ACDC smoke test)
# =============================================================================

# Simplified LV inflow through mitral valve (same as LA outflow, from LV perspective)
_LV_INFLOW_NORMALIZED = _MV_OUTFLOW_NORMALIZED.copy()

# LV outflow through aortic valve
_AV_OUTFLOW_NORMALIZED = np.array([
    [0.00, 0.00],   # Diastole — valve closed
    [0.05, 0.30],   # Isovolumic contraction ends, valve opens
    [0.10, 0.70],
    [0.15, 1.00],   # Peak systolic ejection
    [0.20, 0.90],
    [0.25, 0.65],
    [0.30, 0.35],
    [0.35, 0.10],   # Valve closes
    [0.40, 0.00],
    [0.45, 0.00],
    [0.50, 0.00],
    [0.55, 0.00],
    [0.60, 0.00],
    [0.65, 0.00],
    [0.70, 0.00],
    [0.75, 0.00],
    [0.80, 0.00],
    [0.85, 0.00],
    [0.90, 0.00],
    [0.95, 0.00],
    [1.00, 0.00],
])

AV_PEAK_VELOCITY = 1.0  # m/s (healthy adult)


def get_lv_boundary_conditions(
    heart_rate_bpm: float = 75.0,
    mv_peak_velocity: float = MV_PEAK_VELOCITY,
    av_peak_velocity: float = AV_PEAK_VELOCITY,
    num_cycles: int = 1,
) -> Dict:
    """Get LV boundary conditions scaled to patient parameters."""
    cardiac_cycle = 60.0 / heart_rate_bpm
    all_mv = []
    all_av = []

    for cycle in range(num_cycles):
        start = cycle * cardiac_cycle
        mv = scale_waveform_to_heart_rate(
            _LV_INFLOW_NORMALIZED, mv_peak_velocity, heart_rate_bpm, start
        )
        av = scale_waveform_to_heart_rate(
            _AV_OUTFLOW_NORMALIZED, av_peak_velocity, heart_rate_bpm, start
        )
        if cycle > 0:
            mv = mv[1:]
            av = av[1:]
        all_mv.append(mv)
        all_av.append(av)

    return {
        "mv_inflow": np.vstack(all_mv),
        "av_outflow": np.vstack(all_av),
        "blood_properties": {
            "density_kg_m3": BLOOD_DENSITY,
            "dynamic_viscosity_Pa_s": BLOOD_DYNAMIC_VISCOSITY,
            "kinematic_viscosity_m2_s": BLOOD_KINEMATIC_VISCOSITY,
        },
        "cardiac_cycle_s": cardiac_cycle,
        "heart_rate_bpm": heart_rate_bpm,
    }


# =============================================================================
# OpenFOAM file writers
# =============================================================================

_OPENFOAM_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
"""


def write_openfoam_time_series(
    waveform: np.ndarray,
    output_path: str,
    direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
):
    """Write a time-velocity waveform in OpenFOAM table format.

    Writes (time (vx vy vz)) entries for use with uniformFixedValue BC.
    The scalar velocity magnitude is multiplied by the direction vector.

    Parameters
    ----------
    waveform : array of shape (N, 2) with [time, velocity_magnitude]
    output_path : path to write the table file
    direction : unit vector for flow direction (into the chamber)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise direction
    d = np.array(direction, dtype=float)
    d = d / np.linalg.norm(d)

    with open(output_path, "w") as f:
        f.write("(\n")
        for t, v in waveform:
            vx, vy, vz = v * d[0], v * d[1], v * d[2]
            f.write(f"    ({t:.6f} ({vx:.6f} {vy:.6f} {vz:.6f}))\n")
        f.write(")\n")


def write_transport_properties(output_path: str):
    """Write OpenFOAM transportProperties file for blood."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = _OPENFOAM_HEADER.format(
        class_name="dictionary",
        object_name="transportProperties",
    )
    content = f"""{header}
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] {BLOOD_KINEMATIC_VISCOSITY:.6e};

// Blood properties reference:
// density = {BLOOD_DENSITY} kg/m³
// dynamic viscosity = {BLOOD_DYNAMIC_VISCOSITY} Pa·s
// kinematic viscosity = nu = mu/rho = {BLOOD_KINEMATIC_VISCOSITY:.6e} m²/s
"""
    with open(output_path, "w") as f:
        f.write(content)
