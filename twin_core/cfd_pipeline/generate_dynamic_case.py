"""
Generate a dynamic-mesh OpenFOAM case for cardiac CFD with prescribed wall motion.

Takes a series of topologically consistent, multi-region STL files (from the
temporal registration pipeline) and generates a complete OpenFOAM case that
uses displacementLaplacian to prescribe wall motion.

Approach:
    1. Mesh the initial geometry (ED frame) with blockMesh + snappyHexMesh
    2. Precompute wall vertex displacements between consecutive frames
    3. Use dynamicMotionSolverFvMesh with displacementLaplacian
    4. Wall boundary reads displacement from a timeVaryingPointDisplacement BC
    5. OpenFOAM interpolates between displacement snapshots and diffuses
       the boundary displacement into the interior mesh

Usage:
    python -m twin_core.cfd_pipeline.generate_dynamic_case \
        --stl-dir test_3d/cfd_dynamic/stl_series \
        --template-stl test_3d/cfd_dynamic/template_LV_valves.stl \
        --output test_3d/cfd_dynamic/case \
        --heart-rate 75 --num-cycles 5

Dependencies:
    - OpenFOAM (v2206+ or Foundation v10+) on the target machine
    - trimesh, numpy (for case generation)
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from twin_core.cfd_pipeline.prepare_cfd_mesh import compute_bounding_box
from twin_core.cfd_pipeline.boundary_conditions import (
    BLOOD_KINEMATIC_VISCOSITY,
    BLOOD_DENSITY,
    BLOOD_DYNAMIC_VISCOSITY,
    get_lv_boundary_conditions,
    write_openfoam_time_series,
    write_transport_properties,
)


_OPENFOAM_HEADER = """\
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
"""


def _write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(content)


# ============================================================================
# Displacement precomputation
# ============================================================================

def compute_displacement_series(
    stl_dir: str,
    n_frames: int,
    cardiac_cycle: float,
) -> Tuple[List[float], List[np.ndarray]]:
    """Load all STL frames and compute vertex positions at each time step.

    All STLs must have identical vertex count and face connectivity.
    Coordinates must be in metres (already scaled by cut_valve_openings).

    Returns
    -------
    times : list of float, time in seconds for each frame
    vertex_positions : list of (N, 3) arrays, vertex positions per frame
    """
    stl_dir = Path(stl_dir)
    dt_frame = cardiac_cycle / n_frames

    times = []
    vertex_positions = []

    for i in range(n_frames):
        stl_file = stl_dir / f"patient006_frame{i:02d}_LV.stl"
        if not stl_file.exists():
            raise FileNotFoundError(f"Missing STL: {stl_file}")

        # Load multi-region STL — trimesh reads all solids together
        mesh = trimesh.load(str(stl_file), force="mesh")
        times.append(i * dt_frame)
        vertex_positions.append(mesh.vertices.copy())

    # Verify consistency
    n_verts = len(vertex_positions[0])
    for i, vp in enumerate(vertex_positions):
        if len(vp) != n_verts:
            raise ValueError(
                f"Frame {i} has {len(vp)} vertices, expected {n_verts}"
            )

    return times, vertex_positions


def write_point_displacement_series(
    case_dir: Path,
    times: List[float],
    vertex_positions: List[np.ndarray],
    cardiac_cycle: float,
):
    """Write pointDisplacement boundary data for each time snapshot.

    For timeVaryingUniformFixedValue or similar BCs, OpenFOAM reads
    displacement data from time directories under constant/boundaryData/.

    However, for prescribed point-by-point wall displacement, the most
    robust approach is to write displacement fields in numbered directories
    that the pointDisplacement BC reads via timeVarying.

    We write the displacement (relative to frame 0) at each time snapshot.
    """
    ref_positions = vertex_positions[0]  # Reference = frame 0 (initial mesh)
    boundary_data_dir = case_dir / "constant" / "boundaryData" / "wall"
    boundary_data_dir.mkdir(parents=True, exist_ok=True)

    # Write points file (vertex coordinates on the wall patch)
    # This will be matched against the mesh after snappyHexMesh
    # For now, write the reference positions
    points_path = boundary_data_dir / "points"
    with open(points_path, "w") as f:
        f.write(f"{len(ref_positions)}\n(\n")
        for v in ref_positions:
            f.write(f"({v[0]:.8f} {v[1]:.8f} {v[2]:.8f})\n")
        f.write(")\n")

    # Write displacement at each time
    for t, positions in zip(times, vertex_positions):
        displacements = positions - ref_positions
        time_dir = boundary_data_dir / f"{t:.8f}"
        time_dir.mkdir(parents=True, exist_ok=True)

        disp_path = time_dir / "pointDisplacement"
        with open(disp_path, "w") as f:
            f.write(f"{len(displacements)}\n(\n")
            for d in displacements:
                f.write(f"({d[0]:.8f} {d[1]:.8f} {d[2]:.8f})\n")
            f.write(")\n")

    # Also write displacement for t = cardiac_cycle (same as frame 0 = zero)
    # This ensures smooth cyclic interpolation
    t_end = cardiac_cycle
    time_dir = boundary_data_dir / f"{t_end:.8f}"
    time_dir.mkdir(parents=True, exist_ok=True)
    disp_path = time_dir / "pointDisplacement"
    with open(disp_path, "w") as f:
        zeros = np.zeros_like(ref_positions)
        f.write(f"{len(zeros)}\n(\n")
        for d in zeros:
            f.write(f"({d[0]:.8f} {d[1]:.8f} {d[2]:.8f})\n")
        f.write(")\n")

    print(f"  Wrote {len(times) + 1} displacement snapshots to {boundary_data_dir}")


# ============================================================================
# OpenFOAM file generators (dynamic mesh specific)
# ============================================================================

def generate_dynamic_mesh_dict() -> str:
    """Generate dynamicMeshDict for prescribed wall motion."""
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="dynamicMeshDict"
    )
    return f"""{header}
mover
{{
    type            motionSolver;

    motionSolver    displacementSBRStress;

    displacementSBRStressCoeffs
    {{
        diffusivity     quadratic inverseDistance 1 (wall);
    }}
}}
"""


def generate_point_displacement_bc() -> str:
    """Generate 0/pointDisplacement for dynamic mesh."""
    header = _OPENFOAM_HEADER.format(
        class_name="pointVectorField", object_name="pointDisplacement"
    )
    return f"""{header}
dimensions      [0 1 0 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    wall
    {{
        type            timeVaryingMappedFixedValue;
        offset          (0 0 0);
        setAverage      off;
        mapMethod       nearest;
        value           uniform (0 0 0);
    }}

    inlet
    {{
        type            timeVaryingMappedFixedValue;
        offset          (0 0 0);
        setAverage      off;
        mapMethod       nearest;
        value           uniform (0 0 0);
    }}

    outlet
    {{
        type            timeVaryingMappedFixedValue;
        offset          (0 0 0);
        setAverage      off;
        mapMethod       nearest;
        value           uniform (0 0 0);
    }}

    ".*"
    {{
        type            fixedValue;
        value           uniform (0 0 0);
    }}
}}
"""


def generate_control_dict_dynamic(
    cardiac_cycle: float = 0.8,
    num_cycles: int = 5,
    dt: float = 1e-6,
    write_interval: float = 0.01,
) -> str:
    """Generate controlDict for dynamic mesh pimpleFoam."""
    end_time = cardiac_cycle * num_cycles
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="controlDict"
    )
    return f"""{header}
application     foamRun;
solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {end_time:.4f};

deltaT          {dt:.6e};

writeControl    adjustableRunTime;
writeInterval   {write_interval:.4f};

purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;

timeFormat      general;
timePrecision   8;

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           0.3;
maxDeltaT       {dt * 1000:.6e};

libs            ("libfvMotionSolvers.so");

functions
{{
    writeMeshPoints
    {{
        type            writeObjects;
        libs            ("libutilityFunctionObjects.so");
        writeControl    writeTime;
        objects         (points);
    }}

    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         ("wall");
    }}

    fieldAverage
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        timeStart       {cardiac_cycle:.4f};

        fields
        (
            U
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
            p
            {{
                mean        on;
                prime2Mean  off;
                base        time;
            }}
            wallShearStress
            {{
                mean        on;
                prime2Mean  on;
                base        time;
            }}
        );
    }}
}};
"""


def generate_fv_schemes_dynamic() -> str:
    """Generate fvSchemes with mesh motion schemes."""
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="fvSchemes"
    )
    return f"""{header}
ddtSchemes
{{
    default         backward;
}}

gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
    // Mesh motion diffusion
    laplacian(diffusivity,cellDisplacement) Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}
"""


def generate_fv_solution_dynamic() -> str:
    """Generate fvSolution with mesh motion solver."""
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="fvSolution"
    )
    return f"""{header}
solvers
{{
    "cellDisplacement.*"
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.01;
    }}

    pcorr
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-02;
        relTol          0;
    }}

    pcorrFinal
    {{
        $pcorr;
        tolerance       1e-04;
    }}

    p
    {{
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.01;
    }}

    pFinal
    {{
        $p;
        relTol          0;
    }}

    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.01;
    }}

    UFinal
    {{
        $U;
        relTol          0;
    }}
}}

PIMPLE
{{
    nOuterCorrectors    3;
    nCorrectors         2;
    nNonOrthogonalCorrectors 2;
    pRefCell            0;
    pRefValue           0;
    correctPhi          no;
    moveMeshOuterCorrectors yes;
}}

relaxationFactors
{{
    equations
    {{
        "U.*"           0.7;
        "p.*"           0.3;
    }}
}}
"""


def generate_block_mesh_dict(bbox: Dict, cell_size: float = 2.0) -> str:
    """Generate blockMeshDict — bbox already in metres."""
    padding = [0.1 * e for e in bbox["extents"]]
    xmin = bbox["min"][0] - padding[0]
    ymin = bbox["min"][1] - padding[1]
    zmin = bbox["min"][2] - padding[2]
    xmax = bbox["max"][0] + padding[0]
    ymax = bbox["max"][1] + padding[1]
    zmax = bbox["max"][2] + padding[2]

    cs = cell_size * 0.001  # mm to metres
    nx = max(1, int(np.ceil((xmax - xmin) / cs)))
    ny = max(1, int(np.ceil((ymax - ymin) / cs)))
    nz = max(1, int(np.ceil((zmax - zmin) / cs)))

    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="blockMeshDict"
    )
    return f"""{header}
vertices
(
    ({xmin:.6f} {ymin:.6f} {zmin:.6f})
    ({xmax:.6f} {ymin:.6f} {zmin:.6f})
    ({xmax:.6f} {ymax:.6f} {zmin:.6f})
    ({xmin:.6f} {ymax:.6f} {zmin:.6f})
    ({xmin:.6f} {ymin:.6f} {zmax:.6f})
    ({xmax:.6f} {ymin:.6f} {zmax:.6f})
    ({xmax:.6f} {ymax:.6f} {zmax:.6f})
    ({xmin:.6f} {ymax:.6f} {zmax:.6f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    outerWall
    {{
        type wall;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (2 3 7 6)
            (1 2 6 5)
            (0 4 7 3)
        );
    }}
);

mergePatchPairs
(
);
"""


def generate_snappy_hex_mesh_dict(
    stl_filename: str,
    bbox: Dict,
    refinement_level: int = 2,
    n_surface_layers: int = 3,
) -> str:
    """Generate snappyHexMeshDict for multi-region STL."""
    surface_name = Path(stl_filename).stem
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="snappyHexMeshDict"
    )
    return f"""{header}
castellatedMesh true;
snap            true;
addLayers       true;

geometry
{{
    {stl_filename}
    {{
        type triSurfaceMesh;
        name {surface_name};
        regions
        {{
            wall   {{ name wall; }}
            inlet  {{ name inlet; }}
            outlet {{ name outlet; }}
        }}
    }}
}};

castellatedMeshControls
{{
    maxLocalCells   1000000;
    maxGlobalCells  4000000;
    minRefinementCells 10;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 3;

    features
    (
    );

    refinementSurfaces
    {{
        {surface_name}
        {{
            level ({refinement_level} {refinement_level});
            regions
            {{
                wall
                {{
                    level ({refinement_level} {refinement_level});
                    patchInfo {{ type wall; }}
                }}
                inlet
                {{
                    level ({refinement_level + 1} {refinement_level + 2});
                    patchInfo {{ type patch; }}
                }}
                outlet
                {{
                    level ({refinement_level + 1} {refinement_level + 2});
                    patchInfo {{ type patch; }}
                }}
            }}
        }}
    }};

    resolveFeatureAngle 30;
    refinementRegions {{}};

    locationInMesh ({bbox['center'][0]:.6f} {bbox['center'][1]:.6f} {bbox['center'][2]:.6f});
    allowFreeStandingZoneFaces true;
}};

snapControls
{{
    nSmoothPatch    5;
    tolerance       2.0;
    nSolveIter      200;
    nRelaxIter      8;
    nFeatureSnapIter 10;
    implicitFeatureSnap true;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}};

addLayersControls
{{
    relativeSizes   true;
    layers
    {{
        "wall"
        {{
            nSurfaceLayers {n_surface_layers};
        }}
    }};

    expansionRatio  1.2;
    finalLayerThickness 0.3;
    minThickness    0.05;
    nGrow           0;
    featureAngle    130;
    nRelaxIter      10;
    nSmoothSurfaceNormals 3;
    nSmoothNormals  5;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter      50;
}};

meshQualityControls
{{
    maxNonOrtho     65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave      80;
    minVol          1e-13;
    minTetQuality   1e-15;
    minArea         1e-08;
    minTwist        0.02;
    minDeterminant  0.001;
    minFaceWeight   0.05;
    minVolRatio     0.01;
    minTriangleTwist -1;
    nSmoothScale    4;
    errorReduction  0.75;
}};

writeFormat ascii;
mergeTolerance 1e-6;
"""


def generate_surface_features_dict(stl_filename: str) -> str:
    """Generate surfaceFeaturesDict for edge refinement (OpenFOAM 11 format)."""
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="surfaceFeaturesDict"
    )
    return f"""{header}
surfaces
(
    "{stl_filename}"
);

includedAngle   150;

writeObj        yes;
"""


def generate_initial_conditions_u(inlet_waveform_file: str) -> str:
    """Generate 0/U with pulsatile inlet and moving wall."""
    header = _OPENFOAM_HEADER.format(
        class_name="volVectorField", object_name="U"
    )
    return f"""{header}
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            uniformFixedValue;
        uniformValue
        {{
            type    tableFile;
            file    "{inlet_waveform_file}";
            outOfBounds repeat;
        }}
    }}

    outlet
    {{
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform (0 0 0);
    }}

    wall
    {{
        type            movingWallVelocity;
        value           uniform (0 0 0);
    }}

    ".*"
    {{
        type            noSlip;
    }}
}}
"""


def generate_initial_conditions_p() -> str:
    """Generate 0/p."""
    header = _OPENFOAM_HEADER.format(
        class_name="volScalarField", object_name="p"
    )
    return f"""{header}
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}

    outlet
    {{
        type            fixedValue;
        value           uniform 0;
    }}

    wall
    {{
        type            zeroGradient;
    }}

    ".*"
    {{
        type            zeroGradient;
    }}
}}
"""


def generate_decompose_par_dict(n_processors: int = 32) -> str:
    """Generate decomposeParDict."""
    import math
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="decomposeParDict"
    )
    n = n_processors
    nz = 1
    while nz * nz * nz <= n:
        if n % nz == 0:
            best_z = nz
        nz += 1
    nz = best_z
    rem = n // nz
    ny = int(math.isqrt(rem))
    while rem % ny != 0:
        ny -= 1
    nx = rem // ny
    return f"""{header}
numberOfSubdomains {n_processors};

method          simple;

simpleCoeffs
{{
    n           ({nx} {ny} {nz});
    delta       0.001;
}}
"""


def generate_run_script_dynamic(n_processors: int = 32) -> str:
    """Generate run script for dynamic mesh case."""
    return f"""#!/bin/bash
# OpenFOAM dynamic-mesh cardiac CFD run script
# Generated by twin_core.cfd_pipeline.generate_dynamic_case

set -e

echo "=== Step 1/5: Background mesh (blockMesh) ==="
blockMesh

echo "=== Step 2/5: Volume mesh from STL (snappyHexMesh) ==="
snappyHexMesh -overwrite

echo "=== Step 3/5: Decompose for parallel run ==="
decomposePar

echo "=== Step 4/5: Solve (foamRun incompressibleFluid with dynamic mesh) ==="
mpirun -np {n_processors} foamRun -solver incompressibleFluid -parallel

echo "=== Step 5/5: Reconstruct ==="
reconstructPar

echo "=== Done. Results ready for analysis. ==="
"""


# ============================================================================
# Main case generator
# ============================================================================

def create_dynamic_case(
    stl_dir: str,
    template_stl: str,
    output_dir: str,
    heart_rate_bpm: float = 75.0,
    num_cycles: int = 5,
    dt: float = 1e-6,
    n_processors: int = 32,
    cell_size: float = 2.0,
    refinement_level: int = 2,
    n_surface_layers: int = 3,
    displacement_scale: float = 1.0,
):
    """Create a complete dynamic-mesh OpenFOAM case.

    Parameters
    ----------
    stl_dir : directory containing frame STL series (patient006_frameNN_LV.stl)
    template_stl : multi-region STL for the initial mesh (frame00 or frame01)
    output_dir : output case directory
    heart_rate_bpm : heart rate in bpm
    num_cycles : number of cardiac cycles
    dt : initial time step
    n_processors : parallel processors (VM32 has 32)
    displacement_scale : scale factor for displacement (0.1 = 10%, for testing)
    """
    stl_dir = Path(stl_dir)
    case_dir = Path(output_dir)

    cardiac_cycle = 60.0 / heart_rate_bpm
    n_frames = 28  # from ACDC

    print(f"\n=== Creating Dynamic Mesh OpenFOAM Case ===")
    print(f"  STL series: {stl_dir} ({n_frames} frames)")
    print(f"  Template STL: {template_stl}")
    print(f"  Output: {case_dir}")
    print(f"  Heart rate: {heart_rate_bpm} bpm")
    print(f"  Cardiac cycle: {cardiac_cycle:.3f} s")
    print(f"  Time per frame: {cardiac_cycle / n_frames * 1000:.2f} ms")
    print(f"  Simulation: {num_cycles} cycles ({cardiac_cycle * num_cycles:.2f} s)")
    print(f"  Processors: {n_processors}")
    print()

    # 1. Load and precompute displacements
    print("  Step 1: Computing displacement series...")
    times, vertex_positions = compute_displacement_series(
        str(stl_dir), n_frames, cardiac_cycle
    )
    n_verts = len(vertex_positions[0])
    print(f"    {n_frames} frames, {n_verts} vertices per frame")

    # Apply displacement scaling (for testing mesh motion stability)
    if displacement_scale != 1.0:
        ref = vertex_positions[0]
        vertex_positions = [
            ref + (vp - ref) * displacement_scale for vp in vertex_positions
        ]
        print(f"    Displacement scale: {displacement_scale:.1%}")

    # Displacement stats
    ref = vertex_positions[0]
    max_disps = [np.linalg.norm(vp - ref, axis=1).max() for vp in vertex_positions]
    print(f"    Max displacement from frame00: {max(max_disps):.2f} m "
          f"({max(max_disps) * 1000:.1f} mm)")

    # 2. Copy template STL for meshing
    print("\n  Step 2: Setting up case directory...")
    tri_surface_dir = case_dir / "constant" / "triSurface"
    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    stl_filename = Path(template_stl).name
    dest_stl = tri_surface_dir / stl_filename
    shutil.copy2(str(template_stl), str(dest_stl))
    print(f"    Copied template STL: {stl_filename}")

    # 3. Compute bounding box (template is in metres)
    bbox = compute_bounding_box(str(template_stl))

    # 4. Write displacement boundary data
    print("\n  Step 3: Writing displacement series...")
    # Write for all three moving patches (wall, inlet, outlet)
    # All patches move together since they're part of the same surface
    for patch_name in ["wall", "inlet", "outlet"]:
        boundary_data_dir = case_dir / "constant" / "boundaryData" / patch_name
        boundary_data_dir.mkdir(parents=True, exist_ok=True)

        # Points file
        points_path = boundary_data_dir / "points"
        with open(points_path, "w") as f:
            f.write(f"{n_verts}\n(\n")
            for v in ref:
                f.write(f"({v[0]:.8f} {v[1]:.8f} {v[2]:.8f})\n")
            f.write(")\n")

        # Displacement at each time — repeat for all cardiac cycles
        for cycle in range(num_cycles):
            t_offset = cycle * cardiac_cycle
            for t, positions in zip(times, vertex_positions):
                displacements = positions - ref
                t_abs = t + t_offset
                time_dir = boundary_data_dir / f"{t_abs:.8f}"
                time_dir.mkdir(parents=True, exist_ok=True)
                disp_path = time_dir / "pointDisplacement"
                with open(disp_path, "w") as f:
                    f.write(f"{len(displacements)}\n(\n")
                    for d in displacements:
                        f.write(f"({d[0]:.8f} {d[1]:.8f} {d[2]:.8f})\n")
                    f.write(")\n")

            # End-of-cycle = zero displacement (back to start)
            t_end = (cycle + 1) * cardiac_cycle
            time_dir = boundary_data_dir / f"{t_end:.8f}"
            time_dir.mkdir(parents=True, exist_ok=True)
            disp_path = time_dir / "pointDisplacement"
            with open(disp_path, "w") as f:
                zeros = np.zeros((n_verts, 3))
                f.write(f"{n_verts}\n(\n")
                for d in zeros:
                    f.write(f"({d[0]:.8f} {d[1]:.8f} {d[2]:.8f})\n")
                f.write(")\n")

    total_snapshots = num_cycles * (n_frames + 1)
    print(f"    Wrote displacement data for 3 patches × {total_snapshots} time steps ({num_cycles} cycles)")

    # 5. Get boundary conditions
    bc = get_lv_boundary_conditions(
        heart_rate_bpm=heart_rate_bpm, num_cycles=num_cycles
    )

    # Inlet flow direction
    from twin_core.cfd_pipeline.cut_valve_openings import find_chamber_base
    template_mesh = trimesh.load(str(template_stl), force="mesh")
    _, base_normal, _ = find_chamber_base(template_mesh)
    inlet_direction = tuple(-base_normal)
    print(f"    Inlet direction: ({inlet_direction[0]:.3f}, {inlet_direction[1]:.3f}, {inlet_direction[2]:.3f})")

    # 6. Write all OpenFOAM files
    print("\n  Step 4: Writing OpenFOAM configuration files...")
    system_dir = case_dir / "system"
    zero_dir = case_dir / "0"

    # constant/
    write_openfoam_time_series(
        bc["mv_inflow"],
        str(case_dir / "constant" / "inlet_waveform.csv"),
        direction=inlet_direction,
    )
    write_transport_properties(str(case_dir / "constant" / "transportProperties"))

    turb_header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="turbulenceProperties"
    )
    _write_file(
        case_dir / "constant" / "turbulenceProperties",
        f"{turb_header}\nsimulationType  laminar;\n",
    )

    # constant/dynamicMeshDict
    _write_file(
        case_dir / "constant" / "dynamicMeshDict",
        generate_dynamic_mesh_dict(),
    )

    # system/
    _write_file(
        system_dir / "blockMeshDict",
        generate_block_mesh_dict(bbox, cell_size),
    )
    _write_file(
        system_dir / "snappyHexMeshDict",
        generate_snappy_hex_mesh_dict(stl_filename, bbox, refinement_level, n_surface_layers),
    )
    _write_file(
        system_dir / "controlDict",
        generate_control_dict_dynamic(cardiac_cycle, num_cycles, dt),
    )
    _write_file(system_dir / "fvSchemes", generate_fv_schemes_dynamic())
    _write_file(system_dir / "fvSolution", generate_fv_solution_dynamic())
    _write_file(
        system_dir / "decomposeParDict",
        generate_decompose_par_dict(n_processors),
    )

    # 0/
    _write_file(
        zero_dir / "U",
        generate_initial_conditions_u("constant/inlet_waveform.csv"),
    )
    _write_file(zero_dir / "p", generate_initial_conditions_p())
    _write_file(zero_dir / "pointDisplacement", generate_point_displacement_bc())

    # run.sh
    run_script = case_dir / "run.sh"
    _write_file(run_script, generate_run_script_dynamic(n_processors))
    run_script.chmod(0o755)

    # Summary
    print(f"\n  Case directory created: {case_dir}")
    print(f"  Files:")
    for f in sorted(case_dir.rglob("*")):
        if f.is_file() and "boundaryData" not in str(f):
            rel = f.relative_to(case_dir)
            print(f"    {rel}")
    bd_files = len(list((case_dir / "constant" / "boundaryData").rglob("*")))
    print(f"    constant/boundaryData/ ({bd_files} files)")

    print(f"\n  To run on VM32:")
    print(f"    1. scp -r {case_dir} user@vm32:~/cfd_runs/")
    print(f"    2. ssh user@vm32")
    print(f"    3. cd ~/cfd_runs/{case_dir.name} && bash run.sh")
    print(f"\n  Check after 30-60 min — if it passes the first few hundred")
    print(f"  time steps without crashing, it will likely complete.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic-mesh OpenFOAM case for cardiac CFD"
    )
    parser.add_argument(
        "--stl-dir", type=str, required=True,
        help="Directory containing frame STL series"
    )
    parser.add_argument(
        "--template-stl", type=str, required=True,
        help="Multi-region template STL for initial mesh"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output case directory"
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
        "--dt", type=float, default=1e-6,
        help="Initial time step (default: 1e-6)"
    )
    parser.add_argument(
        "--processors", type=int, default=32,
        help="Number of parallel processors (default: 32)"
    )
    parser.add_argument(
        "--cell-size", type=float, default=2.0,
        help="Background cell size in mm (default: 2.0)"
    )
    parser.add_argument(
        "--refinement", type=int, default=2,
        help="Surface refinement level (default: 2)"
    )
    parser.add_argument(
        "--layers", type=int, default=3,
        help="Boundary layer cells (default: 3)"
    )
    parser.add_argument(
        "--displacement-scale", type=float, default=1.0,
        help="Scale factor for wall displacement (0.1 = 10%%, for testing)"
    )

    args = parser.parse_args()

    create_dynamic_case(
        stl_dir=args.stl_dir,
        template_stl=args.template_stl,
        output_dir=args.output,
        heart_rate_bpm=args.heart_rate,
        num_cycles=args.num_cycles,
        dt=args.dt,
        n_processors=args.processors,
        cell_size=args.cell_size,
        refinement_level=args.refinement,
        n_surface_layers=args.layers,
        displacement_scale=args.displacement_scale,
    )


if __name__ == "__main__":
    main()
