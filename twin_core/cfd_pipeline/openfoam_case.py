"""
Generate a complete OpenFOAM case directory from an STL mesh.

Creates all necessary files for a pulsatile blood flow simulation using
pimpleFoam, including mesh generation (snappyHexMesh), boundary conditions,
solver settings, and WSS post-processing.

Usage:
    python -m twin_core.cfd_pipeline.openfoam_case \
        --stl cfd_meshes/patient006_frame01/LV.stl \
        --output cfd_cases/patient006_frame01 \
        --chamber lv \
        --heart-rate 75

    python -m twin_core.cfd_pipeline.openfoam_case \
        --stl cfd_meshes/patient006_frame01/LA.stl \
        --output cfd_cases/patient006_frame01 \
        --chamber la
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from twin_core.cfd_pipeline.prepare_cfd_mesh import compute_bounding_box
from twin_core.cfd_pipeline.boundary_conditions import (
    BLOOD_KINEMATIC_VISCOSITY,
    BLOOD_DENSITY,
    BLOOD_DYNAMIC_VISCOSITY,
    get_la_boundary_conditions,
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
    """Write content to a file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write(content)


def generate_block_mesh_dict(bbox: Dict, cell_size: float = 2.0) -> str:
    """Generate blockMeshDict with a background hex mesh enclosing the STL.

    Parameters
    ----------
    bbox : bounding box dict from compute_bounding_box()
    cell_size : target cell size in mm
    """
    # Add 10% padding around the geometry
    padding = [0.1 * e for e in bbox["extents"]]
    xmin = bbox["min"][0] - padding[0]
    ymin = bbox["min"][1] - padding[1]
    zmin = bbox["min"][2] - padding[2]
    xmax = bbox["max"][0] + padding[0]
    ymax = bbox["max"][1] + padding[1]
    zmax = bbox["max"][2] + padding[2]

    nx = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_size)))
    nz = max(1, int(np.ceil((zmax - zmin) / cell_size)))

    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="blockMeshDict"
    )
    return f"""{header}
scale 0.001;  // mm to meters

vertices
(
    ({xmin:.4f} {ymin:.4f} {zmin:.4f})
    ({xmax:.4f} {ymin:.4f} {zmin:.4f})
    ({xmax:.4f} {ymax:.4f} {zmin:.4f})
    ({xmin:.4f} {ymax:.4f} {zmin:.4f})
    ({xmin:.4f} {ymin:.4f} {zmax:.4f})
    ({xmax:.4f} {ymin:.4f} {zmax:.4f})
    ({xmax:.4f} {ymax:.4f} {zmax:.4f})
    ({xmin:.4f} {ymax:.4f} {zmax:.4f})
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
    refinement_level: int = 3,
    n_surface_layers: int = 3,
) -> str:
    """Generate snappyHexMeshDict for STL-to-volume mesh conversion.

    Parameters
    ----------
    stl_filename : name of STL file in constant/triSurface/
    refinement_level : number of refinement levels near surface
    n_surface_layers : number of boundary layer cells
    """
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
            patchInfo
            {{
                type wall;
            }}
        }}
    }};

    resolveFeatureAngle 30;

    refinementRegions
    {{
    }};

    locationInMesh (0 0 0);  // Point inside the mesh (will be updated)
    allowFreeStandingZoneFaces true;
}};

snapControls
{{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      100;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}};

addLayersControls
{{
    relativeSizes   true;
    layers
    {{
        {surface_name}
        {{
            nSurfaceLayers {n_surface_layers};
        }}
    }};

    expansionRatio  1.2;
    finalLayerThickness 0.3;
    minThickness    0.1;
    nGrow           0;
    featureAngle    60;
    nRelaxIter      5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals  3;
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
    minTetQuality   -1e30;
    minArea         -1;
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


def generate_control_dict(
    cardiac_cycle: float = 0.8,
    num_cycles: int = 5,
    dt: float = 1e-4,
    write_interval: float = 0.01,
) -> str:
    """Generate controlDict with WSS function object."""
    end_time = cardiac_cycle * num_cycles
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="controlDict"
    )
    return f"""{header}
application     pimpleFoam;

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
maxCo           0.5;
maxDeltaT       {dt * 10:.6e};

functions
{{
    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         (".*");
    }}

    fieldAverage
    {{
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        timeStart       {cardiac_cycle:.4f};  // Start averaging after first cycle

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


def generate_fv_schemes() -> str:
    """Generate fvSchemes for pimpleFoam."""
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


def generate_fv_solution() -> str:
    """Generate fvSolution for pimpleFoam."""
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="fvSolution"
    )
    return f"""{header}
solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
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
    nOuterCorrectors    2;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    pRefCell            0;
    pRefValue           0;
}}
"""


def generate_decompose_par_dict(n_processors: int = 4) -> str:
    """Generate decomposeParDict for parallel runs."""
    header = _OPENFOAM_HEADER.format(
        class_name="dictionary", object_name="decomposeParDict"
    )
    return f"""{header}
numberOfSubdomains {n_processors};

method          scotch;
"""


def generate_initial_conditions_u(inlet_waveform_file: str) -> str:
    """Generate initial velocity field (0/U) with pulsatile inlet."""
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
        uniformValue    tableFile;
        uniformValueCoeffs
        {{
            file            "{inlet_waveform_file}";
            outOfBounds     repeat;
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
        type            noSlip;
    }}

    ".*"
    {{
        type            noSlip;
    }}
}}
"""


def generate_initial_conditions_p() -> str:
    """Generate initial pressure field (0/p)."""
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


def generate_run_script(n_processors: int = 4) -> str:
    """Generate shell script to run the complete OpenFOAM pipeline."""
    return f"""#!/bin/bash
# OpenFOAM cardiac CFD run script
# Generated by twin_core.cfd_pipeline.openfoam_case

set -e

echo "=== Step 1/5: Background mesh (blockMesh) ==="
blockMesh

echo "=== Step 2/5: Volume mesh from STL (snappyHexMesh) ==="
snappyHexMesh -overwrite

echo "=== Step 3/5: Decompose for parallel run ==="
decomposePar

echo "=== Step 4/5: Solve (pimpleFoam) ==="
mpirun -np {n_processors} pimpleFoam -parallel

echo "=== Step 5/5: Reconstruct and post-process ==="
reconstructPar
postProcess -func wallShearStress

echo "=== Done. Results ready for analysis. ==="
"""


def create_openfoam_case(
    stl_path: str,
    output_dir: str,
    chamber: str = "lv",
    heart_rate_bpm: float = 75.0,
    num_cycles: int = 5,
    dt: float = 1e-4,
    n_processors: int = 4,
    cell_size: float = 2.0,
    refinement_level: int = 3,
    n_surface_layers: int = 3,
):
    """Create a complete OpenFOAM case directory from an STL mesh.

    Parameters
    ----------
    stl_path : path to input STL file
    output_dir : path to create the OpenFOAM case directory
    chamber : 'la' or 'lv' (determines boundary conditions)
    heart_rate_bpm : heart rate for pulsatile BC
    num_cycles : number of cardiac cycles to simulate
    dt : initial time step (seconds)
    n_processors : number of parallel processors
    cell_size : background mesh cell size (mm)
    refinement_level : snappyHexMesh refinement near surface
    n_surface_layers : boundary layer cells
    """
    stl_path = Path(stl_path)
    case_dir = Path(output_dir)

    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    # Get geometry bounding box
    bbox = compute_bounding_box(str(stl_path))
    cardiac_cycle = 60.0 / heart_rate_bpm

    # Get boundary conditions
    if chamber == "la":
        bc = get_la_boundary_conditions(
            heart_rate_bpm=heart_rate_bpm, num_cycles=num_cycles
        )
        inlet_waveform = bc["pv_inflow"]
    elif chamber == "lv":
        bc = get_lv_boundary_conditions(
            heart_rate_bpm=heart_rate_bpm, num_cycles=num_cycles
        )
        inlet_waveform = bc["mv_inflow"]
    else:
        raise ValueError(f"Unknown chamber: {chamber}. Use 'la' or 'lv'.")

    stl_filename = stl_path.name
    print(f"\n=== Creating OpenFOAM case ===")
    print(f"  STL: {stl_path}")
    print(f"  Output: {case_dir}")
    print(f"  Chamber: {chamber.upper()}")
    print(f"  Heart rate: {heart_rate_bpm} bpm")
    print(f"  Cardiac cycle: {cardiac_cycle:.3f} s")
    print(f"  Simulation: {num_cycles} cycles ({cardiac_cycle * num_cycles:.2f} s)")
    print(f"  Processors: {n_processors}")
    print()

    # Copy STL to case
    tri_surface_dir = case_dir / "constant" / "triSurface"
    tri_surface_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(stl_path), str(tri_surface_dir / stl_filename))

    # Write inlet waveform
    waveform_path = case_dir / "constant" / "inlet_waveform.csv"
    write_openfoam_time_series(inlet_waveform, str(waveform_path))

    # Write transport properties
    write_transport_properties(str(case_dir / "constant" / "transportProperties"))

    # Write system files
    system_dir = case_dir / "system"
    _write_file(
        system_dir / "blockMeshDict",
        generate_block_mesh_dict(bbox, cell_size),
    )
    _write_file(
        system_dir / "snappyHexMeshDict",
        generate_snappy_hex_mesh_dict(stl_filename, refinement_level, n_surface_layers),
    )
    _write_file(
        system_dir / "controlDict",
        generate_control_dict(cardiac_cycle, num_cycles, dt),
    )
    _write_file(system_dir / "fvSchemes", generate_fv_schemes())
    _write_file(system_dir / "fvSolution", generate_fv_solution())
    _write_file(
        system_dir / "decomposeParDict",
        generate_decompose_par_dict(n_processors),
    )

    # Write initial conditions
    zero_dir = case_dir / "0"
    _write_file(
        zero_dir / "U",
        generate_initial_conditions_u("constant/inlet_waveform.csv"),
    )
    _write_file(zero_dir / "p", generate_initial_conditions_p())

    # Write run script
    run_script = case_dir / "run.sh"
    _write_file(run_script, generate_run_script(n_processors))
    run_script.chmod(0o755)

    print(f"  Case directory created: {case_dir}")
    print(f"  Files:")
    for f in sorted(case_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(case_dir)
            print(f"    {rel}")

    print(f"\n  To run (after installing OpenFOAM):")
    print(f"    cd {case_dir} && bash run.sh")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate OpenFOAM case directory from STL mesh"
    )
    parser.add_argument(
        "--stl", type=str, required=True,
        help="Path to input STL file"
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output directory for the OpenFOAM case"
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
        help="Number of cardiac cycles to simulate (default: 5)"
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
        "--cell-size", type=float, default=2.0,
        help="Background mesh cell size in mm (default: 2.0)"
    )
    parser.add_argument(
        "--refinement", type=int, default=3,
        help="Surface refinement level (default: 3)"
    )
    parser.add_argument(
        "--layers", type=int, default=3,
        help="Number of boundary layer cells (default: 3)"
    )

    args = parser.parse_args()

    create_openfoam_case(
        stl_path=args.stl,
        output_dir=args.output,
        chamber=args.chamber,
        heart_rate_bpm=args.heart_rate,
        num_cycles=args.num_cycles,
        dt=args.dt,
        n_processors=args.processors,
        cell_size=args.cell_size,
        refinement_level=args.refinement,
        n_surface_layers=args.layers,
    )


if __name__ == "__main__":
    main()
