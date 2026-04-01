# OpenFOAM 11 (Foundation) Command Reference
## For Cardiac Digital Twin CFD Project

**CRITICAL**: OpenFOAM 11 (Foundation version, openfoam.org) introduced MODULAR SOLVERS.
Traditional application solvers (pimpleFoam, simpleFoam, etc.) are REPLACED by `foamRun`
with solver modules. Old solver names still work as backward-compatible wrapper scripts
but may behave differently. Always prefer the new syntax.

**This is NOT ESI/OpenCFD OpenFOAM (openfoam.com, e.g. v2206, v2306).** The two OpenFOAMs
have diverged significantly. Do not mix documentation between them.

---

## 1. SOLVER COMMANDS

### New (v11) modular solver syntax:
```bash
# General solver — loads module specified in controlDict or via -solver flag
foamRun
foamRun -solver incompressibleFluid

# Multi-region solver (e.g., conjugate heat transfer)
foamMultiRun
```

### Solver module mapping (old → new):
| Old Application Solver | New Solver Module | Usage |
|------------------------|-------------------|-------|
| pimpleFoam | incompressibleFluid | `foamRun` or `foamRun -solver incompressibleFluid` |
| pisoFoam | incompressibleFluid | same as above |
| simpleFoam | incompressibleFluid | same (detects steady from controlDict) |
| icoFoam | incompressibleFluid | same |
| rhoSimpleFoam | fluid | `foamRun -solver fluid` |
| rhoPimpleFoam | fluid | same |
| buoyantFoam | fluid | same |
| reactingFoam | multicomponentFluid | `foamRun -solver multicomponentFluid` |
| interFoam | incompressibleVoF | `foamRun -solver incompressibleVoF` |
| compressibleInterFoam | compressibleVoF | `foamRun -solver compressibleVoF` |
| chtMultiRegionFoam | fluid + solid | `foamMultiRun` |
| moveMesh | movingMesh | `foamRun -solver movingMesh` |
| scalarTransportFoam | functions | `foamRun -solver functions` |

### controlDict solver specification (preferred over command line):
```c
// In system/controlDict:
application     foamRun;

solver          incompressibleFluid;  // NEW in v11 — specify solver module here
```

### Backward-compatible (old names still work as scripts):
```bash
# These still work — they internally call foamRun with the right module
pimpleFoam          # → foamRun -solver incompressibleFluid
simpleFoam          # → foamRun -solver incompressibleFluid
```

### Parallel execution:
```bash
# New way (recommended)
mpirun -np 32 foamRun -parallel

# Old way (still works)
mpirun -np 32 pimpleFoam -parallel
```

---

## 2. MESHING COMMANDS

```bash
# Background hex mesh from blockMeshDict
blockMesh

# Feature edge extraction (for snappyHexMesh)
surfaceFeatures

# Automatic mesh from STL (castellate + snap + layers)
snappyHexMesh
snappyHexMesh -overwrite    # overwrite existing mesh (most common)

# NEW in v11: auto-configure snappyHexMesh input files
snappyHexMeshConfig         # generates blockMeshDict, snappyHexMeshDict, surfaceFeaturesDict

# Check mesh quality
checkMesh
checkMesh -allGeometry -allTopology   # thorough check

# Mesh manipulation
transformPoints "scale=(0.001 0.001 0.001)"   # scale mesh (e.g., mm to m)
transformPoints "translate=(1 0 0)"
topoSet                     # create cell/face/point sets from geometric criteria
createPatch                 # create patches from face sets
splitMesh                   # split mesh at specified face zone
mergeMeshes                 # merge two meshes
```

---

## 3. PARALLEL COMMANDS

```bash
# Decompose mesh for parallel run
decomposePar

# Reconstruct from parallel results
reconstructPar
reconstructPar -time 0:0.5          # reconstruct specific time range
reconstructPar -latestTime          # reconstruct only latest time step

# Redistribute mesh (dynamic load balancing)
redistributePar
```

### decomposeParDict (system/decomposeParDict):
```c
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}

numberOfSubdomains  32;

// Methods: simple, scotch, hierarchical, manual
method          scotch;    // scotch is generally best for complex geometries

// Only needed for simple method:
simpleCoeffs
{
    n           (4 4 2);
    delta       0.001;
}
```

---

## 4. POST-PROCESSING COMMANDS

```bash
# NEW in v11: replaces postProcess
foamPostProcess
foamPostProcess -func wallShearStress
foamPostProcess -func "mag(U)"

# Old (still works in v11 as wrapper):
postProcess -func wallShearStress

# List available function objects:
foamPostProcess -list

# Run function objects during simulation — add to controlDict:
# functions { ... }

# Launch ParaView (create empty .foam file first)
touch case.foam
paraFoam
# or just: paraview case.foam
```

### Common function objects for cardiac CFD:
```c
// In system/controlDict → functions { }
functions
{
    wallShearStress
    {
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         ("wall");
    }

    fieldAverage
    {
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        timeStart       0.8;    // skip first cardiac cycle

        fields
        (
            U
            {
                mean        on;
                prime2Mean  on;
                base        time;
            }
            p
            {
                mean        on;
                prime2Mean  off;
                base        time;
            }
        );
    }

    // Volume-averaged quantities
    volAverage
    {
        type            volFieldValue;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        operation       volAverage;
        fields          (U p);
    }
}
```

---

## 5. INFORMATION & DEBUGGING COMMANDS

```bash
# NEW in v11: list all available models, BCs, etc.
foamToC
foamToC -table incompressibleMomentumTransportModel   # list turbulence models

# Get info about a model, BC, or function object
foamInfo wallShearStress
foamInfo fixedValue
foamInfo incompressibleFluid

# Read/modify dictionary entries
foamDictionary system/controlDict -entry endTime
foamDictionary system/controlDict -entry endTime -set 4.0
foamDictionary constant/transportProperties -entry nu

# List time directories
foamListTimes
foamListTimes -latestTime
```

---

## 6. SURFACE & GEOMETRY UTILITIES

```bash
# Surface operations
surfaceCheck constant/triSurface/LV.stl          # check STL quality
surfaceOrient constant/triSurface/LV.stl          # fix normals
surfaceTransformPoints "scale=(0.001 0.001 0.001)" input.stl output.stl
surfaceConvert input.stl output.obj               # convert formats

# Extract surface from volume mesh
foamToSurface
```

---

## 7. DYNAMIC MESH (for moving wall cardiac CFD)

### dynamicMeshDict (constant/dynamicMeshDict):
```c
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      dynamicMeshDict;
}

// For prescribed wall motion:
mover
{
    type    motionSolver;

    motionSolver    displacementLaplacian;

    displacementLaplacianCoeffs
    {
        diffusivity     inverseDistance (wall);
    }
}
```

**NOTE on v11 dynamic mesh syntax**: OpenFOAM 11 changed the dynamicMeshDict format.
The old `dynamicFvMesh dynamicMotionSolverFvMesh;` syntax is REPLACED by:
```c
// OLD (pre-v11):
dynamicFvMesh    dynamicMotionSolverFvMesh;
motionSolverLibs ("libfvMotionSolvers.so");
motionSolver     displacementLaplacian;

// NEW (v11):
mover
{
    type    motionSolver;
    motionSolver    displacementLaplacian;
    // ...
}
```

### 0/pointDisplacement:
```c
FoamFile
{
    version     2.0;
    format      ascii;
    class       pointVectorField;
    object      pointDisplacement;
}

dimensions      [0 1 0 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    wall
    {
        type            timeVaryingMappedFixedValue;
        offset          (0 0 0);
        setAverage      off;
        mapMethod       nearest;
        value           uniform (0 0 0);
    }

    inlet
    {
        type            timeVaryingMappedFixedValue;
        offset          (0 0 0);
        setAverage      off;
        mapMethod       nearest;
        value           uniform (0 0 0);
    }

    outlet
    {
        type            timeVaryingMappedFixedValue;
        offset          (0 0 0);
        setAverage      off;
        mapMethod       nearest;
        value           uniform (0 0 0);
    }

    ".*"
    {
        type            fixedValue;
        value           uniform (0 0 0);
    }
}
```

---

## 8. BOUNDARY CONDITIONS REFERENCE

### Velocity (0/U):
```c
boundaryField
{
    inlet
    {
        // Pulsatile inlet from table file
        type            uniformFixedValue;
        uniformValue
        {
            type    tableFile;
            file    "constant/inlet_waveform.csv";
            outOfBounds repeat;
        }
    }

    outlet
    {
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform (0 0 0);
    }

    wall
    {
        // For STATIC (rigid) wall:
        type            noSlip;

        // For MOVING wall (dynamic mesh):
        type            movingWallVelocity;
        value           uniform (0 0 0);
    }
}
```

### Pressure (0/p):
```c
boundaryField
{
    inlet
    {
        type            zeroGradient;
    }

    outlet
    {
        type            fixedValue;
        value           uniform 0;
    }

    wall
    {
        type            zeroGradient;
    }
}
```

---

## 9. KEY CONFIGURATION FILES

### system/controlDict:
```c
application     foamRun;          // NEW in v11 (was pimpleFoam)
solver          incompressibleFluid;  // NEW in v11

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         4.0;

deltaT          1e-4;

writeControl    adjustableRunTime;
writeInterval   0.01;

adjustTimeStep  yes;
maxCo           0.5;
maxDeltaT       1e-3;

// libs needed for dynamic mesh:
libs            ("libfvMotionSolvers.so");
```

### system/fvSchemes:
```c
ddtSchemes
{
    default         backward;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
    // For dynamic mesh — cell displacement diffusion:
    laplacian(diffusivity,cellDisplacement) Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
```

### system/fvSolution:
```c
solvers
{
    "cellDisplacement.*"    // for dynamic mesh
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
    }

    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.01;
        smoother        GaussSeidel;
    }

    pFinal
    {
        $p;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0.01;
    }

    UFinal
    {
        $U;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors    2;
    nCorrectors         2;
    nNonOrthogonalCorrectors 1;
    pRefCell            0;
    pRefValue           0;
    // For dynamic mesh:
    correctPhi          yes;
    moveMeshOuterCorrectors yes;
}
```

### constant/transportProperties:
```c
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 3.333e-06;   // blood kinematic viscosity
```

### constant/turbulenceProperties:
```c
simulationType  laminar;    // cardiac flow is laminar (Re ~1100-2400)
```

---

## 10. TYPICAL RUN SCRIPT (cardiac CFD)

```bash
#!/bin/bash
set -e

# Source OpenFOAM environment
source /usr/share/openfoam/etc/bashrc   # adjust path as needed

echo "=== Step 1: Background mesh ==="
blockMesh

echo "=== Step 2: Volume mesh from STL ==="
snappyHexMesh -overwrite

echo "=== Step 3: Check mesh quality ==="
checkMesh

echo "=== Step 4: Decompose for parallel ==="
decomposePar

echo "=== Step 5: Solve ==="
mpirun -np 32 foamRun -parallel        # NEW in v11 (was: mpirun -np 32 pimpleFoam -parallel)

echo "=== Step 6: Reconstruct ==="
reconstructPar

echo "=== Step 7: Post-process ==="
foamPostProcess -func wallShearStress   # NEW in v11 (was: postProcess -func wallShearStress)

echo "=== Done ==="
```

---

## 11. COMMON PITFALLS IN v11

1. **`application pimpleFoam` in controlDict** — still works but should be `application foamRun` + `solver incompressibleFluid`
2. **`dynamicFvMesh` keyword** — replaced by `mover { }` block in dynamicMeshDict
3. **`motionSolverLibs`** — still specify in controlDict as `libs ("libfvMotionSolvers.so");`
4. **`postProcess`** — replaced by `foamPostProcess` but old command still works as wrapper
5. **ESI vs Foundation docs** — NEVER use openfoam.com docs for Foundation v11. Use openfoam.org or doc.cfd.direct
6. **snappyHexMeshConfig** — new v11 utility that auto-generates meshing config files. Very useful.
7. **`writeObjects` in controlDict** — may need to be `writeObjects` function object, not a direct keyword

---

## 12. QUICK DIAGNOSTIC COMMANDS

```bash
# Check what OpenFOAM version is installed
foamVersion                           # may not exist in all versions
dpkg -l | grep openfoam               # Ubuntu/Debian
echo $WM_PROJECT_VERSION              # environment variable

# Check if a command exists
which foamRun                         # should return a path
which blockMesh

# Monitor running simulation
ls processor0/ | grep -E '^[0-9]' | sort -g | tail -5    # latest time steps
ps aux | grep foamRun                                       # check if solver is running
tail -f log.foamRun                                         # follow log file

# Quick mesh statistics
checkMesh 2>&1 | grep -E "cells|faces|points|Overall"
```
