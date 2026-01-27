from pathlib import Path

meshes_root = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\meshes")
found = list(meshes_root.rglob("*.stl")) + list(meshes_root.rglob("*.STL"))
print(f"Found {len(found)} files")
for p in sorted(found):
    print(p)
