import trimesh
from pathlib import Path

meshdir = Path(r'c:\Users\dimok\VSCodeProjects\digital_twin_project\test_4d_meshes')
for frame_dir in sorted(meshdir.iterdir()):
    if not frame_dir.is_dir():
        continue
    for struct in ['RV', 'MYO', 'LV']:
        stl = frame_dir / f'{struct}.stl'
        if stl.exists():
            m = trimesh.load(str(stl))
            wt = 'OK' if m.is_watertight else 'HOLES'
            print(f'{frame_dir.name}/{struct}: faces={len(m.faces):6d}  {wt}')