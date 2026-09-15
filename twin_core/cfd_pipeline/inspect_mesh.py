import meshio
m = meshio.read(r"C:/Users/dimok/Desktop/converted_to_vtu_with_paraview.vtu")
for cb in m.cells:
    print(cb.type, len(cb.data))        # want: tetra <N>   (NOT just triangle)
print("cell arrays :", list(m.cell_data.keys()))   # elemTag? -> domains
print("point arrays:", list(m.point_data.keys()))