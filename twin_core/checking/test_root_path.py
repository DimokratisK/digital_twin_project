from pathlib import Path

root = Path(r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\train")
print([p.name for p in root.iterdir()][:10])



