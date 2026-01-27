import json
m = json.load(open(r"c:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database\preprocessed\mask_manifest.json"))
total_saved = sum(v["saved_masks"] for v in m.values())
total_real = sum(v["real_masks"] for v in m.values())
print("total saved masks:", total_saved)
print("total real masks:", total_real)
print("fraction labeled:", total_real/total_saved)
