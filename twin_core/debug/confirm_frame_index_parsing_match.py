import re
def parse_frame(fname):
    m = re.search(r"(?:frame|f)0*([0-9]+)", fname, flags=re.I)
    return int(m.group(1)) - 1 if m else None

print(parse_frame("patient001_frame01_gt.nii.gz"))   # expect 0
print(parse_frame("patient001_frame12_gt.nii.gz"))   # expect 11
