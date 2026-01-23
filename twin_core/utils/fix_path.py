path = r"C:\Users\dimok\Downloads\PhD\Digital Twin\Data\Human Heart Project - Automated Cardiac Diagnosis Challenge (ACDC)\Resources\database"

def fix_path(path_str: str) -> str:
    """Convert all backslashes to forward slashes in a given path string."""
    return path_str.replace("\\", "/")
                            
path = fix_path(path)
print(path)