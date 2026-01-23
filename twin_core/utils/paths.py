from pathlib import Path

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(base: Path, name: str, timestamp: str) -> Path:
    return ensure_dir(base / f"{name}_{timestamp}")


def dataset_paths(root: Path) -> dict[str, Path]:
    """
    Defines a standard project layout.
    Does NOT create directories automatically.
    """
    return {
        "raw": root / "raw",
        "preprocessed": root / "preprocessed",
        "weights": root / "weights",
        "meshes": root / "meshes",
        "logs": root / "logs",
    }
