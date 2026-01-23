from datetime import datetime


def run_timestamp() -> str:
    """Return a timestamp string for folder naming: DDMMYY_HHMM."""
    return datetime.now().strftime("%d%m%y_%H%M")


def get_timestamp() -> str:
    """Alias for run_timestamp."""
    return run_timestamp()
