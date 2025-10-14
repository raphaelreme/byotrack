import os
import warnings


def parse_bool_from_env(key: str, default: bool):
    """Parse a boolean environment variable"""
    if key not in os.environ:
        return default

    value = os.environ[key].strip().lower()

    if value in ["0", "false", "no", "f", "n"]:
        return False

    if value in ["1", "true", "yes", "t", "y"]:
        return True

    warnings.warn(f"Environment variable {key} expects a boolean value (found {value}) => Defaults to {default}.")
    return default


# Compress segmentation mask in Detections to reduce RAM usage (Experimental feature)
ZSTD_SEG = parse_bool_from_env("ZSTD_SEG", False)

# Cache numba compilation to improve future runtime
NUMBA_CACHE = parse_bool_from_env("NUMBA_CACHE", True)
