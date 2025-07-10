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


NUMBA_CACHE = parse_bool_from_env("NUMBA_CACHE", True)
