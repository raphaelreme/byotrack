"""All linking experiments for the CLB (See `link.py`)

Submitted to the CLB in Nov. 2024. Cleaned in Jan. 2025.
"""

import argparse
import json
import pathlib
import subprocess
from typing import Any, Dict


def build_params_string(params: Dict[str, Any]) -> str:
    args = []
    for key, value in params.items():
        args.append(f"--{key}={value}")

    return " ".join(args)


def run_clb(data_path: str, default_parameters=False):
    folder = pathlib.Path(__file__).parent
    params: Dict[str, Dict[str, Any]] = json.loads((folder / "hyper_parameters.json").read_text())

    if default_parameters:  # Erase parameters
        params = {dataset: {} for dataset in params}

    for dataset, params_ in params.items():
        for seq in [1, 2]:
            args = build_params_string(params_)
            subprocess.run(
                f"python {folder / 'link.py'} --data_path {data_path} --dataset {dataset} --seq {seq} {args}",
                check=True,
                shell=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KOFT on CTC")

    parser.add_argument(
        "--data_path",
        default="../",
        help="Path to the CTC datasets where each dataset is stored",
    )
    parser.add_argument("--default_parameters", action="store_true", help="Use the default parameters")

    args_ = parser.parse_args()
    print(args_)

    run_clb(
        args_.data_path,
        args_.default_parameters,
    )
