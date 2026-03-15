from __future__ import annotations

import pathlib
import platform
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os


class FijiRunner:
    """Runs fiji in headless with specified protocol and arguments.

    Attributes:
        fiji_path (pathlib.Path): Path to the fiji executable
            The executable can be found inside the installation folder of Fiji.
            Linux: Fiji.app/ImageJ-<os>
            Windows: Fiji.app/ImageJ-<os>.exe
            MacOs: Fiji.app/Contents/MacOs/ImageJ-<os>
        capture_outputs (bool): Whether to PIPE stderr and stdout into Python
            This will allow you to find the stdout/stderr inside `last_outputs`.
            But outputs are captured, and you do not see them while the scripts run.
            Default: False
        last_outputs (subprocess.CompletedProcess): Outputs of the last subprocess.run

    """

    cmd = './{fiji} --ij2 --headless --console --run "{script}" "{args}"'

    def __init__(self, fiji_path: str | os.PathLike, *, capture_outputs=False) -> None:
        self.fiji_path = pathlib.Path(fiji_path)
        self.capture_outputs = capture_outputs
        self.last_outputs: subprocess.CompletedProcess = subprocess.CompletedProcess("", 0)

        if not self.fiji_path.is_file():
            raise FileNotFoundError("Unable to locate the file `fiji_path`.")

    def run(self, script: str | os.PathLike, **kwargs: Any) -> int:
        """Runs fiji with the given script and additional kwargs.

        Args:
            script (str | os.PathLike): Path to a Fiji script file
            **kwargs: Additional arguments given as key=value to the cmd line

        Returns:
            int: Return code of fiji

        """
        args = ",".join(f"{key}='{value}'" for key, value in kwargs.items())

        cmd = self.cmd.format(fiji=self.fiji_path.name, script=script, args=args)

        if platform.system().lower() == "windows":
            cmd = cmd[2:]  # Strip ./ on windows

        print("Calling Fiji with:", cmd)  # noqa: T201
        self.last_outputs = subprocess.run(  # noqa: S602  # XXX: Check this vulnerability
            cmd,
            check=True,
            cwd=self.fiji_path.parent,
            shell=True,
            capture_output=self.capture_outputs,
        )

        return self.last_outputs.returncode
