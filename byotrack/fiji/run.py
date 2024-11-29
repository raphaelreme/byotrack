import os
import platform
import subprocess
from typing import Union


class FijiRunner:  # pylint: disable=too-few-public-methods
    """Runs fiji in headless with specified protocol and arguments

    Attributes:
        fiji_path (str | os.PathLike): Path to the fiji executable
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

    def __init__(self, fiji_path: Union[str, os.PathLike], capture_outputs=False) -> None:
        self.fiji_path = fiji_path
        self.capture_outputs = capture_outputs
        self.last_outputs: subprocess.CompletedProcess = subprocess.CompletedProcess("", 0)

        assert os.path.isfile(fiji_path), "Unable to found the given path"

    def run(self, script: Union[str, os.PathLike], **kwargs) -> int:
        """Runs fiji with the given script and additional kwargs

        Args:
            script (str | os.PathLike): Path to a Fiji script file
            **kwargs: Additional arguments given as key=value to the cmd line

        Returns:
            int: Return code of fiji

        """
        args = ",".join(f"{key}='{value}'" for key, value in kwargs.items())

        cmd = self.cmd.format(fiji=os.path.basename(self.fiji_path), script=script, args=args)

        if platform.system().lower() == "windows":
            cmd = cmd[2:]  # Strip ./ on windows

        print("Calling Fiji with:", cmd)
        self.last_outputs = subprocess.run(
            cmd, check=True, cwd=os.path.dirname(self.fiji_path), shell=True, capture_output=self.capture_outputs
        )

        return self.last_outputs.returncode
