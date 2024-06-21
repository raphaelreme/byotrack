import os
import platform
import subprocess
from typing import Union


class FijiRunner:  # pylint: disable=too-few-public-methods
    """Runs fiji in headless with specified protocol and arguments

    Attributes:
        fiji_path (str | os.PathLike): Path to the fiji executable
            The executable can be found inside the installation folder of Fiji.
            Linux: Fiji.app/ImageJ-<os>.exe
            Windows: Fiji.app/ImageJ-<os>.exe
            MacOs: Fiji.app/Contents/MacOs/ImageJ-<os>

    """

    cmd = './{fiji} --ij2 --headless --console --run "{script}" "{args}"'

    def __init__(self, fiji_path: Union[str, os.PathLike]) -> None:
        self.fiji_path = fiji_path

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
        return subprocess.run(cmd, check=True, cwd=os.path.dirname(self.fiji_path), shell=True).returncode
