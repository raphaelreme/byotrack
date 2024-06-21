import os
import shutil
import subprocess
from typing import Optional, Union


class IcyRunner:  # pylint: disable=too-few-public-methods
    """Runs icy in headless with specified protocol and arguments

    Attributes:
        icy_path (str | os.PathLike): Path to the icy jar (Icy is called with java -jar <icy_jar>)
            If not given, icy is searched in the PATH
        tiemout (Optional[float]): Optional timeout in seconds for Icy protocol.
            Useful for EMHT which may enter an infinite loop.
    """

    cmd = 'java -jar icy.jar -hl -x plugins.adufour.protocols.Protocols protocol="{protocol}" '

    def __init__(self, icy_path: Optional[Union[str, os.PathLike]] = None, timeout: Optional[float] = None) -> None:
        if icy_path is None:
            icy_path = shutil.which("icy")
            if icy_path is None:
                raise RuntimeError("Icy not found, please use `icy_path` to precise where it should be found")

        assert os.path.isfile(os.path.join(os.path.dirname(icy_path), "icy.jar")), f"Icy jar not found at {icy_path}"

        self.icy_path = icy_path
        self.timeout = timeout

    def run(self, protocol: Union[str, os.PathLike], **kwargs) -> int:
        """Runs icy with the given protocol and additional kwargs

        Args:
            protocol (str | os.PathLike): Path to an Icy protocol file
            **kwargs: Additional arguments given as key=value to the cmd line

        Returns:
            int: Return code (Always 0 with Icy... but still let's keep it)
        """

        cmd = self.cmd.format(protocol=protocol) + " ".join((f"{key}={value}" for key, value in kwargs.items()))

        print("Calling Icy with:", cmd)
        return subprocess.run(
            cmd, check=True, cwd=os.path.dirname(self.icy_path), shell=True, timeout=self.timeout
        ).returncode
