from __future__ import annotations

import pathlib
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import os


class IcyRunner:
    """Runs icy in headless with specified protocol and arguments.

    Attributes:
        icy_path (str | os.PathLike): Path to the icy jar (Icy is called with java -jar <icy_jar>)
            If not given, icy is searched in the PATH
        timeout (float | None): Optional timeout in seconds for Icy protocol.
            Useful for EMHT which may enter an infinite loop.
    """

    cmd = 'java -jar icy.jar -hl -x plugins.adufour.protocols.Protocols protocol="{protocol}" '

    def __init__(self, icy_path: str | os.PathLike | None = None, timeout: float | None = None) -> None:
        if icy_path is None:
            icy_path = shutil.which("icy")
            if icy_path is None:
                raise RuntimeError("Icy not found, please use `icy_path` to precise where it should be found")

        self.icy_path = pathlib.Path(icy_path)
        self.timeout = timeout

        if not (self.icy_path.parent / "icy.jar").is_file():
            raise FileNotFoundError("Unable to locate the icy.jar file, please provide a correct `icy_path`")

    def run(self, protocol: str | os.PathLike, **kwargs: Any) -> int:
        """Runs icy with the given protocol and additional kwargs.

        Args:
            protocol (str | os.PathLike): Path to an Icy protocol file
            **kwargs: Additional arguments given as key=value to the cmd line

        Returns:
            int: Return code (Always 0 with Icy... but still let's keep it)
        """
        cmd = self.cmd.format(protocol=protocol) + " ".join((f"{key}={value}" for key, value in kwargs.items()))

        print("Calling Icy with:", cmd)  # noqa: T201
        return subprocess.run(  # noqa: S602
            cmd, check=True, cwd=self.icy_path.parent, shell=True, timeout=self.timeout
        ).returncode
