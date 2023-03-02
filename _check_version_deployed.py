"""Check that the github tag and package version matches before deployement"""

import os
import pathlib
import re


def parse_package_version(package) -> str:
    """Parse package version.

    It does not import the package, but rather parse the package.
    It expects the version to be stored in __init__.py as __version__ = xxx
    """
    version_regexp = r'.*?__version__ = "(.*?)".*?'

    path = pathlib.Path(package) / "__init__.py"
    text = path.read_text(encoding="utf-8")
    match = re.match(version_regexp, text, re.S)

    if match is None:
        raise RuntimeError(f"Unable to parse version in {package}/__init__.py")

    version = match.group(1)
    return f"v{version}"


if __name__ == "__main__":
    package_version = parse_package_version("byotrack")
    tag_version = os.environ["GITHUB_REF_NAME"]

    assert tag_version == package_version, f"{package_version} != {tag_version}"
