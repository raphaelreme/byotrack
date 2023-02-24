import re
from typing import Iterable


def sorted_alphanumeric(data: Iterable[str]):
    """Sorts alphanumeriacally an iterable of strings

    "1" < "2" < "10" < "foo1" < "foo2" < "foo3"

    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)
