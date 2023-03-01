import re
from typing import Iterable, List, Union


def sorted_alphanumeric(data: Iterable[str]):
    """Sorts alphanumeriacally an iterable of strings

    "1" < "2" < "10" < "foo1" < "foo2" < "foo3"

    """

    def alphanum_key(key: str) -> List[Union[str, int]]:
        def convert(text: str) -> Union[str, int]:
            return int(text) if text.isdigit() else text.lower()

        return [convert(text) for text in re.split("([0-9]+)", key)]

    return sorted(data, key=alphanum_key)
