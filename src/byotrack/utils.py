from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable

T = TypeVar("T")


def sorted_alphanumeric(data: Iterable[T]) -> list[T]:
    """Sorts alphanumeriacally an iterable of strings-like objects.

    "1" < "2" < "10" < "foo1" < "foo2" < "foo3"
    """

    def convert(text: str) -> int | str:
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return tuple(convert(c) for c in re.split("([0-9]+)", str(key)))

    return sorted(data, key=alphanum_key)
