"""Tests for byotrack.utils."""

from __future__ import annotations

import pytest

from byotrack.utils import sorted_alphanumeric


def test_numeric_strings():
    assert sorted_alphanumeric(["10", "2", "1"]) == ["1", "2", "10"]


def test_alphanumeric_strings():
    assert sorted_alphanumeric(["foo2", "foo10", "foo1"]) == ["foo1", "foo2", "foo10"]


def test_pure_alpha_strings():
    assert sorted_alphanumeric(["b", "a", "c"]) == ["a", "b", "c"]


def test_already_sorted():
    data = ["a1", "a2", "a3"]
    assert sorted_alphanumeric(data) == data


def test_empty():
    assert sorted_alphanumeric([]) == []


def test_mixed_alpha_and_numeric():
    assert sorted_alphanumeric(["1", "2", "10", "foo1", "foo2", "foo3"]) == [
        "1",
        "2",
        "10",
        "foo1",
        "foo2",
        "foo3",
    ]


def test_path_like_objects(tmp_path):
    paths = [tmp_path / "frame_10.tif", tmp_path / "frame_2.tif", tmp_path / "frame_1.tif"]
    result = sorted_alphanumeric(paths)
    assert result == [tmp_path / "frame_1.tif", tmp_path / "frame_2.tif", tmp_path / "frame_10.tif"]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (["z9", "z10", "z2"], ["z2", "z9", "z10"]),
        (["img_100", "img_20", "img_3"], ["img_3", "img_20", "img_100"]),
    ],
)
def test_parametrized_cases(data, expected):
    assert sorted_alphanumeric(data) == expected
