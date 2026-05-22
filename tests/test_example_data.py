"""Tests for byotrack.example_data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

import byotrack
from byotrack import example_data

if TYPE_CHECKING:
    import pathlib


def _write_minimal_avi(dest) -> None:
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"MJPG"), 10, (12, 10))  # type: ignore[attr-defined]
    for _ in range(3):
        writer.write(np.random.randint(0, 255, (10, 12, 3), dtype=np.uint8))
    writer.release()


def test_download_triggered_when_file_absent(tmp_path: pathlib.Path, mocker):
    mocker.patch.object(example_data, "data_dir", tmp_path)

    def fake_retrieve(_, dest):
        _write_minimal_avi(dest)

    mock_retrieve = mocker.patch("urllib.request.urlretrieve", side_effect=fake_retrieve)

    video = example_data.hydra_neurons()

    mock_retrieve.assert_called_once()
    assert isinstance(video, byotrack.Video)


def test_no_download_when_file_exists(tmp_path, mocker):
    mocker.patch.object(example_data, "data_dir", tmp_path)
    _write_minimal_avi(tmp_path / "tdt_contrxn-1.avi")

    mock_retrieve = mocker.patch("urllib.request.urlretrieve")

    video = example_data.hydra_neurons()

    mock_retrieve.assert_not_called()
    assert isinstance(video, byotrack.Video)


def test_runtime_error_on_corrupted_file(tmp_path, mocker):
    mocker.patch.object(example_data, "data_dir", tmp_path)

    def fake_retrieve(_, dest):
        dest.write_bytes(b"not a valid avi")

    mocker.patch("urllib.request.urlretrieve", side_effect=fake_retrieve)

    with pytest.raises(RuntimeError, match="corrupted"):
        example_data.hydra_neurons()
