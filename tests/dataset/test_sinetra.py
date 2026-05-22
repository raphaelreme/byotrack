"""Tests for byotrack.dataset.sinetra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from byotrack.dataset.sinetra import load_metadata, load_tracks

if TYPE_CHECKING:
    import pathlib

T, N, D = 5, 3, 2


@pytest.fixture
def sinetra_data():
    return {
        "mu": torch.rand(T, N, D),
        "std": torch.rand(T, N, D),
        "theta": torch.rand(T, N, 1),
        "weight": torch.rand(T, N),
    }


@pytest.fixture
def sinetra_pt_file(tmp_path, sinetra_data):
    path = tmp_path / "video_data.pt"
    torch.save(sinetra_data, path)
    return path


def test_load_metadata_from_file(sinetra_pt_file: pathlib.Path, sinetra_data):
    meta = load_metadata(sinetra_pt_file)
    assert set(meta.keys()) == {"mu", "std", "theta", "weight"}
    assert meta["mu"].shape == (T, N, D)
    assert meta["std"].shape == (T, N, D)
    assert meta["theta"].shape == (T, N, 1)
    assert meta["weight"].shape == (T, N)
    assert torch.allclose(meta["mu"], sinetra_data["mu"])


def test_load_metadata_from_directory(sinetra_pt_file: pathlib.Path, sinetra_data):
    meta = load_metadata(sinetra_pt_file.parent)
    assert set(meta.keys()) == {"mu", "std", "theta", "weight"}
    assert torch.allclose(meta["mu"], sinetra_data["mu"])


def test_load_tracks_count(sinetra_pt_file: pathlib.Path):
    tracks = load_tracks(sinetra_pt_file)
    assert len(tracks) == N


def test_load_tracks_start(sinetra_pt_file: pathlib.Path):
    tracks = load_tracks(sinetra_pt_file)
    for track in tracks:
        assert track.start == 0


def test_load_tracks_positions(sinetra_pt_file: pathlib.Path, sinetra_data):
    tracks = load_tracks(sinetra_pt_file)
    for i, track in enumerate(tracks):
        assert torch.allclose(track.points, sinetra_data["mu"][:, i])


def test_load_tracks_identifiers(sinetra_pt_file: pathlib.Path):
    tracks = load_tracks(sinetra_pt_file)
    for i, track in enumerate(tracks):
        assert track.identifier == i
