from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from byotrack.fiji.run import FijiRunner

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture


## FijiRunner.__init__


def test_fiji_runner_valid_path(tmp_path: pathlib.Path):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji)
    assert runner.fiji_path == fiji


def test_fiji_runner_invalid_path_raises(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        FijiRunner(tmp_path / "nonexistent")


def test_fiji_runner_capture_outputs_default(tmp_path: pathlib.Path):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji)
    assert runner.capture_outputs is False


def test_fiji_runner_capture_outputs_set(tmp_path: pathlib.Path):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji, capture_outputs=True)
    assert runner.capture_outputs is True


## FijiRunner.run


def test_fiji_runner_run_calls_subprocess(tmp_path: pathlib.Path, mocker: MockerFixture):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "script.py")
    mock_run.assert_called_once()


def test_fiji_runner_run_command_contains_flags(tmp_path: pathlib.Path, mocker: MockerFixture):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "script.py")

    cmd = mock_run.call_args[0][0]
    assert "--ij2" in cmd
    assert "--headless" in cmd
    assert "--console" in cmd
    assert "--run" in cmd


def test_fiji_runner_run_kwargs_format(tmp_path: pathlib.Path, mocker: MockerFixture):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "script.py", input_path="/data/in.tif", output_path="/data/out.xml")

    cmd = mock_run.call_args[0][0]
    assert "input_path='/data/in.tif'" in cmd
    assert "output_path='/data/out.xml'" in cmd


def test_fiji_runner_run_cwd_is_parent(tmp_path: pathlib.Path, mocker: MockerFixture):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "script.py")

    assert mock_run.call_args[1]["cwd"] == tmp_path


def test_fiji_runner_run_capture_output_forwarded(tmp_path: pathlib.Path, mocker: MockerFixture):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji, capture_outputs=True)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "script.py")

    assert mock_run.call_args[1]["capture_output"] is True


def test_fiji_runner_run_returns_returncode(tmp_path: pathlib.Path, mocker: MockerFixture):
    fiji = tmp_path / "ImageJ-linux64"
    fiji.touch()
    runner = FijiRunner(fiji)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 42

    result = runner.run(tmp_path / "script.py")
    assert result == 42
