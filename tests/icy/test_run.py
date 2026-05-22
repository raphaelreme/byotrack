from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from byotrack.icy.run import IcyRunner

if TYPE_CHECKING:
    import pathlib

    from pytest_mock import MockerFixture


## IcyRunner.__init__


def test_icy_runner_valid_path(tmp_path: pathlib.Path):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy)
    assert runner.icy_path == icy


def test_icy_runner_missing_jar_raises(tmp_path: pathlib.Path):
    icy = tmp_path / "icy"
    icy.touch()
    # icy.jar deliberately not created
    with pytest.raises(FileNotFoundError):
        IcyRunner(icy)


def test_icy_runner_no_path_uses_which(tmp_path: pathlib.Path, mocker: MockerFixture):
    icy = tmp_path / "icy.jar"
    icy.touch()

    mocker.patch("shutil.which", return_value=str(icy))
    runner = IcyRunner()

    assert runner.icy_path == icy


def test_icy_runner_no_path_no_which_raises(mocker: MockerFixture):
    mocker.patch("shutil.which", return_value=None)
    with pytest.raises(RuntimeError):
        IcyRunner()


def test_icy_runner_timeout_stored(tmp_path: pathlib.Path):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy, timeout=30.0)
    assert runner.timeout == 30.0


def test_icy_runner_default_timeout_is_none(tmp_path: pathlib.Path):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy)
    assert runner.timeout is None


## IcyRunner.run


def test_icy_runner_run_calls_subprocess(tmp_path: pathlib.Path, mocker: MockerFixture):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "protocol.xml")
    mock_run.assert_called_once()


def test_icy_runner_run_command_contains_flags(tmp_path: pathlib.Path, mocker: MockerFixture):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "protocol.xml")

    cmd = mock_run.call_args[0][0]
    assert "java -jar icy.jar" in cmd
    assert "-hl" in cmd
    assert "-x" in cmd
    assert "plugins.adufour.protocols.Protocols" in cmd


def test_icy_runner_run_kwargs_format(tmp_path: pathlib.Path, mocker: MockerFixture):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "protocol.xml", input_path="/data/in.tif", output_path="/data/out.xml")

    cmd = mock_run.call_args[0][0]
    assert "input_path=/data/in.tif" in cmd
    assert "output_path=/data/out.xml" in cmd
    # Icy format has no quotes around values (unlike Fiji)
    assert "input_path='/data/in.tif'" not in cmd


def test_icy_runner_run_cwd_is_parent(tmp_path: pathlib.Path, mocker: MockerFixture):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "protocol.xml")

    assert mock_run.call_args[1]["cwd"] == tmp_path


def test_icy_runner_timeout_passed_to_subprocess(tmp_path: pathlib.Path, mocker: MockerFixture):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy, timeout=5.0)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "protocol.xml")

    assert mock_run.call_args[1]["timeout"] == 5.0


def test_icy_runner_no_timeout_passed_when_none(tmp_path: pathlib.Path, mocker: MockerFixture):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    runner.run(tmp_path / "protocol.xml")

    # timeout=None is still passed through (subprocess accepts None gracefully)
    assert mock_run.call_args[1]["timeout"] is None


def test_icy_runner_returncode(tmp_path: pathlib.Path, mocker: MockerFixture):
    icy = tmp_path / "icy"
    icy.touch()
    (tmp_path / "icy.jar").touch()
    runner = IcyRunner(icy)

    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0

    result = runner.run(tmp_path / "protocol.xml")
    assert result == 0
