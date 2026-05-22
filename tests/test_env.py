"""Tests for byotrack._env."""

from __future__ import annotations

import pytest

from byotrack._env import parse_bool_from_env

_KEY = "BYOTRACK_TEST_BOOL_VAR"


def test_absent_key_default_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_KEY, raising=False)
    assert parse_bool_from_env(_KEY, default=False) is False


def test_absent_key_default_true(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_KEY, raising=False)
    assert parse_bool_from_env(_KEY, default=True) is True


@pytest.mark.parametrize("value", ["0", "false", "no", "f", "n"])
def test_false_values(monkeypatch: pytest.MonkeyPatch, value):
    monkeypatch.setenv(_KEY, value)
    assert parse_bool_from_env(_KEY, default=True) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "t", "y"])
def test_true_values(monkeypatch: pytest.MonkeyPatch, value):
    monkeypatch.setenv(_KEY, value)
    assert parse_bool_from_env(_KEY, default=False) is True


@pytest.mark.parametrize("value", ["FALSE", "TRUE", "Yes", "No"])
def test_case_insensitive(monkeypatch: pytest.MonkeyPatch, value):
    monkeypatch.setenv(_KEY, value)
    expected = value.lower() in ("true", "yes", "1")
    assert parse_bool_from_env(_KEY, default=not expected) is expected


@pytest.mark.parametrize("value", ["  true  ", "  0  "])
def test_strips_whitespace(monkeypatch: pytest.MonkeyPatch, value):
    monkeypatch.setenv(_KEY, value)
    expected = value.strip() in ("true", "yes", "1")
    assert parse_bool_from_env(_KEY, default=not expected) is expected


@pytest.mark.parametrize("default", [True, False])
def test_invalid_value_warns_and_returns_default(monkeypatch: pytest.MonkeyPatch, default):
    monkeypatch.setenv(_KEY, "maybe")

    with pytest.warns(UserWarning, match="expects a boolean value"):
        result = parse_bool_from_env(_KEY, default=default)

    assert result == default
