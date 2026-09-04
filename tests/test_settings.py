"""Unit tests for SettingsRegistry settings-file loading.

`open_settings` drives every parameter from a registered name, so a key that no
parameter claims is simply never visited. Renaming a parameter therefore orphans the
old key in every settings file still using it and leaves the parameter silently at its
default -- exactly the failure that sent a whole reconstruction off the detector. These
tests lock the warning that makes an orphaned key visible.
"""

from pathlib import Path

import pytest

from ptychodus.api.settings import SettingsRegistry

_LOGGER_NAME = 'ptychodus.api.settings'


def _write_settings(tmp_path: Path, text: str) -> Path:
    file_path = tmp_path / 'settings.ini'
    file_path.write_text(text)
    return file_path


def test_open_settings_warns_about_unrecognized_key(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A key matching no registered parameter is reported and leaves the default intact."""
    registry = SettingsRegistry()
    group = registry.create_group('Demo')
    kept = group.create_integer_parameter('BeamCenterXInPixels', 32)

    file_path = _write_settings(
        tmp_path, '[Demo]\nCropCenterXInPixels = 540\nBeamCenterXInPixels = 41\n'
    )

    with caplog.at_level('WARNING', logger=_LOGGER_NAME):
        registry.open_settings(file_path)

    assert kept.get_value() == 41
    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert 'Demo' in message
    assert 'cropcenterxinpixels' in message.lower()


def test_open_settings_is_quiet_for_a_fully_recognized_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A valid file must not warn -- a warning that always fires stops being read."""
    registry = SettingsRegistry()
    group = registry.create_group('Demo')
    parameter = group.create_integer_parameter('BeamCenterXInPixels', 32)

    file_path = _write_settings(tmp_path, '[Demo]\nBeamCenterXInPixels = 540\n')

    with caplog.at_level('WARNING', logger=_LOGGER_NAME):
        registry.open_settings(file_path)

    assert parameter.get_value() == 540
    assert not caplog.records


def test_open_settings_matches_keys_case_insensitively(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """configparser folds option case, so a differently-cased key must still bind."""
    registry = SettingsRegistry()
    group = registry.create_group('Demo')
    parameter = group.create_integer_parameter('BeamCenterXInPixels', 32)

    file_path = _write_settings(tmp_path, '[Demo]\nbeamcenterxinpixels = 540\n')

    with caplog.at_level('WARNING', logger=_LOGGER_NAME):
        registry.open_settings(file_path)

    assert parameter.get_value() == 540
    assert not caplog.records


def test_open_settings_ignores_default_section_keys(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """configparser injects [DEFAULT] keys into every section; they are not orphans."""
    registry = SettingsRegistry()
    group = registry.create_group('Demo')
    parameter = group.create_integer_parameter('BeamCenterXInPixels', 32)

    file_path = _write_settings(
        tmp_path, '[DEFAULT]\nSharedKey = 7\n\n[Demo]\nBeamCenterXInPixels = 540\n'
    )

    with caplog.at_level('WARNING', logger=_LOGGER_NAME):
        registry.open_settings(file_path)

    assert parameter.get_value() == 540
    assert not caplog.records


def test_open_settings_skips_sections_with_no_registered_group(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An entire unknown section is not walked, so it produces no per-key noise."""
    registry = SettingsRegistry()
    registry.create_group('Demo')

    file_path = _write_settings(tmp_path, '[NotAGroup]\nWhatever = 1\n')

    with caplog.at_level('WARNING', logger=_LOGGER_NAME):
        registry.open_settings(file_path)

    assert not caplog.records
