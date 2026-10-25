"""Kickstart's declaration must reproduce the settings layout it persists.

The keys are the external contract -- a missing one orphans a saved setting silently --
and the golden values pin the defaults a run starts from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ptychodus.api.constants import LengthUnit
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing._kickstart_sections import sections
from ptychodus.model.processing.flatten import flatten_options
from ptychodus.model.processing.generate import populate_parameter_group, restore_options
from ptychodus.model.processing.parameter_store import OptionsSection
from ptychodus.plugins.ptychodus.options import KickstartOptions


def _golden() -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    current = ''

    for line in (Path(__file__).parent / 'data' / 'settings_defaults.ini').read_text().splitlines():
        if line.startswith('['):
            current = line.strip('[]')
            groups[current] = {}
        elif ' = ' in line and current:
            key, value = line.split(' = ', 1)
            groups[current][key] = value

    return groups


def _root(options: KickstartOptions, section: OptionsSection) -> Any:
    return getattr(options, section.root_path) if section.root_path else options


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_group_reproduces_its_persisted_keys(section: OptionsSection) -> None:
    root = _root(KickstartOptions(), section)
    produced = {flat.ini_key for flat in flatten_options(root, section.policy)}
    assert produced == set(_golden()[section.group_name])


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_declared_default_matches_the_persisted_one(section: OptionsSection) -> None:
    root = _root(KickstartOptions(), section)
    expected = _golden()[section.group_name]

    for flat in flatten_options(root, section.policy):
        assert flat.spelling.render(flat.default) == expected[flat.ini_key], flat.ini_key


@pytest.mark.parametrize('section', sections(), ids=lambda s: s.group_name)
def test_every_key_carries_display_text(section: OptionsSection) -> None:
    root = _root(KickstartOptions(), section)
    unlabelled = [f.ini_key for f in flatten_options(root, section.policy) if not f.label]
    assert not unlabelled


def test_the_defocus_reads_in_micrometers_until_it_is_set() -> None:
    (section,) = sections()
    defocus = next(
        f
        for f in flatten_options(KickstartOptions(), section.policy)
        if f.ini_key == 'DefocusDistanceInMeters'
    )
    assert defocus.display_unit is LengthUnit.MICROMETER


class TestRoundTrip:
    def _build(self) -> tuple[SettingsRegistry, KickstartOptions]:
        registry = SettingsRegistry()
        options = KickstartOptions()

        for section in sections():
            populate_parameter_group(
                registry.create_group(section.group_name),
                _root(options, section),
                section.policy,
            )

        return registry, options

    def test_an_edit_reaches_the_restored_branch(self) -> None:
        registry, options = self._build()
        (section,) = sections()
        parameters = registry['KickstartOptions'].parameters()
        parameters['DefocusDistanceInMeters'].set_value(-2.5e-4)
        restored = restore_options(options, parameters, section.policy)
        assert restored.defocus_distance_m == -2.5e-4

    def test_a_bound_clamps_a_value_loaded_from_file(self) -> None:
        registry, _ = self._build()
        parameter = registry['KickstartOptions'].parameters()['ParallaxMinDefocusDof']
        parameter.set_value_from_string('0.1')
        assert parameter.get_value() == 0.5
