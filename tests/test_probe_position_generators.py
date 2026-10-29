"""Unit tests for the scan patterns offered as plugins.

The golden settings fixture catches a key that *changed*. It cannot catch one a
generator never contributed, because a generator that quietly declares nothing
simply adds nothing to the section. These tests cover that other direction: the
roster is what it was before the patterns became plugins, every declared key
lands in the group a scan is configured by, and every field has something to
write beside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from ptychodus.api.plugins import Plugin, PluginRegistry
from ptychodus.api.simulate.probe_positions import ProbePositionsGenerator
from ptychodus.model.settings import SettingsRegistry
from ptychodus.model.declared import declare_in, describe, restore

_GOLDEN_PATH = Path(__file__).parent / 'data' / 'settings_defaults.ini'

_EXPECTED_NAMES = {
    'concentric',
    'hexagonal_raster',
    'hexagonal_snake',
    'lissajous',
    'rectangular_raster',
    'rectangular_snake',
    'spiral',
    'square_raster',
    'square_snake',
    'triangular_raster',
    'triangular_snake',
}


def _generators() -> Sequence[Plugin[ProbePositionsGenerator]]:
    return [*PluginRegistry.load_plugins().probe_position_generators]


def _probe_positions_keys() -> set[str]:
    """Every key the golden fixture holds in the section a scan is configured by."""
    keys: set[str] = set()
    inside = False

    for line in _GOLDEN_PATH.read_text().splitlines():
        if line.startswith('['):
            inside = line == '[ProbePositions]'
        elif inside and ' = ' in line:
            keys.add(line.split(' = ', 1)[0])

    return keys


def test_every_pattern_is_registered() -> None:
    """The names are persisted, documented and accepted by the workflow API."""
    assert {plugin.simple_name for plugin in _generators()} == _EXPECTED_NAMES


@pytest.mark.parametrize('plugin', _generators(), ids=lambda p: p.simple_name)
class TestEachGenerator:
    def test_its_keys_are_in_the_settings_section(
        self, plugin: Plugin[ProbePositionsGenerator]
    ) -> None:
        declared = {d.key for d in describe(plugin.strategy.get_default_options(), expand_num=True)}
        assert declared <= _probe_positions_keys()

    def test_every_field_has_something_to_write_beside_it(
        self, plugin: Plugin[ProbePositionsGenerator]
    ) -> None:
        for description in describe(plugin.strategy.get_default_options()):
            assert description.label, f'{plugin.simple_name}.{description.name} states no title'

    def test_it_generates_from_the_values_the_settings_hold(
        self, plugin: Plugin[ProbePositionsGenerator]
    ) -> None:
        """The whole round trip: declare, read back, hand over, generate."""
        generator = plugin.strategy
        defaults = generator.get_default_options()
        described = describe(defaults, expand_num=True)
        group = SettingsRegistry().create_group('ProbePositions')
        parameters = declare_in(group, described)

        positions = generator.generate(restore(defaults, described, parameters))

        assert len(positions) > 0
        assert [p.index for p in positions] == list(range(len(positions)))

    def test_a_value_written_to_the_settings_reaches_the_pattern(
        self, plugin: Plugin[ProbePositionsGenerator]
    ) -> None:
        """A count held in the settings must change how many points come back."""
        generator = plugin.strategy
        defaults = generator.get_default_options()
        described = describe(defaults, expand_num=True)
        group = SettingsRegistry().create_group('ProbePositions')
        parameters = declare_in(group, described)

        counts = [d for d in described if d.key.startswith('NumberOf')]
        assert counts, f'{plugin.simple_name} declares no count'

        before = len(generator.generate(restore(defaults, described, parameters)))
        parameters[counts[0].key].set_value(parameters[counts[0].key].get_value() + 1)
        after = len(generator.generate(restore(defaults, described, parameters)))

        assert after != before


def test_the_equilateral_variants_state_no_y_step() -> None:
    """Their y step follows from their x step, so there is nothing to set."""
    for plugin in _generators():
        if not plugin.simple_name.startswith(('square', 'hexagonal')):
            continue

        keys = {d.key for d in describe(plugin.strategy.get_default_options(), expand_num=True)}
        assert 'StepSizeYInMeters' not in keys


def test_spiral_and_lissajous_agree_on_the_count_they_share() -> None:
    """They write one key, so the first registered would otherwise set its default."""
    registry = PluginRegistry.load_plugins().probe_position_generators
    defaults = {}

    for name in ('spiral', 'lissajous'):
        for description in describe(
            registry.get_strategy_by_name(name).get_default_options(), expand_num=True
        ):
            if description.key == 'NumberOfPoints':
                defaults[name] = description.default

    assert defaults == {'spiral': 100, 'lissajous': 100}
