"""Unit tests for the scan patterns offered as plugins.

The golden settings fixture catches a key that *changed*. It cannot catch one a
generator never contributed, because a generator that quietly declares nothing
simply adds nothing to the section. These tests cover that other direction: the
roster is what it was before the patterns became plugins, every declared key
lands in the group a scan is configured by, and every field has something to
write beside it. The Lissajous figure's angles are pinned to the degrees their
suffix claims.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy
import pytest

from ptychodus.api.plugins import Plugin, PluginRegistry
from ptychodus.api.simulate.probe_positions import (
    ProbePositionsGenerator,
    generate_lissajous_probe_positions,
)
from ptychodus.model.settings import SettingsRegistry
from ptychodus.model.declared import declare_in, describe, restore


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


def test_every_pattern_is_registered() -> None:
    """The names are persisted, documented and accepted by the workflow API."""
    assert {plugin.simple_name for plugin in _generators()} == _EXPECTED_NAMES


@pytest.mark.parametrize('plugin', _generators(), ids=lambda p: p.simple_name)
class TestEachGenerator:
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


def _lissajous(
    *, step_x_deg: float, step_y_deg: float, shift_deg: float, num_points: int = 8
) -> tuple[numpy.ndarray, numpy.ndarray]:
    positions = list(
        generate_lissajous_probe_positions(
            num_points,
            amplitude_x_m=2.0,
            amplitude_y_m=3.0,
            angular_step_x_deg=step_x_deg,
            angular_step_y_deg=step_y_deg,
            angular_shift_deg=shift_deg,
        )
    )
    return numpy.array([p.x_m for p in positions]), numpy.array([p.y_m for p in positions])


def test_a_lissajous_shift_of_ninety_degrees_is_a_quarter_cycle() -> None:
    """The shift was once applied in radians while labelled in turns."""
    x_m, y_m = _lissajous(step_x_deg=0.0, step_y_deg=0.0, shift_deg=90.0)

    numpy.testing.assert_allclose(x_m, 2.0)
    numpy.testing.assert_allclose(y_m, 0.0, atol=1e-15)


def test_a_full_cycle_lissajous_step_revisits_the_first_point() -> None:
    x_m, y_m = _lissajous(step_x_deg=360.0, step_y_deg=-720.0, shift_deg=30.0)

    numpy.testing.assert_allclose(x_m, 1.0)
    numpy.testing.assert_allclose(y_m, 0.0, atol=1e-12)


def test_lissajous_steps_are_in_degrees() -> None:
    _, y_m = _lissajous(step_x_deg=0.0, step_y_deg=90.0, shift_deg=0.0, num_points=4)

    numpy.testing.assert_allclose(y_m, [0.0, 3.0, 0.0, -3.0], atol=1e-12)
