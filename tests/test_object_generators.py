"""Object generators as plugins, and the model-side inputs they are handed.

A generator states what it accepts and turns those values into an object; it knows
nothing about where the values, the random draws or the probe come from. These pin the
contracts that make that work: what a generator returns is raw and the builder
conditions it once, the keys generators declare do not collide, and only the generators
that read the probe are handed one.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from unittest.mock import MagicMock

import numpy
import numpy.testing
import pytest

from ptychodus.api.accumulate import DiffractionMoments
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.object import ObjectGeometry, ObjectGeometryProvider
from ptychodus.api.plugins import PluginChooser, PluginRegistry
from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.simulate.object import ObjectGenerator, pad_object
from ptychodus.model.declared import ParameterDescription, describe
from ptychodus.model.product.object.builder import FromFileObjectBuilder
from ptychodus.model.product.object.builder_factory import ObjectBuilderFactory
from ptychodus.model.product.object.plugin import PluginObjectBuilder
from ptychodus.model.product.object.settings import ObjectSettings

_EXTENT_PX = 32
_PIXEL_SIZE_M = 1.0e-7


class _Provider(ObjectGeometryProvider):
    """A bound geometry without diffraction data."""

    def __init__(self, *, mean_amplitude_transmission: float = 1.0) -> None:
        self._mean_amplitude_transmission = mean_amplitude_transmission

    @property
    def photon_wavelength_m(self) -> float:
        return 1.0e-10

    @property
    def mean_amplitude_transmission(self) -> float:
        return self._mean_amplitude_transmission

    @property
    def object_plane_propagation_distance_m(self) -> float:
        return 1.0

    @property
    def far_field(self) -> bool:
        return True

    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return ()

    def get_object_geometry(self) -> ObjectGeometry:
        return ObjectGeometry(
            width_px=_EXTENT_PX,
            height_px=_EXTENT_PX,
            pixel_width_m=_PIXEL_SIZE_M,
            pixel_height_m=_PIXEL_SIZE_M,
            center_x_m=0.0,
            center_y_m=0.0,
        )

    def get_assembled_data(self) -> AssembledDiffractionData | None:
        return None

    def get_diffraction_moments(self) -> DiffractionMoments | None:
        return None


@pytest.fixture(scope='module')
def registry() -> PluginRegistry:
    return PluginRegistry.load_plugins()


def _described(generator: ObjectGenerator) -> list[ParameterDescription]:
    return list(describe(generator.get_default_options(), expand_num=True))


def _make_factory(
    registry: PluginRegistry, settings: ObjectSettings | None = None
) -> ObjectBuilderFactory:
    return ObjectBuilderFactory(
        numpy.random.default_rng(42),
        ObjectSettings(SettingsRegistry()) if settings is None else settings,
        registry.object_generators,
        PluginChooser(),
        PluginChooser(),
    )


def _make_builder(registry: PluginRegistry, name: str) -> PluginObjectBuilder:
    builder = _make_factory(registry).create(name, has_diffraction_data=True)
    assert isinstance(builder, PluginObjectBuilder)
    return builder


# -- generators ----------------------------------------------------------------------


def test_every_generator_returns_one_raw_layer_on_the_grid(registry: PluginRegistry) -> None:
    provider = _Provider()

    for plugin in registry.object_generators:
        generator = plugin.strategy
        options = generator.get_default_options()
        rng = numpy.random.default_rng(0)

        if generator.reads_diffraction_data:
            with pytest.raises(GeometryNotDefinedError):
                generator.generate(options, provider, rng=rng, probe=None)

            continue

        object_ = generator.generate(options, provider, rng=rng, probe=None)

        assert object_.num_layers == 1, plugin.simple_name
        assert (object_.height_px, object_.width_px) == (_EXTENT_PX, _EXTENT_PX), plugin.simple_name


def test_the_builder_pads_and_rescales_exactly_once(registry: PluginRegistry) -> None:
    """Padding and the transmission scale are the builder's concern, not the generator's."""
    transmission = 0.6
    provider = _Provider(mean_amplitude_transmission=transmission)
    generator = registry.object_generators.get_strategy_by_name('random')
    builder = _make_builder(registry, 'random')
    builder.parameters()['amplitude_deviation'].set_value(0.1)
    builder.parameters()['phase_deviation_tr'].set_value(0.1)
    builder.extra_padding_x.set_value(3)
    builder.extra_padding_y.set_value(2)
    options = generator.get_default_options().model_copy(
        update={'amplitude_deviation': 0.1, 'phase_deviation_tr': 0.1}
    )

    raw = generator.generate(options, provider, rng=numpy.random.default_rng(42), probe=None)
    built = builder.build(provider, [])

    expected = pad_object(raw, 3, 2).get_array() * transmission
    numpy.testing.assert_allclose(built.get_array(), expected, atol=1e-12)


def test_same_draws_give_the_same_texture(registry: PluginRegistry) -> None:
    provider = _Provider()

    for plugin in registry.object_generators:
        generator = plugin.strategy

        if generator.reads_diffraction_data:
            continue

        options = generator.get_default_options()
        first = generator.generate(options, provider, rng=numpy.random.default_rng(3), probe=None)
        again = generator.generate(options, provider, rng=numpy.random.default_rng(3), probe=None)

        numpy.testing.assert_array_equal(first.get_array(), again.get_array())


def test_generators_hold_no_state(registry: PluginRegistry) -> None:
    for plugin in registry.object_generators:
        assert vars(plugin.strategy) == {}, plugin.simple_name


def test_keys_shared_by_several_generators_agree(registry: PluginRegistry) -> None:
    """A key shared across generators holds one value, so its declarations must agree.

    No object key is shared today; this guards the day a field name is reused.
    """
    by_key: dict[str, list[tuple[str, ParameterDescription]]] = defaultdict(list)

    for plugin in registry.object_generators:
        for description in _described(plugin.strategy):
            by_key[description.key].append((plugin.simple_name, description))

    for key, entries in by_key.items():
        signatures = {
            (repr(d.annotation), repr(d.default), d.minimum, d.maximum) for _, d in entries
        }
        assert len(signatures) == 1, f'{key} disagrees across {[name for name, _ in entries]}'


def test_no_declaration_collides_with_a_hand_written_setting(registry: PluginRegistry) -> None:
    """A declared key equal to a hand-written one would silently share its value."""
    settings_registry = SettingsRegistry()
    settings = ObjectSettings(settings_registry)
    hand_written_keys = set(settings_registry['Object'].parameters())
    # The from-file builder holds every parameter the base class adds, and its own.
    base_names = set(FromFileObjectBuilder(settings, MagicMock()).parameters())

    for plugin in registry.object_generators:
        for description in _described(plugin.strategy):
            assert description.key not in hand_written_keys, (plugin.simple_name, description.key)
            assert description.name not in base_names, (plugin.simple_name, description.name)


# -- builders ------------------------------------------------------------------------


def test_every_generator_builds_into_a_builder_named_where_it_registered(
    registry: PluginRegistry,
) -> None:
    factory = _make_factory(registry)

    for name in factory:
        assert factory.create(name, has_diffraction_data=True).get_name() == name


def test_the_default_builder_is_named(registry: PluginRegistry) -> None:
    assert _make_factory(registry).create_default().get_name() == 'random'


def test_exactly_the_phase_retrievals_read_the_data_and_the_probe(
    registry: PluginRegistry,
) -> None:
    reading_data = {
        p.simple_name for p in registry.object_generators if p.strategy.reads_diffraction_data
    }
    reading_probe = {p.simple_name for p in registry.object_generators if p.strategy.reads_probe}

    assert reading_data == {'paganin', 'stxm_dpc'}
    assert reading_probe == {'paganin', 'stxm_dpc'}


def test_a_builder_that_reads_data_needs_data(registry: PluginRegistry) -> None:
    with pytest.raises(RuntimeError, match='requires an associated diffraction dataset'):
        _make_factory(registry).create('stxm_dpc', has_diffraction_data=False)


def test_the_probe_is_not_fetched_for_a_generator_that_does_not_read_it(
    registry: PluginRegistry,
) -> None:
    for plugin in registry.object_generators:
        if plugin.strategy.reads_probe:
            continue

        builder = _make_builder(registry, plugin.simple_name)
        supplier = MagicMock()
        builder.bind_probe_supplier(supplier)
        builder.build(_Provider(), [])

        supplier.assert_not_called()


def test_a_copy_carries_every_declared_value(registry: PluginRegistry) -> None:
    builder = _make_builder(registry, 'siemens_star')
    builder.parameters()['num_spokes'].set_value(12)
    builder.parameters()['spoke_phase_tr'].set_value(0.125)
    builder.extra_padding_x.set_value(4)

    duplicate = builder.copy()

    assert isinstance(duplicate, PluginObjectBuilder)
    assert duplicate.describe_parameters() == builder.describe_parameters()
    assert {key: p.get_value() for key, p in duplicate.parameters().items()} == {
        key: p.get_value() for key, p in builder.parameters().items()
    }


def test_a_builder_name_synced_to_settings_is_found_again(registry: PluginRegistry) -> None:
    """The Gaussian random field once named itself other than its registered name, so
    syncing it to settings wrote a name the factory could not find."""
    settings = ObjectSettings(SettingsRegistry())
    factory = _make_factory(registry, settings)

    for name in factory:
        factory.create(name, has_diffraction_data=True).sync_to_settings()
        assert factory.create_from_settings(has_diffraction_data=True).get_name() == name
