"""Probe generators as plugins, and the model-side sources of their option values.

A generator states what it accepts and turns those values into a probe; it knows nothing
about where the values come from. These pin the contracts that make that work: the keys
generators share agree, a declared size names a quantity the data can measure, and a
preset type fits exactly the generator whose fields it names.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy
import pytest
from pydantic import BaseModel, Field

from ptychodus.api.accumulate import DiffractionMoments
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import GeometryNotDefinedError, PixelGeometry, ZernikeMode
from ptychodus.api.plugins import PluginCatalog, PluginList, PluginRegistry
from ptychodus.api.probe import Probe, ProbeGeometry, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    KirkpatrickBaezMirrorPreset,
    ProbeGenerator,
    generate_disk_probe,
    generate_zernike_probe,
)
from ptychodus.model.observer import Observable, Observer
from ptychodus.model.plugins import PluginChooser
from ptychodus.model.settings import SettingsRegistry
from ptychodus.model.declared import ParameterDescription, describe, fields_of
from ptychodus.model.product.probe.builder_factory import ProbeBuilderFactory
from ptychodus.model.product.probe.plugin import PluginProbeBuilder
from ptychodus.model.product.probe.settings import ProbeSettings

_PROBE_EXTENT_PX = 64


class _Provider(ProbeGeometryProvider):
    """A bound geometry without diffraction data."""

    def __init__(self, *, pixel_size_m: float = 1.0e-7) -> None:
        self._pixel_size_m = pixel_size_m

    def get_assembled_data(self) -> AssembledDiffractionData | None:
        return None

    def get_diffraction_moments(self) -> DiffractionMoments | None:
        return None

    @property
    def detector_distance_m(self) -> float:
        return 1.0

    @property
    def far_field(self) -> bool:
        return True

    @property
    def focus_object_distance_m(self) -> float:
        return 0.0

    @property
    def probe_photon_count(self) -> float:
        return 1.0e6

    @property
    def photon_wavelength_m(self) -> float:
        return 1.0e-10

    @property
    def probe_power_W(self) -> float:  # noqa: N802
        return 1.0

    @property
    def num_scan_points(self) -> int:
        return 7

    def get_detector_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(width_m=1.0e-7, height_m=1.0e-7)

    def get_probe_geometry(self) -> ProbeGeometry:
        return ProbeGeometry(
            width_px=_PROBE_EXTENT_PX,
            height_px=_PROBE_EXTENT_PX,
            pixel_width_m=self._pixel_size_m,
            pixel_height_m=self._pixel_size_m,
        )


class _NotificationCounter(Observer):
    def __init__(self, observable: Observable) -> None:
        self.count = 0
        observable.add_observer(self)

    def _update(self, observable: Observable) -> None:
        self.count += 1


@pytest.fixture(scope='module')
def registry() -> PluginRegistry:
    return PluginRegistry.load_plugins()


def _described(generator: ProbeGenerator) -> list[ParameterDescription]:
    return list(describe(generator.get_default_options(), expand_num=True))


def _make_factory(
    registry: PluginRegistry,
    *,
    kb_mirrors: PluginCatalog[KirkpatrickBaezMirrorPreset] | None = None,
) -> ProbeBuilderFactory:
    return ProbeBuilderFactory(
        numpy.random.default_rng(42),
        ProbeSettings(SettingsRegistry()),
        registry.probe_generators,
        registry.fresnel_zone_plates,
        registry.kb_mirrors if kb_mirrors is None else kb_mirrors,
        PluginChooser(PluginList()),
        PluginChooser(PluginList()),
    )


def _make_builder(
    registry: PluginRegistry,
    name: str,
    *,
    kb_mirrors: PluginCatalog[KirkpatrickBaezMirrorPreset] | None = None,
) -> PluginProbeBuilder:
    factory = _make_factory(registry, kb_mirrors=kb_mirrors)
    builder = factory.create(name, has_diffraction_data=True)
    assert isinstance(builder, PluginProbeBuilder)
    return builder


# -- generators ----------------------------------------------------------------------


def test_every_generator_but_the_data_estimate_returns_one_raw_mode(
    registry: PluginRegistry,
) -> None:
    for plugin in registry.probe_generators:
        generator = plugin.strategy
        # The default mirror pair's focus needs a finer grid than the apertures do.
        provider = _Provider(pixel_size_m=5.0e-9 if plugin.simple_name == 'kb_mirror' else 1e-7)

        if generator.reads_diffraction_data:
            with pytest.raises(GeometryNotDefinedError):
                generator.generate(generator.get_default_options(), provider)

            continue

        probe = generator.generate(generator.get_default_options(), provider)

        assert isinstance(probe, Probe), plugin.simple_name
        assert probe.num_incoherent_modes == 1, plugin.simple_name


def test_a_generated_probe_is_not_rescaled(registry: PluginRegistry) -> None:
    """Normalizing the photon count is the builder's concern, not the generator's."""
    generator = registry.probe_generators.get_strategy_by_name('disk')
    provider = _Provider()

    probe = generator.generate(generator.get_default_options(), provider)

    expected = generate_disk_probe(provider.get_probe_geometry(), radius_m=0.5e-6)
    numpy.testing.assert_array_equal(probe.get_array(), expected.get_array())


def test_keys_shared_by_several_generators_agree(registry: PluginRegistry) -> None:
    """A key shared across generators holds one value, so its declarations must agree.

    The first generator to declare a key creates its parameter, so a disagreement would
    make the stored default depend on which generator registered first.
    """
    by_key: dict[str, list[tuple[str, ParameterDescription]]] = defaultdict(list)

    for plugin in registry.probe_generators:
        for description in _described(plugin.strategy):
            by_key[description.key].append((plugin.simple_name, description))

    shared = {key: entries for key, entries in by_key.items() if len(entries) > 1}

    assert {'DefocusDistanceInMeters', 'PropagationDistanceInMeters', 'DiameterInMeters'} <= set(
        shared
    )

    for key, entries in shared.items():
        signatures = {
            (repr(d.annotation), repr(d.default), d.minimum, d.maximum) for _, d in entries
        }
        assert len(signatures) == 1, f'{key} disagrees across {[name for name, _ in entries]}'


def test_every_generator_builds_into_a_builder(registry: PluginRegistry) -> None:
    """Construction checks each size annotation, so this pins every declaration."""
    factory = _make_factory(registry)

    for name in factory:
        assert factory.create(name, has_diffraction_data=True).get_name() == name


def test_exactly_the_apertures_estimate_a_size(registry: PluginRegistry) -> None:
    factory = _make_factory(registry)
    estimating = set()

    for name in factory:
        builder = factory.create(name, has_diffraction_data=True)
        assert isinstance(builder, PluginProbeBuilder)

        if builder.estimates_size:
            estimating.add(name)

    assert estimating == {'disk', 'rectangular', 'super_gaussian'}


def test_only_the_average_pattern_reads_diffraction_data(registry: PluginRegistry) -> None:
    reading = {
        plugin.simple_name
        for plugin in registry.probe_generators
        if plugin.strategy.reads_diffraction_data
    }

    assert reading == {'average_pattern'}


class _BadExtentOptions(BaseModel):
    size_m: float = Field(1e-6, json_schema_extra={'probe_extent': 'no_such_quantity'})
    defocus_distance_m: float = 0.0


class _NoDefocusOptions(BaseModel):
    size_m: float = Field(1e-6, json_schema_extra={'probe_extent': 'disk_diameter_m'})


class _StubGenerator(ProbeGenerator):
    def __init__(self, options: BaseModel) -> None:
        self._options = options

    def get_default_options(self) -> BaseModel:
        return self._options

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        raise NotImplementedError


@pytest.mark.parametrize(
    ('options', 'message'),
    [
        (_BadExtentOptions(), 'names no ProbeExtent quantity'),
        (_NoDefocusOptions(), 'none is declared'),
    ],
)
def test_a_faulty_size_declaration_is_refused(options: BaseModel, message: str) -> None:
    generators = PluginList[ProbeGenerator]()
    generators.register_plugin(_StubGenerator(options), simple_name='stub', display_name='stub')
    factory = ProbeBuilderFactory(
        numpy.random.default_rng(42),
        ProbeSettings(SettingsRegistry()),
        generators,
        PluginList(),
        PluginList(),
        PluginChooser(PluginList()),
        PluginChooser(PluginList()),
    )

    with pytest.raises(ValueError, match=message):
        factory.create('stub')


def test_the_coefficient_list_is_the_mode_list(registry: PluginRegistry) -> None:
    builder = _make_builder(registry, 'zernike')
    coefficients = [1 + 0j, 0.5j, -0.25 + 0j]
    builder.parameters()['zernike_coefficients'].set_value(coefficients)
    provider = _Provider()

    built = builder.build(provider).get_probe_no_opr().get_array()[0]

    expected = generate_zernike_probe(
        provider.get_probe_geometry(),
        [ZernikeMode.from_osa_index(index, value) for index, value in enumerate(coefficients)],
        radius_m=0.5e-6,
    ).get_array()[0]
    numpy.testing.assert_allclose(
        built / numpy.abs(built).max(), expected / numpy.abs(expected).max()
    )


@pytest.mark.parametrize(
    ('name', 'field'),
    [('zernike', 'zernike_coefficients'), ('hermite_gauss', 'hermite_gauss_coefficients')],
)
def test_an_empty_coefficient_list_is_refused(
    registry: PluginRegistry, name: str, field: str
) -> None:
    generator = registry.probe_generators.get_strategy_by_name(name)
    options = generator.get_default_options().model_copy(update={field: []})

    with pytest.raises(ValueError, match='at least one coefficient'):
        generator.generate(options, _Provider())


def test_coefficients_set_as_literals_are_stored_as_complex(registry: PluginRegistry) -> None:
    """The path a workflow parameter mapping takes into a builder."""
    builder = _make_builder(registry, 'zernike')

    builder.parameters()['zernike_coefficients'].set_value(['1+0j', '0.5j'])

    assert builder.parameters()['zernike_coefficients'].get_value() == [1 + 0j, 0.5j]
    assert builder.build(_Provider()).num_incoherent_modes == 1


# -- presets -------------------------------------------------------------------------


def test_each_preset_type_fits_exactly_one_generator(registry: PluginRegistry) -> None:
    """Checked on the types, because no mirror pair is registered to read one off."""
    for preset_type in (FresnelZonePlate, KirkpatrickBaezMirrorPreset):
        preset_fields = fields_of(preset_type)
        fitting = []

        for plugin in registry.probe_generators:
            described = {d.name: d for d in _described(plugin.strategy)}

            if set(preset_fields) <= set(described):
                fitting.append(plugin.simple_name)

                for name, info in preset_fields.items():
                    assert described[name].annotation == info.annotation, name

        assert len(fitting) == 1, f'{preset_type.__name__} fits {fitting}'


def test_every_zone_plate_goes_by_one_name(registry: PluginRegistry) -> None:
    assert len(list(registry.fresnel_zone_plates)) > 0

    for plugin in registry.fresnel_zone_plates:
        assert plugin.simple_name == plugin.display_name


def test_a_zone_plate_preset_fills_its_fields_and_nothing_else(registry: PluginRegistry) -> None:
    builder = _make_builder(registry, 'fresnel_zone_plate')
    builder.parameters()['defocus_distance_m'].set_value(1.0e-4)
    counter = _NotificationCounter(builder)
    label = 'APS 31-ID-E LamNI'

    assert label in list(builder.labels_for_presets())

    builder.apply_presets(label)

    zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(label)
    parameters = builder.parameters()
    assert parameters['zone_plate_diameter_m'].get_value() == zone_plate.zone_plate_diameter_m
    assert parameters['outermost_zone_width_m'].get_value() == zone_plate.outermost_zone_width_m
    assert (
        parameters['central_beamstop_diameter_m'].get_value()
        == zone_plate.central_beamstop_diameter_m
    )
    assert parameters['defocus_distance_m'].get_value() == 1.0e-4
    assert counter.count == 1


def test_a_mirror_preset_fills_its_six_fields() -> None:
    registry = PluginRegistry.load_plugins()
    preset = KirkpatrickBaezMirrorPreset(
        horizontal_acceptance_length_m=0.2,
        horizontal_grazing_angle_rad=2e-3,
        horizontal_focus_distance_m=0.1,
        vertical_acceptance_length_m=0.15,
        vertical_grazing_angle_rad=4e-3,
        vertical_focus_distance_m=0.2,
    )
    # The loaded registry is frozen, so the preset goes into a catalog of its own.
    mirrors = PluginList[KirkpatrickBaezMirrorPreset]()
    mirrors.register_plugin(preset, simple_name='Test Pair', display_name='Test Pair')
    builder = _make_builder(registry, 'kb_mirror', kb_mirrors=mirrors)

    builder.apply_presets('Test Pair')

    values: dict[str, Any] = preset.model_dump()
    assert {name: builder.parameters()[name].get_value() for name in values} == values
    assert builder.parameters()['astigmatism_m'].get_value() == 0.0


def test_an_unknown_preset_is_refused(registry: PluginRegistry) -> None:
    builder = _make_builder(registry, 'fresnel_zone_plate')

    with pytest.raises(KeyError, match='No preset'):
        builder.apply_presets('No Such Optic')


def test_a_generator_no_preset_fits_offers_none(registry: PluginRegistry) -> None:
    """The disk is handed the zone plates too, and declares none of their fields."""
    builder = _make_builder(registry, 'disk')

    assert list(builder.labels_for_presets()) == []

    with pytest.raises(KeyError, match='No preset'):
        builder.apply_presets('APS 2-ID-D')


def test_a_preset_registered_after_the_builder_is_offered() -> None:
    registry = PluginRegistry.load_plugins()
    # The loaded registry is frozen; a catalog still open to registration shows that
    # the builder reads its presets live rather than from a snapshot.
    mirrors = PluginList[KirkpatrickBaezMirrorPreset]()
    builder = _make_builder(registry, 'kb_mirror', kb_mirrors=mirrors)

    assert list(builder.labels_for_presets()) == []

    mirrors.register_plugin(
        KirkpatrickBaezMirrorPreset(
            horizontal_acceptance_length_m=0.2,
            horizontal_grazing_angle_rad=2e-3,
            horizontal_focus_distance_m=0.1,
            vertical_acceptance_length_m=0.15,
            vertical_grazing_angle_rad=4e-3,
            vertical_focus_distance_m=0.2,
        ),
        simple_name='Late Pair',
        display_name='Late Pair',
    )

    assert list(builder.labels_for_presets()) == ['Late Pair']


def test_a_copy_keeps_its_presets(registry: PluginRegistry) -> None:
    builder = _make_builder(registry, 'fresnel_zone_plate')

    assert list(builder.copy().labels_for_presets()) == list(builder.labels_for_presets())
