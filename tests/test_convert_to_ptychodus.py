"""Plugin resolution, fold_slice pairing, and product rewriting in convert-to-ptychodus.

The script picks a plugin for each ``--*-type``, derives a fold_slice ``_para`` file from
its ``_dp`` partner, and folds the metadata and component arguments onto the product it
read. None of that shows up in the output file as anything but wrong data, so it is pinned
down here: that each override lands on its own field and leaves the others alone, that a
component file carrying no geometry of its own inherits the product's, that the detector
pixel pitch resolves in the documented order, that the companion name is derived only for
names that follow the convention, that the probe energy and detector distance come from
the parameter file unless the command line says otherwise, and that a name no plugin
answers to is an error rather than a different plugin.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy
import pytest

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectFileReader
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeFileReader, ProbeSequence
from ptychodus.api.probe_positions import (
    ProbePosition,
    ProbePositionFileReader,
    ProbePositionSequence,
)
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.cli.convert_to_ptychodus import (
    _apply_metadata_overrides,
    _apply_product_overrides,
    _built_product_name,
    _inherit_object_geometry,
    _inherit_probe_geometry,
    _read_fold_slice_parameters_if_available,
    _resolve_built_product_geometry,
    _resolve_position_file,
    _resolve_raw_pixel_geometry,
)
from ptychodus.plugins.fold_slice.position_file import FoldSliceParameters

PIXEL_M = 1.0e-9
CENTER_M = 1.5e-6
OBJ_HEIGHT_PX = 16
OBJ_WIDTH_PX = 20
PROBE_HEIGHT_PX = 8
PROBE_WIDTH_PX = 8
NUM_POSITIONS = 3

PLUGIN_NAME = 'Override'


def _make_object(value: complex, *, with_geometry: bool = True) -> Object:
    return Object(
        array=numpy.full((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX), value, dtype=numpy.complex128),
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M) if with_geometry else None,
        center=ObjectCenter(x_m=CENTER_M, y_m=CENTER_M) if with_geometry else None,
        layer_spacing_m=[],
    )


def _make_probes(value: complex, *, with_geometry: bool = True) -> ProbeSequence:
    return ProbeSequence(
        array=numpy.full((1, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX), value, dtype=numpy.complex128),
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M) if with_geometry else None,
    )


def _make_positions(offset_m: float) -> ProbePositionSequence:
    return ProbePositionSequence(
        [
            ProbePosition(index=i, x_m=offset_m + i * PIXEL_M, y_m=offset_m)
            for i in range(NUM_POSITIONS)
        ]
    )


def _make_metadata() -> ProductMetadata:
    # The zeros are what the fold_slice readers produce for what their formats omit.
    return ProductMetadata(
        name='original',
        comments='unchanged',
        detector_distance_m=0.0,
        photon_energy_eV=10_000.0,
        probe_photon_count=0.0,
        exposure_time_s=0.0,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
    )


def _make_product() -> Product:
    return Product(
        metadata=_make_metadata(),
        probe_positions=_make_positions(0.0),
        probes=_make_probes(1.0 + 0.0j),
        object_=_make_object(1.0 + 0.0j),
        losses=[],
    )


class _StubObjectFileReader(ObjectFileReader):
    def read(self, file_path: Path) -> Object:
        return _make_object(2.0 + 0.0j, with_geometry=False)


class _StubProbeFileReader(ProbeFileReader):
    def read(self, file_path: Path) -> ProbeSequence:
        return _make_probes(3.0 + 0.0j, with_geometry=False)


class _StubProbePositionFileReader(ProbePositionFileReader):
    def read(self, file_path: Path) -> ProbePositionSequence:
        return _make_positions(5.0)


@pytest.fixture
def registry() -> PluginRegistry:
    """A registry holding only the stub readers, so no real plugin can answer instead."""
    registry = PluginRegistry()
    registry.object_file_readers.register_plugin(
        _StubObjectFileReader(), display_name=PLUGIN_NAME, simple_name=PLUGIN_NAME
    )
    registry.probe_file_readers.register_plugin(
        _StubProbeFileReader(), display_name=PLUGIN_NAME, simple_name=PLUGIN_NAME
    )
    registry.probe_position_file_readers.register_plugin(
        _StubProbePositionFileReader(), display_name=PLUGIN_NAME, simple_name=PLUGIN_NAME
    )
    return registry


def _args(**overrides: object) -> argparse.Namespace:
    namespace = argparse.Namespace(
        product_name=None,
        detector_distance_m=None,
        photon_energy_eV=None,
        probe_photon_count=None,
        exposure_time_s=None,
        mass_attenuation_m2_per_kg=None,
        tomography_angle_deg=None,
        override_object=None,
        override_object_type=PLUGIN_NAME,
        override_probe=None,
        override_probe_type=PLUGIN_NAME,
        probe_positions=None,
        probe_positions_type=PLUGIN_NAME,
        diffraction_input=None,
        diffraction_input_type='fold_slice',
        seed=0,
    )

    for key, value in overrides.items():
        setattr(namespace, key, value)

    return namespace


def test_no_overrides_leaves_the_product_alone(registry: PluginRegistry) -> None:
    product = _make_product()

    assert _apply_product_overrides(registry, product, _args()) is product


def test_product_name_rewrites_only_the_name(registry: PluginRegistry) -> None:
    product = _make_product()

    converted = _apply_product_overrides(registry, product, _args(product_name='renamed'))

    assert converted.metadata.name == 'renamed'
    assert converted.metadata.comments == product.metadata.comments
    assert converted.metadata.photon_energy_eV == product.metadata.photon_energy_eV
    assert converted.object_ is product.object_
    assert converted.probes is product.probes
    assert converted.probe_positions is product.probe_positions


def test_metadata_arguments_fill_in_what_the_format_omits() -> None:
    # fold_slice_mat hardcodes 0.0 for all three of these, and a zero detector distance
    # collapses the sample-plane pixel size wherever it is derived.
    converted = _apply_metadata_overrides(
        _make_metadata(),
        _args(detector_distance_m=2.335, probe_photon_count=1.0e9, exposure_time_s=0.05),
    )

    assert converted.detector_distance_m == 2.335
    assert converted.probe_photon_count == 1.0e9
    assert converted.exposure_time_s == 0.05


def test_metadata_arguments_that_were_not_given_leave_the_file_value() -> None:
    metadata = _make_metadata()

    converted = _apply_metadata_overrides(metadata, _args(detector_distance_m=2.335))

    assert converted.photon_energy_eV == metadata.photon_energy_eV
    assert converted.comments == metadata.comments
    assert converted.tomography_angle_deg == metadata.tomography_angle_deg


def test_object_override_replaces_only_the_object(registry: PluginRegistry, tmp_path: Path) -> None:
    product = _make_product()

    converted = _apply_product_overrides(
        registry, product, _args(override_object=tmp_path / 'object.npy')
    )

    assert converted.object_.get_array()[0, 0, 0] == 2.0 + 0.0j
    assert converted.object_.get_pixel_geometry() == product.object_.get_pixel_geometry()
    assert converted.object_.get_center() == product.object_.get_center()
    assert converted.probes is product.probes
    assert converted.probe_positions is product.probe_positions


def test_probe_override_replaces_only_the_probe(registry: PluginRegistry, tmp_path: Path) -> None:
    product = _make_product()

    converted = _apply_product_overrides(
        registry, product, _args(override_probe=tmp_path / 'probe.npy')
    )

    assert converted.probes.get_array()[0, 0, 0, 0] == 3.0 + 0.0j
    assert converted.probes.get_pixel_geometry() == product.probes.get_pixel_geometry()
    assert converted.object_ is product.object_
    assert converted.probe_positions is product.probe_positions


def test_probe_positions_replace_only_the_positions(
    registry: PluginRegistry, tmp_path: Path
) -> None:
    product = _make_product()

    converted = _apply_product_overrides(
        registry, product, _args(probe_positions=tmp_path / 'positions.csv')
    )

    assert converted.probe_positions[0].y_m == 5.0
    assert converted.object_ is product.object_
    assert converted.probes is product.probes


def test_every_override_applies_together(registry: PluginRegistry, tmp_path: Path) -> None:
    converted = _apply_product_overrides(
        registry,
        _make_product(),
        _args(
            product_name='renamed',
            detector_distance_m=2.335,
            override_object=tmp_path / 'object.npy',
            override_probe=tmp_path / 'probe.npy',
            probe_positions=tmp_path / 'positions.csv',
        ),
    )

    assert converted.metadata.name == 'renamed'
    assert converted.metadata.detector_distance_m == 2.335
    assert converted.object_.get_array()[0, 0, 0] == 2.0 + 0.0j
    assert converted.probes.get_array()[0, 0, 0, 0] == 3.0 + 0.0j
    assert converted.probe_positions[0].y_m == 5.0


def test_an_unregistered_type_names_the_registered_plugins(
    registry: PluginRegistry, tmp_path: Path
) -> None:
    # PluginChooser.set_current_plugin warns and keeps its selection, which would convert
    # the file with a reader other than the one asked for; get_strategy_by_name raises.
    with pytest.raises(LookupError, match=PLUGIN_NAME):
        _apply_product_overrides(
            registry,
            _make_product(),
            _args(override_object=tmp_path / 'object.npy', override_object_type='nonesuch'),
        )


def test_an_object_override_inherits_the_geometry_it_lacks() -> None:
    # No object format records a pixel size or a center, so without this the override
    # cannot be written back out at all -- save_product raises on a geometry-less object.
    pixel_geometry = PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M)
    center = ObjectCenter(x_m=CENTER_M, y_m=CENTER_M)

    inherited = _inherit_object_geometry(
        _make_object(2.0 + 0.0j, with_geometry=False),
        pixel_geometry=pixel_geometry,
        center=center,
    )

    assert inherited.get_array()[0, 0, 0] == 2.0 + 0.0j
    assert inherited.get_pixel_geometry() == pixel_geometry
    assert inherited.get_center() == center


def test_an_object_override_keeps_the_geometry_it_records() -> None:
    override = Object(
        array=numpy.full((1, OBJ_HEIGHT_PX, OBJ_WIDTH_PX), 2.0 + 0.0j, dtype=numpy.complex128),
        pixel_geometry=PixelGeometry(width_m=2.0 * PIXEL_M, height_m=3.0 * PIXEL_M),
        center=ObjectCenter(x_m=-CENTER_M, y_m=CENTER_M),
        layer_spacing_m=[],
    )

    inherited = _inherit_object_geometry(
        override,
        pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
    )

    assert inherited.get_pixel_geometry() == override.get_pixel_geometry()
    assert inherited.get_center() == override.get_center()


def test_a_probe_override_inherits_the_geometry_it_lacks() -> None:
    pixel_geometry = PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M)

    inherited = _inherit_probe_geometry(
        _make_probes(3.0 + 0.0j, with_geometry=False), pixel_geometry=pixel_geometry
    )

    assert inherited.get_array()[0, 0, 0, 0] == 3.0 + 0.0j
    assert inherited.get_pixel_geometry() == pixel_geometry


def test_a_probe_override_keeps_its_opr_weights() -> None:
    override = ProbeSequence(
        array=numpy.full(
            (2, 1, PROBE_HEIGHT_PX, PROBE_WIDTH_PX), 3.0 + 0.0j, dtype=numpy.complex128
        ),
        opr_weights=numpy.ones((NUM_POSITIONS, 2)),
        pixel_geometry=None,
    )

    inherited = _inherit_probe_geometry(
        override, pixel_geometry=PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M)
    )

    assert inherited.get_opr_weights().shape == (NUM_POSITIONS, 2)


def test_the_companion_is_found_for_a_fold_slice_input(tmp_path: Path) -> None:
    diffraction_file = tmp_path / 'data_roi0_dp.hdf5'
    position_file = tmp_path / 'data_roi0_para.hdf5'
    diffraction_file.touch()
    position_file.touch()

    resolved = _resolve_position_file(_args(diffraction_input=diffraction_file))

    assert resolved == (position_file, 'fold_slice')


def test_no_companion_is_derived_for_another_diffraction_format(tmp_path: Path) -> None:
    # Only fold_slice writes this pair, so the convention must not be assumed elsewhere.
    diffraction_file = tmp_path / 'data_roi0_dp.hdf5'
    diffraction_file.touch()
    (tmp_path / 'data_roi0_para.hdf5').touch()

    resolved = _resolve_position_file(
        _args(diffraction_input=diffraction_file, diffraction_input_type='APS_Velociprobe')
    )

    assert resolved is None


def test_a_missing_companion_resolves_to_nothing(tmp_path: Path) -> None:
    diffraction_file = tmp_path / 'data_roi0_dp.hdf5'
    diffraction_file.touch()

    assert _resolve_position_file(_args(diffraction_input=diffraction_file)) is None


def test_an_explicit_position_file_wins_over_the_companion(tmp_path: Path) -> None:
    diffraction_file = tmp_path / 'data_roi0_dp.hdf5'
    diffraction_file.touch()
    (tmp_path / 'data_roi0_para.hdf5').touch()
    chosen = tmp_path / 'elsewhere.csv'

    resolved = _resolve_position_file(
        _args(
            diffraction_input=diffraction_file,
            probe_positions=chosen,
            probe_positions_type='CSV',
        )
    )

    assert resolved == (chosen, 'CSV')


# The IC_1 scan of the NXSchool set, which is the shape this is pointed at: a 256 x 256
# Eiger frame, 8 keV, and the sample pixel size the preprocessing recorded for it.
NXS_WAVELENGTH_M = 1.549802e-10
NXS_OBJECT_PIXEL_SIZE_M = 1.884786e-08
NXS_DETECTOR_PITCH_M = 75.0e-6
NXS_DETECTOR_PX = 256
NXS_DETECTOR_DISTANCE_M = 2.335


def _assembled(num_patterns: int = 2) -> AssembledDiffractionData:
    return AssembledDiffractionData(
        indexes=numpy.arange(num_patterns),
        patterns=numpy.ones((num_patterns, NXS_DETECTOR_PX, NXS_DETECTOR_PX), dtype=numpy.float32),
        pixel_geometry=PixelGeometry(width_m=NXS_DETECTOR_PITCH_M, height_m=NXS_DETECTOR_PITCH_M),
        bad_pixels=numpy.zeros((NXS_DETECTOR_PX, NXS_DETECTOR_PX), dtype=numpy.bool_),
    )


def _nxs_parameters() -> FoldSliceParameters:
    return FoldSliceParameters(
        photon_wavelength_m=NXS_WAVELENGTH_M,
        object_pixel_size_m=NXS_OBJECT_PIXEL_SIZE_M,
        tomography_angle_deg=17.5,
    )


def test_the_parameter_file_supplies_the_geometry() -> None:
    geometry = _resolve_built_product_geometry(_args(), _nxs_parameters(), _assembled())

    assert geometry.photon_energy_eV == pytest.approx(8000.0)
    assert geometry.detector_distance_m == pytest.approx(NXS_DETECTOR_DISTANCE_M)
    assert geometry.tomography_angle_deg == pytest.approx(17.5)


def test_the_command_line_overrides_the_parameter_file() -> None:
    geometry = _resolve_built_product_geometry(
        _args(detector_distance_m=1.5, photon_energy_eV=12_000.0, tomography_angle_deg=90.0),
        _nxs_parameters(),
        _assembled(),
    )

    assert geometry.photon_energy_eV == pytest.approx(12_000.0)
    assert geometry.detector_distance_m == pytest.approx(1.5)
    assert geometry.tomography_angle_deg == pytest.approx(90.0)


def test_an_overridden_energy_also_moves_the_derived_distance() -> None:
    # The distance comes out of the same relation the energy enters, so the two cannot be
    # resolved independently: a hand-given energy has to feed the derivation.
    geometry = _resolve_built_product_geometry(
        _args(photon_energy_eV=16_000.0), _nxs_parameters(), _assembled()
    )

    assert geometry.detector_distance_m == pytest.approx(2.0 * NXS_DETECTOR_DISTANCE_M, rel=1e-4)


def test_a_missing_wavelength_names_the_energy_argument() -> None:
    with pytest.raises(ValueError, match='--probe-energy-eV'):
        _resolve_built_product_geometry(_args(), FoldSliceParameters(), _assembled())


def test_a_missing_object_pixel_size_names_the_distance_argument() -> None:
    parameters = FoldSliceParameters(photon_wavelength_m=NXS_WAVELENGTH_M)

    with pytest.raises(ValueError, match='--detector-distance-m'):
        _resolve_built_product_geometry(_args(), parameters, _assembled())


def test_an_unrecorded_tomography_angle_is_zero() -> None:
    parameters = FoldSliceParameters(
        photon_wavelength_m=NXS_WAVELENGTH_M, object_pixel_size_m=NXS_OBJECT_PIXEL_SIZE_M
    )

    geometry = _resolve_built_product_geometry(_args(), parameters, _assembled())

    assert geometry.tomography_angle_deg == 0.0


def test_no_parameters_are_read_for_another_position_format(tmp_path: Path) -> None:
    # Only fold_slice writes these datasets; reading a CSV as one would be a type error
    # at best and a wrong geometry at worst.
    assert (
        _read_fold_slice_parameters_if_available(tmp_path / 'positions.csv', 'CSV')
        == FoldSliceParameters()
    )


def test_a_built_product_is_named_after_its_scan_directory() -> None:
    # fold_slice gives every scan the same generic pattern filename, so the stem would
    # name five different scans identically.
    args = _args(diffraction_input=Path('/scans/IC_1/data_roi0_Ndp256_rs1_dp.hdf5'))

    assert _built_product_name(args) == 'IC_1'


def test_an_explicit_product_name_wins() -> None:
    args = _args(
        product_name='chosen',
        diffraction_input=Path('/scans/IC_1/data_roi0_Ndp256_rs1_dp.hdf5'),
    )

    assert _built_product_name(args) == 'chosen'


def test_the_pixel_size_argument_wins_over_the_file() -> None:
    geometry = _resolve_raw_pixel_geometry(
        75.0e-6, PixelGeometry(width_m=55.0e-6, height_m=55.0e-6)
    )

    assert geometry == PixelGeometry(width_m=75.0e-6, height_m=75.0e-6)


def test_the_file_supplies_the_pixel_size_when_the_argument_does_not() -> None:
    from_file = PixelGeometry(width_m=55.0e-6, height_m=60.0e-6)

    assert _resolve_raw_pixel_geometry(None, from_file) is from_file


def test_a_pixel_size_from_nowhere_names_the_argument_to_pass() -> None:
    with pytest.raises(ValueError, match='--detector-pixel-size-m'):
        _resolve_raw_pixel_geometry(None, None)
