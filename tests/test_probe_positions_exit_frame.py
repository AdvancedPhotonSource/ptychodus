"""A product's positions follow its scattering geometry into the exit frame and back."""

import numpy
import pytest

from ptychodus.api.affine import AffineTransform
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.scattering.frame import compute_exit_frame_transform
from ptychodus.api.scattering.geometry import ScatteringGeometry
from ptychodus.model.product.metadata import MetadataRepositoryItem, UniqueNameFactory
from ptychodus.model.product.probe_positions.builder import FromMemoryProbePositionsBuilder
from ptychodus.model.product.probe_positions.item import ProbePositionsRepositoryItem
from ptychodus.model.product.probe_positions.settings import ProbePositionsSettings
from ptychodus.model.product.settings import ProductSettings
from ptychodus.model.settings import SettingsRegistry

_BRAGG = ScatteringGeometry(
    two_theta_deg=47.35, detector_gamma_deg=-0.27, incidence_angle_deg=24.15
)
_LAUE = ScatteringGeometry(two_theta_deg=57.42, incidence_angle_deg=90.0)


class _NameFactory(UniqueNameFactory):
    def create_unique_name(self, candidate_name: str) -> str:
        return candidate_name


def _lab_positions() -> ProbePositionSequence:
    return ProbePositionSequence(
        [
            ProbePosition(index=index, x_m=1e-7 * index, y_m=-2e-7 * (index % 3))
            for index in range(9)
        ]
    )


def _make_item(
    geometry: ScatteringGeometry, *, far_field: bool = True
) -> tuple[ProbePositionsRepositoryItem, MetadataRepositoryItem]:
    registry = SettingsRegistry()
    metadata_item = MetadataRepositoryItem(
        ProductSettings(registry),
        _NameFactory(),
        far_field=far_field,
        scattering_geometry=geometry,
    )
    settings = ProbePositionsSettings(registry)
    rng = numpy.random.default_rng(0)
    builder = FromMemoryProbePositionsBuilder(rng, settings, _lab_positions())
    item = ProbePositionsRepositoryItem(rng, settings, builder, metadata_item=metadata_item)
    return item, metadata_item


def _coordinates(positions: ProbePositionSequence) -> numpy.ndarray:
    return numpy.column_stack((positions.get_coordinates_x_m(), positions.get_coordinates_y_m()))


def _expected(geometry: ScatteringGeometry) -> numpy.ndarray:
    transform = compute_exit_frame_transform(geometry)
    return _coordinates(ProbePositionSequence([transform(p) for p in _lab_positions()]))


def test_lab_positions_are_exposed_in_the_exit_frame() -> None:
    item, _ = _make_item(_BRAGG)

    numpy.testing.assert_allclose(
        _coordinates(item.get_probe_positions()), _expected(_BRAGG), atol=1e-18
    )
    numpy.testing.assert_array_equal(item.get_probe_positions().get_indexes(), numpy.arange(9))


def test_a_forward_geometry_leaves_the_positions_alone() -> None:
    item, _ = _make_item(ScatteringGeometry())

    numpy.testing.assert_array_equal(
        _coordinates(item.get_probe_positions()), _coordinates(_lab_positions())
    )


def test_editing_an_angle_remaps_the_positions_exactly() -> None:
    item, metadata_item = _make_item(_BRAGG)

    metadata_item.two_theta_deg.set_value(_LAUE.two_theta_deg)
    metadata_item.detector_gamma_deg.set_value(_LAUE.detector_gamma_deg)
    metadata_item.incidence_angle_deg.set_value(_LAUE.incidence_angle_deg)

    numpy.testing.assert_allclose(
        _coordinates(item.get_probe_positions()), _expected(_LAUE), atol=1e-18
    )


def test_exit_frame_positions_are_not_transformed_twice() -> None:
    """Reconstruction output arrives in the exit frame and stays exactly where it was put."""
    item, _ = _make_item(_BRAGG)
    exit_positions = item.get_probe_positions()

    item.assign(exit_positions)

    numpy.testing.assert_allclose(
        _coordinates(item.get_probe_positions()), _coordinates(exit_positions), atol=1e-20
    )


def test_assigned_positions_follow_a_later_geometry_edit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    item, metadata_item = _make_item(_BRAGG)
    item.assign(item.get_probe_positions())

    metadata_item.two_theta_deg.set_value(0.0)
    metadata_item.detector_gamma_deg.set_value(0.0)

    numpy.testing.assert_allclose(
        _coordinates(item.get_probe_positions()), _coordinates(_lab_positions()), atol=1e-20
    )
    assert 'previous exit frame' in caplog.text


def test_near_field_is_left_untransformed_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    item, _ = _make_item(_BRAGG, far_field=False)

    numpy.testing.assert_array_equal(
        _coordinates(item.get_probe_positions()), _coordinates(_lab_positions())
    )
    assert 'far-field' in caplog.text


def test_without_metadata_the_transform_is_the_identity() -> None:
    settings = ProbePositionsSettings(SettingsRegistry())
    rng = numpy.random.default_rng(0)
    item = ProbePositionsRepositoryItem(
        rng, settings, FromMemoryProbePositionsBuilder(rng, settings, _lab_positions())
    )

    assert item.get_exit_transform() == AffineTransform.create_identity()
