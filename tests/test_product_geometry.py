"""Unit tests for ProductGeometry's reciprocal-plane derivations.

``get_object_plane_pixel_geometry`` and ``fresnel_number`` both describe the sample
plane, which is conjugate to the detector across the propagation. These pin the plane
each one reports on -- the pair are reciprocal, so reporting the wrong one is not a
small error but an inversion.
"""

from unittest.mock import MagicMock

import pytest

from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.geometry import ProductGeometry
from ptychodus.model.product.metadata import MetadataRepositoryItem
from ptychodus.model.product.settings import ProductSettings


_NUM_PX = 256
_DETECTOR_PITCH_M = 75e-6
_PROBE_ENERGY_EV = 10000.0
_DETECTOR_DISTANCE_M = 1.0
_WAVELENGTH_M = energy_eV_to_wavelength_m(_PROBE_ENERGY_EV)


class _NameFactory:
    def create_unique_name(self, candidate_name: str) -> str:
        return candidate_name


def _make_geometry(
    *,
    detector_distance_m: float = _DETECTOR_DISTANCE_M,
    bind_detector: bool = True,
) -> ProductGeometry:
    metadata_item = MetadataRepositoryItem(
        ProductSettings(SettingsRegistry()),
        _NameFactory(),
        detector_distance_m=detector_distance_m,
        probe_energy_eV=_PROBE_ENERGY_EV,
    )
    geometry = ProductGeometry(metadata_item, MagicMock())

    if bind_detector:
        geometry.set_detector_extent(ImageExtent(width_px=_NUM_PX, height_px=_NUM_PX))
        geometry.set_detector_pixel_geometry(
            PixelGeometry(width_m=_DETECTOR_PITCH_M, height_m=_DETECTOR_PITCH_M)
        )

    return geometry


class TestObjectPlanePixelGeometry:
    def test_is_the_conjugate_of_the_detector_pixel(self) -> None:
        expected_m = _WAVELENGTH_M * _DETECTOR_DISTANCE_M / (_NUM_PX * _DETECTOR_PITCH_M)

        pixel_geometry = _make_geometry().get_object_plane_pixel_geometry()

        assert pixel_geometry.width_m == pytest.approx(expected_m, rel=1e-12)
        assert pixel_geometry.height_m == pytest.approx(expected_m, rel=1e-12)

    def test_is_an_involution(self) -> None:
        """Applying the conjugate relation twice returns the detector pixel."""
        geometry = _make_geometry()
        object_plane = geometry.get_object_plane_pixel_geometry()

        round_trip_m = _WAVELENGTH_M * _DETECTOR_DISTANCE_M / (_NUM_PX * object_plane.width_m)
        assert round_trip_m == pytest.approx(_DETECTOR_PITCH_M, rel=1e-12)

    def test_degrades_to_zero_while_no_dataset_is_bound(self) -> None:
        """The detector extent is 0 px until bind_dataset runs; dividing by it must not
        propagate a ZeroDivisionError into the GUI."""
        pixel_geometry = _make_geometry(bind_detector=False).get_object_plane_pixel_geometry()

        assert pixel_geometry == PixelGeometry(width_m=0.0, height_m=0.0)
        assert not pixel_geometry.is_valid

    def test_collapses_at_zero_detector_distance(self) -> None:
        """``lambda z / (N dx)`` vanishes with z. The plane is degenerate, not undefined."""
        pixel_geometry = _make_geometry(detector_distance_m=0.0).get_object_plane_pixel_geometry()

        assert pixel_geometry == PixelGeometry(width_m=0.0, height_m=0.0)


class TestFresnelNumber:
    def test_reports_the_object_plane_regime(self) -> None:
        """A typical APS geometry is deeply far field, so the reported number is far
        below one. The detector-plane number for the same geometry is ~2.97e+06, which
        reads as 'wildly near field' -- the opposite conclusion.
        """
        # W_obj * H_obj / (lambda z) with W_obj = lambda z / dx_detector.
        expected = _WAVELENGTH_M * _DETECTOR_DISTANCE_M / _DETECTOR_PITCH_M**2

        fresnel_number = _make_geometry().fresnel_number

        assert fresnel_number == pytest.approx(expected, rel=1e-12)
        assert fresnel_number == pytest.approx(0.022041, rel=1e-4)
        assert fresnel_number < 1.0

    def test_times_the_detector_plane_number_is_the_pixel_count(self) -> None:
        """``Fr_detector * Fr_object == width_px * height_px`` exactly -- the identity
        that makes the choice of plane an inversion rather than a rescaling.
        """
        geometry = _make_geometry()
        detector_extent_m2 = (_NUM_PX * _DETECTOR_PITCH_M) ** 2
        detector_fresnel_number = detector_extent_m2 / (_WAVELENGTH_M * _DETECTOR_DISTANCE_M)

        product = detector_fresnel_number * geometry.fresnel_number

        assert product == pytest.approx(_NUM_PX * _NUM_PX, rel=1e-12)

    def test_degrades_to_zero_while_no_dataset_is_bound(self) -> None:
        assert _make_geometry(bind_detector=False).fresnel_number == 0.0

    def test_degrades_to_zero_at_zero_detector_distance(self) -> None:
        """The correct object-plane limit: ``W_obj^2 / (lambda z) = lambda z / dx_d^2 -> 0``.

        At the detector plane the same limit diverges, which is why the product editor
        used to need an 'inf' branch here.
        """
        assert _make_geometry(detector_distance_m=0.0).fresnel_number == 0.0
