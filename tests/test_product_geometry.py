"""Unit tests for ProductGeometryProvider's reciprocal-plane derivations.

``get_object_plane_pixel_geometry`` and ``fresnel_number`` both describe the sample
plane, which is conjugate to the detector across the propagation. These pin the plane
each one reports on -- the pair are reciprocal, so reporting the wrong one is not a
small error but an inversion.
"""

from unittest.mock import MagicMock
import math

import pytest

from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import GeometryNotDefinedError, ImageExtent, PixelGeometry
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.geometry import ProductGeometryProvider
from ptychodus.model.product.metadata import MetadataRepositoryItem, UniqueNameFactory
from ptychodus.model.product.settings import ProductSettings


_NUM_PX = 256
_DETECTOR_PITCH_M = 75e-6
_PHOTON_ENERGY_EV = 10000.0
_DETECTOR_DISTANCE_M = 1.0
_WAVELENGTH_M = energy_eV_to_wavelength_m(_PHOTON_ENERGY_EV)


class _NameFactory(UniqueNameFactory):
    def create_unique_name(self, candidate_name: str) -> str:
        return candidate_name


def _make_geometry(
    *,
    detector_distance_m: float = _DETECTOR_DISTANCE_M,
    focus_object_distance_m: float = 0.0,
    far_field: bool = True,
    bind_detector: bool = True,
) -> ProductGeometryProvider:
    metadata_item = MetadataRepositoryItem(
        ProductSettings(SettingsRegistry()),
        _NameFactory(),
        detector_distance_m=detector_distance_m,
        focus_object_distance_m=focus_object_distance_m,
        photon_energy_eV=_PHOTON_ENERGY_EV,
        far_field=far_field,
    )
    geometry = ProductGeometryProvider(metadata_item, MagicMock())

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
        with pytest.raises(GeometryNotDefinedError):
            _make_geometry(bind_detector=False).get_object_plane_pixel_geometry()

    def test_collapses_at_zero_detector_distance(self) -> None:
        """``lambda z / (N dx)`` vanishes with z, so there is no sampling to report.

        The pure function still returns the zero pitch; the provider is what turns a
        collapsed plane into a raise.
        """
        values = _make_geometry(detector_distance_m=0.0).get_derived_values()
        assert values.object_plane_pixel_geometry == PixelGeometry(width_m=0.0, height_m=0.0)

        with pytest.raises(GeometryNotDefinedError):
            _make_geometry(detector_distance_m=0.0).get_object_plane_pixel_geometry()


class TestFresnelNumber:
    def test_reports_the_object_plane_regime(self) -> None:
        """A typical APS geometry is deeply far field, so the reported number is far
        below one. The detector-plane number for the same geometry is ~2.97e+06, which
        reads as 'wildly near field' -- the opposite conclusion.
        """
        # W_obj * H_obj / (lambda z) with W_obj = lambda z / dx_detector.
        expected = _WAVELENGTH_M * _DETECTOR_DISTANCE_M / _DETECTOR_PITCH_M**2

        fresnel_number = _make_geometry().get_derived_values().fresnel_number

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

        product = detector_fresnel_number * geometry.get_derived_values().fresnel_number

        assert product == pytest.approx(_NUM_PX * _NUM_PX, rel=1e-12)

    def test_degrades_to_zero_while_no_dataset_is_bound(self) -> None:
        assert _make_geometry(bind_detector=False).get_derived_values().fresnel_number == 0.0

    def test_degrades_to_zero_at_zero_detector_distance(self) -> None:
        """The correct object-plane limit: ``W_obj^2 / (lambda z) = lambda z / dx_d^2 -> 0``.

        At the detector plane the same limit diverges, which is why the product editor
        used to need an 'inf' branch here.
        """
        assert _make_geometry(detector_distance_m=0.0).get_derived_values().fresnel_number == 0.0


class TestConeBeamGeometry:
    """A focusing optic sets the magnification, which scales the near-field geometric
    projection and the equivalent parallel-beam distance. The sign of the focus
    coordinate selects which side of the object the focus sits on. The optic does not
    select the propagation regime -- the product declares that independently.
    """

    # Focus 5 mm downstream of the object: the object sits in a converging beam that
    # crosses over before the detector. M = (1.0 - 0.005) / 0.005 = 199.
    _CONVERGING_FOCUS_M = 5e-3
    _CONVERGING_MAGNIFICATION = 199.0

    # Focus 5 mm upstream: the object sits in a diverging beam.
    # M = (1.0 + 0.005) / 0.005 = 201.
    _DIVERGING_FOCUS_M = -5e-3
    _DIVERGING_MAGNIFICATION = 201.0

    # The magnification itself is ProductMetadata.magnification, covered in
    # tests/test_product.py; these pin how the provider's geometry responds to it.

    def test_diverging_beam_propagation_distance(self) -> None:
        """The sign of the focus coordinate picks the other branch of ``M``."""
        geometry = _make_geometry(focus_object_distance_m=self._DIVERGING_FOCUS_M)
        expected_m = _DETECTOR_DISTANCE_M / self._DIVERGING_MAGNIFICATION

        derived = geometry.get_derived_values()

        assert derived.object_plane_propagation_distance_m == pytest.approx(expected_m)

    def test_object_plane_pixel_is_the_demagnified_detector_pixel(self) -> None:
        geometry = _make_geometry(focus_object_distance_m=self._CONVERGING_FOCUS_M, far_field=False)
        expected_m = _DETECTOR_PITCH_M / self._CONVERGING_MAGNIFICATION

        pixel_geometry = geometry.get_object_plane_pixel_geometry()

        assert pixel_geometry.width_m == pytest.approx(expected_m)
        assert pixel_geometry.height_m == pytest.approx(expected_m)

    def test_a_focusing_optic_does_not_imply_near_field_sampling(self) -> None:
        """Regression guard: the optic sets M, the product sets the regime.

        Before the regime was declared, a nonzero focus distance alone switched the
        sample plane to the geometric projection, which made far field with a
        focusing optic inexpressible.
        """
        without_optic = _make_geometry().get_object_plane_pixel_geometry()
        with_optic = _make_geometry(focus_object_distance_m=self._CONVERGING_FOCUS_M)

        assert with_optic.get_object_plane_pixel_geometry() == without_optic

    def test_propagation_distance_is_reduced_by_the_magnification(self) -> None:
        geometry = _make_geometry(focus_object_distance_m=self._CONVERGING_FOCUS_M)
        expected_m = _DETECTOR_DISTANCE_M / self._CONVERGING_MAGNIFICATION

        assert geometry.get_derived_values().object_plane_propagation_distance_m == pytest.approx(
            expected_m
        )

    def test_propagation_distance_is_the_detector_distance_without_an_optic(self) -> None:
        assert (
            _make_geometry().get_derived_values().object_plane_propagation_distance_m
            == _DETECTOR_DISTANCE_M
        )

    def test_fresnel_number_uses_the_equivalent_parallel_beam_distance(self) -> None:
        geometry = _make_geometry(focus_object_distance_m=self._CONVERGING_FOCUS_M, far_field=False)
        magnification = self._CONVERGING_MAGNIFICATION
        width_m = _NUM_PX * _DETECTOR_PITCH_M / magnification
        expected = width_m**2 / (_WAVELENGTH_M * _DETECTOR_DISTANCE_M / magnification)

        assert geometry.get_derived_values().fresnel_number == pytest.approx(expected)

    def test_fresnel_number_reports_near_field_for_a_magnifying_geometry(self) -> None:
        """The indicator must actually flip regime, not merely change value."""
        geometry = _make_geometry(focus_object_distance_m=self._CONVERGING_FOCUS_M, far_field=False)

        assert _make_geometry().get_derived_values().fresnel_number < 1.0
        assert geometry.get_derived_values().fresnel_number > 1.0

    def test_detector_at_the_focus_degrades_rather_than_dividing_by_zero(self) -> None:
        """M = 0 is degenerate, and the three fields degrade differently by design.

        The equivalent distance takes its true limit and diverges, while the pixel
        geometry and the Fresnel number are the two documented exceptions: the first is
        the zero pitch the provider reads as undetermined, the second a path limit that
        really is zero.
        """
        geometry = _make_geometry(focus_object_distance_m=_DETECTOR_DISTANCE_M, far_field=False)

        assert geometry.get_derived_values().object_plane_propagation_distance_m == math.inf
        assert geometry.get_derived_values().object_plane_pixel_geometry == PixelGeometry(0.0, 0.0)
        assert geometry.get_derived_values().fresnel_number == 0.0

        with pytest.raises(GeometryNotDefinedError):
            geometry.get_object_plane_pixel_geometry()


class TestNearFieldSampling:
    """A declared near-field product samples the object on the back-projected detector
    grid, including the parallel-beam case where the magnification is exactly one and
    the two grids coincide. That case has no cone to key off, which is why the regime
    is declared rather than inferred.
    """

    _FOCUS_M = 5e-3
    _MAGNIFICATION = 199.0

    def test_parallel_beam_object_pixel_is_the_detector_pixel(self) -> None:
        pixel_geometry = _make_geometry(far_field=False).get_object_plane_pixel_geometry()

        assert pixel_geometry.width_m == pytest.approx(_DETECTOR_PITCH_M)
        assert pixel_geometry.height_m == pytest.approx(_DETECTOR_PITCH_M)

    def test_parallel_beam_does_not_return_the_far_field_value(self) -> None:
        """The two differ by orders of magnitude here, so the branch is unambiguous."""
        near = _make_geometry(far_field=False).get_object_plane_pixel_geometry()
        far = _make_geometry(far_field=True).get_object_plane_pixel_geometry()

        assert near.width_m != pytest.approx(far.width_m)

    def test_cone_beam_object_pixel_is_demagnified(self) -> None:
        geometry = _make_geometry(focus_object_distance_m=self._FOCUS_M, far_field=False)

        pixel_geometry = geometry.get_object_plane_pixel_geometry()

        assert pixel_geometry.width_m == pytest.approx(_DETECTOR_PITCH_M / self._MAGNIFICATION)

    def test_probe_geometry_carries_the_near_field_pitch(self) -> None:
        geometry = _make_geometry(far_field=False)

        assert geometry.get_probe_geometry().pixel_width_m == pytest.approx(_DETECTOR_PITCH_M)

    def test_object_geometry_carries_the_near_field_pitch(self) -> None:
        geometry = _make_geometry(far_field=False)
        geometry._scan_item.get_probe_positions.return_value = []  # type: ignore[attr-defined]

        assert geometry.get_object_geometry().pixel_width_m == pytest.approx(_DETECTOR_PITCH_M)

    def test_parallel_beam_fresnel_number_reports_near_field(self) -> None:
        """Declaring near field on this geometry is self-consistent: 256 x 75 um at
        1 m is deeply near field, where the far-field declaration reads 0.022."""
        assert _make_geometry(far_field=True).get_derived_values().fresnel_number < 1.0
        assert _make_geometry(far_field=False).get_derived_values().fresnel_number > 1.0

    def test_flipping_the_regime_notifies_observers(self) -> None:
        """The probe and object rebuild off this notification, so the sampling change
        must propagate without a separate push."""
        geometry = _make_geometry()
        observer = MagicMock()
        geometry.add_observer(observer)

        geometry._metadata_item.far_field.set_value(False)

        observer._update.assert_called_once_with(geometry)
