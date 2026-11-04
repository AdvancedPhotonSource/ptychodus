"""Bragg and Laue geometry: the exit-frame transform and the scattering forward model."""

from collections.abc import Callable
from dataclasses import replace
import math

import numpy
import pytest

from ptychodus.api.affine import AffineTransform
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, compute_object_geometry
from ptychodus.api.probe import ProbeEnsemble, ProbeGeometry
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.scattering import ScatteringGeometry
from ptychodus.api.simulate.diffraction import generate_diffraction_data, generate_scattering_data
from ptychodus.api.typing import ComplexArrayType, RealArrayType

_PHOTON_ENERGY_EV = 11200.0
_DETECTOR_DISTANCE_M = 0.89
_DETECTOR_PIXEL_SIZE_M = 55e-6
_DETECTOR_PX = 32

# An asymmetric Bragg reflection with the detector arm slightly below the horizontal plane.
_BRAGG_ASYMMETRIC = ScatteringGeometry(
    two_theta_deg=47.35, detector_gamma_deg=-0.27, incidence_angle_deg=24.15
)
# The detector arm off axis with the sample at normal incidence.
_LAUE = ScatteringGeometry(two_theta_deg=57.42, incidence_angle_deg=90.0)


def _matrix(transform: AffineTransform) -> RealArrayType:
    return numpy.array([[transform.a00, transform.a01], [transform.a10, transform.a11]])


def _probe_geometry() -> ProbeGeometry:
    return ProbeGeometry.from_far_field(
        PixelGeometry(width_m=_DETECTOR_PIXEL_SIZE_M, height_m=_DETECTOR_PIXEL_SIZE_M),
        ImageExtent(width_px=_DETECTOR_PX, height_px=_DETECTOR_PX),
        wavelength_m=energy_eV_to_wavelength_m(_PHOTON_ENERGY_EV),
        distance_m=_DETECTOR_DISTANCE_M,
    )


def _metadata(*, far_field: bool = True) -> ProductMetadata:
    return ProductMetadata(
        name='scattering',
        comments='',
        detector_distance_m=_DETECTOR_DISTANCE_M,
        photon_energy_eV=_PHOTON_ENERGY_EV,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
        far_field=far_field,
    )


def _incident_probe(x_m: RealArrayType, y_m: RealArrayType) -> ComplexArrayType:
    """An elliptical Gaussian with a weak tilt, complex so its pattern is not centrosymmetric."""
    pixel_m = _probe_geometry().pixel_width_m
    x_px = x_m / pixel_m
    y_px = y_m / pixel_m
    envelope = numpy.exp(-0.5 * (numpy.square(x_px / 2.5) + numpy.square(y_px / 2.0)))
    return envelope * numpy.exp(1j * (0.3 * x_px + 0.02 * numpy.square(y_px)))


def _surface_reflectivity(s1_m: RealArrayType, s2_m: RealArrayType) -> ComplexArrayType:
    """A smooth complex surface: amplitude contrast along s1, a displacement-like phase along s2."""
    pixel_m = _probe_geometry().pixel_width_m
    amplitude = 1.0 + 0.4 * numpy.cos(2.0 * numpy.pi * s1_m / (30.0 * pixel_m))
    phase = 0.8 * numpy.sin(2.0 * numpy.pi * s2_m / (14.0 * pixel_m) + 0.5)
    return amplitude * numpy.exp(1j * phase)


def _probes(function: Callable[[RealArrayType, RealArrayType], ComplexArrayType]) -> ProbeEnsemble:
    geometry = _probe_geometry()
    coordinates = geometry.get_transverse_coordinates()
    return ProbeEnsemble(
        array=function(coordinates.x_m, coordinates.y_m),
        opr_weights=None,
        pixel_geometry=geometry.get_pixel_geometry(),
    )


def _surface_object(
    function: Callable[[RealArrayType, RealArrayType], ComplexArrayType],
    *,
    width_px: int = 256,
    height_px: int = 96,
) -> Object:
    """*function* sampled on a surface grid at half the probe pitch, centered on the origin."""
    pixel_m = _probe_geometry().pixel_width_m / 2.0
    s2_px, s1_px = numpy.mgrid[:height_px, :width_px]
    s1_m = (s1_px - (width_px - 1) / 2) * pixel_m
    s2_m = (s2_px - (height_px - 1) / 2) * pixel_m
    return Object(
        array=function(s1_m, s2_m),
        pixel_geometry=PixelGeometry(width_m=pixel_m, height_m=pixel_m),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
    )


def _positions() -> ProbePositionSequence:
    pixel_m = _probe_geometry().pixel_width_m
    return ProbePositionSequence(
        [
            ProbePosition(index=0, x_m=0.0, y_m=0.0),
            ProbePosition(index=1, x_m=3.2 * pixel_m, y_m=-1.7 * pixel_m),
            ProbePosition(index=2, x_m=-2.6 * pixel_m, y_m=2.9 * pixel_m),
        ]
    )


def _lab_product(object_: Object, probes: ProbeEnsemble | None = None) -> Product:
    return Product(
        metadata=_metadata(),
        probe_positions=_positions(),
        probes=_probes(_incident_probe) if probes is None else probes,
        object_=object_,
        losses=[],
    )


# ---------------------------------------------------------------------------
# ScatteringGeometry and AffineTransform.from_scattering_geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('incidence_angle_deg', [90.0, 24.15, -60.0, 135.0])
def test_forward_exit_is_the_identity_whatever_the_incidence(incidence_angle_deg: float) -> None:
    geometry = ScatteringGeometry(incidence_angle_deg=incidence_angle_deg)
    transform = AffineTransform.from_scattering_geometry(geometry)

    assert geometry.is_forward
    assert not geometry.is_reflection
    numpy.testing.assert_allclose(_matrix(transform), numpy.eye(2), atol=1e-12)
    assert (transform.a02, transform.a12) == (0.0, 0.0)


@pytest.mark.parametrize(
    ('two_theta_deg', 'incidence_angle_deg'),
    [(47.35, 24.15), (47.35, 90.0), (57.42, 90.0), (20.0, 70.0), (30.0, 10.0), (-35.0, -80.0)],
)
def test_in_plane_geometry_scales_x_only(two_theta_deg: float, incidence_angle_deg: float) -> None:
    """With the arm in the x-z plane, ``x_D = x sin(θ_i - 2θ) / sin θ_i`` and y is untouched."""
    geometry = ScatteringGeometry(
        two_theta_deg=two_theta_deg, incidence_angle_deg=incidence_angle_deg
    )
    two_theta_rad = math.radians(two_theta_deg)
    incidence_rad = math.radians(incidence_angle_deg)
    expected = math.sin(incidence_rad - two_theta_rad) / math.sin(incidence_rad)

    numpy.testing.assert_allclose(
        _matrix(AffineTransform.from_scattering_geometry(geometry)),
        [[expected, 0.0], [0.0, 1.0]],
        atol=1e-12,
    )


@pytest.mark.parametrize('two_theta_deg', [10.0, 47.35, 90.0, 150.0])
def test_symmetric_bragg_is_a_mirror(two_theta_deg: float) -> None:
    geometry = ScatteringGeometry(
        two_theta_deg=two_theta_deg, incidence_angle_deg=two_theta_deg / 2.0
    )

    assert geometry.is_reflection
    numpy.testing.assert_allclose(
        _matrix(AffineTransform.from_scattering_geometry(geometry)),
        [[-1.0, 0.0], [0.0, 1.0]],
        atol=1e-12,
    )


@pytest.mark.parametrize('two_theta_deg', [10.0, 47.35, 57.42])
def test_normal_incidence_laue_foreshortens_by_cos_two_theta(two_theta_deg: float) -> None:
    geometry = ScatteringGeometry(two_theta_deg=two_theta_deg, incidence_angle_deg=90.0)

    assert not geometry.is_reflection
    numpy.testing.assert_allclose(
        _matrix(AffineTransform.from_scattering_geometry(geometry)),
        [[math.cos(math.radians(two_theta_deg)), 0.0], [0.0, 1.0]],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    'geometry',
    [
        _BRAGG_ASYMMETRIC,
        _LAUE,
        ScatteringGeometry(30.0, 4.0, 60.0),
        ScatteringGeometry(-20.0, -3.0, 50.0),
    ],
)
def test_out_of_plane_angle_adds_a_shear_of_y_by_x(geometry: ScatteringGeometry) -> None:
    two_theta_rad = math.radians(geometry.two_theta_deg)
    gamma_rad = math.radians(geometry.detector_gamma_deg)
    incidence_rad = math.radians(geometry.incidence_angle_deg)
    expected = [
        [math.sin(incidence_rad - two_theta_rad) / math.sin(incidence_rad), 0.0],
        [
            -math.sin(gamma_rad)
            * math.cos(two_theta_rad - incidence_rad)
            / math.sin(incidence_rad),
            math.cos(gamma_rad),
        ],
    ]

    numpy.testing.assert_allclose(
        _matrix(AffineTransform.from_scattering_geometry(geometry)), expected, atol=1e-12
    )


def test_asymmetric_bragg_mirrors_compresses_and_shears() -> None:
    numpy.testing.assert_allclose(
        _matrix(AffineTransform.from_scattering_geometry(_BRAGG_ASYMMETRIC)),
        [[-0.96288, 0.0], [0.01059, 0.99999]],
        atol=1e-5,
    )


def test_determinant_is_negative_exactly_for_reflections() -> None:
    num_reflections = 0

    for two_theta_deg in numpy.linspace(-170.0, 170.0, 18):
        for gamma_deg in (-10.0, 0.0, 7.0):
            for incidence_angle_deg in numpy.linspace(-175.0, 175.0, 15):
                geometry = ScatteringGeometry(
                    float(two_theta_deg), gamma_deg, float(incidence_angle_deg)
                )

                try:
                    transform = AffineTransform.from_scattering_geometry(geometry)
                except ValueError:
                    continue

                determinant = float(numpy.linalg.det(_matrix(transform)))
                assert (determinant < 0.0) == geometry.is_reflection, geometry
                num_reflections += geometry.is_reflection

    assert num_reflections > 0


@pytest.mark.parametrize(
    'geometry', [_BRAGG_ASYMMETRIC, _LAUE, ScatteringGeometry(30.0, 4.0, 60.0)]
)
def test_mirrored_instrument_conjugates_the_transform(geometry: ScatteringGeometry) -> None:
    """Negating 2θ and θ_i mirrors x on both sides: the diagonal stays, the shear flips."""
    mirrored = replace(
        geometry,
        two_theta_deg=-geometry.two_theta_deg,
        incidence_angle_deg=-geometry.incidence_angle_deg,
    )
    flip = numpy.diag([-1.0, 1.0])

    numpy.testing.assert_allclose(
        _matrix(AffineTransform.from_scattering_geometry(mirrored)),
        flip @ _matrix(AffineTransform.from_scattering_geometry(geometry)) @ flip,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    'geometry',
    [
        ScatteringGeometry(two_theta_deg=30.0, incidence_angle_deg=0.0),  # grazing incidence
        ScatteringGeometry(two_theta_deg=30.0, incidence_angle_deg=30.0),  # grazing exit
    ],
)
def test_a_grazing_beam_is_refused(geometry: ScatteringGeometry) -> None:
    with pytest.raises(ValueError, match='grazes'):
        AffineTransform.from_scattering_geometry(geometry)


def test_geometry_vectors_are_orthonormal() -> None:
    column_axis, row_axis = _BRAGG_ASYMMETRIC.get_detector_axes()
    exit_direction = _BRAGG_ASYMMETRIC.get_exit_direction()
    trace, y_axis = _BRAGG_ASYMMETRIC.get_surface_axes()
    normal = _BRAGG_ASYMMETRIC.get_surface_normal()

    numpy.testing.assert_allclose(
        numpy.vstack((column_axis, row_axis, exit_direction))
        @ numpy.vstack((column_axis, row_axis, exit_direction)).T,
        numpy.eye(3),
        atol=1e-12,
    )
    numpy.testing.assert_allclose(
        numpy.vstack((trace, y_axis, normal)) @ numpy.vstack((trace, y_axis, normal)).T,
        numpy.eye(3),
        atol=1e-12,
    )
    numpy.testing.assert_allclose(numpy.cross(column_axis, row_axis), exit_direction, atol=1e-12)


# ---------------------------------------------------------------------------
# generate_scattering_data
# ---------------------------------------------------------------------------


def test_forward_geometry_reduces_to_the_transmission_model() -> None:
    """At normal incidence and forward exit the surface is the object plane itself."""
    probe_geometry = _probe_geometry()
    object_geometry = compute_object_geometry(_positions(), probe_geometry)
    coordinates = object_geometry.get_transverse_coordinates()
    object_ = Object(
        array=_surface_reflectivity(
            object_geometry.center_x_m + coordinates.x_m,
            object_geometry.center_y_m + coordinates.y_m,
        ),
        pixel_geometry=object_geometry.get_pixel_geometry(),
        center=object_geometry.get_center(),
    )
    product = _lab_product(object_)

    data, exit_product = generate_scattering_data(product, ScatteringGeometry())
    expected = generate_diffraction_data(product)

    numpy.testing.assert_allclose(
        data.get_patterns(), expected.get_patterns(), atol=1e-10 * expected.get_patterns().max()
    )
    numpy.testing.assert_array_equal(data.get_indexes(), expected.get_indexes())
    assert exit_product.object_.get_geometry() == object_geometry


@pytest.mark.parametrize(
    'geometry', [_BRAGG_ASYMMETRIC, _LAUE, ScatteringGeometry(30.0, 4.0, 60.0)]
)
def test_a_surface_feature_lands_where_the_exit_beam_projects_it(
    geometry: ScatteringGeometry,
) -> None:
    pixel_m = _probe_geometry().pixel_width_m
    s1_m = 5.0 * pixel_m
    s2_m = -3.0 * pixel_m

    def bump(x_m: RealArrayType, y_m: RealArrayType) -> ComplexArrayType:
        radius2 = numpy.square(x_m - s1_m) + numpy.square(y_m - s2_m)
        return numpy.exp(-radius2 / (2.0 * numpy.square(2.0 * pixel_m))) + 0j

    _, exit_product = generate_scattering_data(_lab_product(_surface_object(bump)), geometry)
    exit_geometry = exit_product.object_.get_geometry()
    coordinates = exit_geometry.get_transverse_coordinates()
    weights = numpy.square(numpy.abs(exit_product.object_.get_layer(0)))
    centroid = (
        float(numpy.sum(weights * (exit_geometry.center_x_m + coordinates.x_m)) / weights.sum()),
        float(numpy.sum(weights * (exit_geometry.center_y_m + coordinates.y_m)) / weights.sum()),
    )

    trace, y_axis = geometry.get_surface_axes()
    column_axis, row_axis = geometry.get_detector_axes()
    surface_point = s1_m * trace + s2_m * y_axis
    expected = (float(column_axis @ surface_point), float(row_axis @ surface_point))

    numpy.testing.assert_allclose(centroid, expected, atol=0.02 * pixel_m)


@pytest.mark.parametrize('geometry', [_BRAGG_ASYMMETRIC, _LAUE])
def test_the_exit_probe_keeps_its_photons_and_takes_the_transformed_shape(
    geometry: ScatteringGeometry,
) -> None:
    lab_product = _lab_product(_surface_object(_surface_reflectivity))
    _, exit_product = generate_scattering_data(lab_product, geometry)

    def second_moments(probes: ProbeEnsemble) -> RealArrayType:
        coordinates = probes.get_geometry().get_transverse_coordinates()
        weights = numpy.square(numpy.abs(probes.get_array()[0, 0]))
        points = numpy.stack((coordinates.x_m, coordinates.y_m))
        mean = numpy.sum(weights * points, axis=(1, 2)) / weights.sum()
        centered = points - mean[:, numpy.newaxis, numpy.newaxis]
        return numpy.einsum('ihw,jhw,hw->ij', centered, centered, weights) / weights.sum()

    transform = _matrix(AffineTransform.from_scattering_geometry(geometry))
    lab_power = numpy.sum(numpy.square(numpy.abs(lab_product.probes.get_array())))
    exit_power = numpy.sum(numpy.square(numpy.abs(exit_product.probes.get_array())))

    assert exit_power == pytest.approx(lab_power, rel=1e-12)
    expected = transform @ second_moments(lab_product.probes) @ transform.T

    # Sampling a Gaussian one or two pixels wide moves its moments by parts in 1e4.
    numpy.testing.assert_allclose(
        second_moments(exit_product.probes), expected, atol=1e-3 * numpy.abs(expected).max()
    )


def _kinematic_patterns(product: Product, geometry: ScatteringGeometry) -> RealArrayType:
    """Patterns summed directly over the surface, with exact path lengths to each pixel.

    No projection, no Fourier transform and no far-field approximation: every surface point
    scatters the incident probe with phase ``k (z + |R - r|)`` to a pixel at ``R`` on the
    rotated detector. The surface object is the envelope of a lattice term ``exp(i G · r)``
    with ``G = k (k_f - z)``, which meets the Bragg condition at the detector center; the
    lattice term is put back here. This is the independent check on the projection model.
    """
    wavelength_m = product.metadata.photon_wavelength_m
    wavenumber = 2.0 * numpy.pi / wavelength_m
    distance_m = product.metadata.detector_distance_m
    column_axis, row_axis = geometry.get_detector_axes()
    exit_direction = geometry.get_exit_direction()

    offsets_px = numpy.arange(_DETECTOR_PX) - _DETECTOR_PX // 2  # fftshift's zero frequency
    row_px, column_px = numpy.meshgrid(offsets_px, offsets_px, indexing='ij')
    pixels_m = (
        distance_m * exit_direction
        + (column_px * _DETECTOR_PIXEL_SIZE_M)[..., numpy.newaxis] * column_axis
        + (row_px * _DETECTOR_PIXEL_SIZE_M)[..., numpy.newaxis] * row_axis
    ).reshape(-1, 3)
    pixel_distances_m = numpy.linalg.norm(pixels_m, axis=1)

    surface = product.object_
    surface_geometry = surface.get_geometry()
    coordinates = surface_geometry.get_transverse_coordinates()
    s1_m = (surface_geometry.center_x_m + coordinates.x_m).ravel()
    s2_m = (surface_geometry.center_y_m + coordinates.y_m).ravel()
    trace, y_axis = geometry.get_surface_axes()
    points_m = s1_m[:, numpy.newaxis] * trace + s2_m[:, numpy.newaxis] * y_axis
    lattice_phase = wavenumber * points_m @ (exit_direction - numpy.array([0.0, 0.0, 1.0]))
    reflectivity = surface.get_layer(0).ravel() * numpy.exp(1j * lattice_phase)

    patterns = []

    for position in product.probe_positions:
        illumination = _incident_probe(points_m[:, 0] - position.x_m, points_m[:, 1] - position.y_m)
        weights = reflectivity * illumination
        lit = numpy.abs(illumination) > 1e-6
        lit_points_m = points_m[lit]
        # |R - r| - |R| without cancellation, so the phase keeps its precision.
        path_m = (
            numpy.sum(numpy.square(lit_points_m), axis=1)[:, numpy.newaxis]
            - 2.0 * lit_points_m @ pixels_m.T
        ) / (
            numpy.linalg.norm(pixels_m[numpy.newaxis] - lit_points_m[:, numpy.newaxis], axis=2)
            + pixel_distances_m
        )
        phase = wavenumber * (lit_points_m[:, 2:3] + path_m)
        amplitude = weights[lit] @ numpy.exp(1j * phase)
        patterns.append(numpy.square(numpy.abs(amplitude)).reshape(_DETECTOR_PX, _DETECTOR_PX))

    return numpy.stack(patterns)


@pytest.mark.parametrize('geometry', [_BRAGG_ASYMMETRIC, _LAUE])
def test_patterns_match_a_direct_kinematic_sum(geometry: ScatteringGeometry) -> None:
    product = _lab_product(_surface_object(_surface_reflectivity))
    data, _ = generate_scattering_data(product, geometry)
    simulated = data.get_patterns()
    expected = _kinematic_patterns(product, geometry)

    for index in range(len(product.probe_positions)):
        simulated_pattern = simulated[index] / simulated[index].sum()
        expected_pattern = expected[index] / expected[index].sum()
        error = numpy.linalg.norm(simulated_pattern - expected_pattern)
        # What remains is interpolation and the depth the projection model drops.
        assert error / numpy.linalg.norm(expected_pattern) < 5e-3, index


def test_a_multislice_object_is_refused() -> None:
    surface = _surface_object(_surface_reflectivity)
    layered = Object(
        array=numpy.stack([surface.get_layer(0)] * 2),
        pixel_geometry=surface.get_pixel_geometry(),
        center=surface.get_center(),
        layer_spacing_m=[1e-6],
    )

    with pytest.raises(ValueError, match='single-layer'):
        generate_scattering_data(_lab_product(layered), _BRAGG_ASYMMETRIC)


def test_a_near_field_product_is_refused() -> None:
    product = replace(
        _lab_product(_surface_object(_surface_reflectivity)), metadata=_metadata(far_field=False)
    )

    with pytest.raises(ValueError, match='far-field'):
        generate_scattering_data(product, _BRAGG_ASYMMETRIC)


def test_a_grazing_geometry_is_refused_by_the_generator() -> None:
    with pytest.raises(ValueError, match='grazes'):
        generate_scattering_data(
            _lab_product(_surface_object(_surface_reflectivity)),
            ScatteringGeometry(two_theta_deg=30.0, incidence_angle_deg=30.0),
        )
