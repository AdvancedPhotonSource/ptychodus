"""Forward models for simulating diffraction patterns from a ptychography data product."""

import numpy
from scipy.ndimage import map_coordinates

from ..affine import AffineTransform, transform_probe_positions
from ..diffraction import (
    BadPixels,
    DetectorBackground,
    DiffractionIndexes,
    DiffractionPatterns,
    FlatField,
    validate_pixel_map,
)
from ..geometry import ImageExtent, PixelGeometry
from ..object import Object, compute_object_geometry
from ..patch import iter_probe_patches
from ..assemble import AssembledDiffractionData
from ..probe import ProbeEnsemble
from ..probe_positions import ProbePositionSequence
from ..product import Product
from ..propagate import (
    AngularSpectrumPropagator,
    FresnelTransformPropagator,
    Propagator,
    PropagatorParameters,
    choose_propagator,
    compute_far_field_pixel_geometry,
)
from ..scattering import ScatteringGeometry
from ..typing import ComplexArrayType, RealArrayType

__all__ = ['generate_diffraction_data', 'generate_scattering_data']


def generate_diffraction_data(
    product: Product,
    rng: numpy.random.Generator | None = None,
    *,
    background: DetectorBackground | None = None,
    flat_field: FlatField | None = None,
) -> AssembledDiffractionData:
    """Simulate diffraction patterns for all scan positions in *product* using a multislice forward model.

    The propagation follows the product's declared regime. Far field uses the Fresnel
    transform onto the reciprocal grid ``lambda z / (N dx)``, ignoring any
    magnification. Near field propagates in the equivalent parallel-beam geometry --
    the Fresnel scaling theorem maps a cone of magnification ``M`` onto a parallel beam
    over ``z_d / M`` with transverse coordinates scaled by ``1 / M`` -- so the reported
    detector pitch is the equivalent-plane pitch scaled back up by ``M``.

    Two near-field caveats, both exact at ``M == 1``: the returned amplitudes are not
    ``1 / M**2``-normalized, so simulated counts scale with the magnified area, and
    ``layer_spacing_m`` is taken in the object frame unchanged rather than rescaled into
    the equivalent geometry.

    A detector is modeled when either map is given, both on the pattern grid: each
    intensity is multiplied by the `flat_field` response, then the `background` (in
    counts per frame) is added, before any noise. The result carries both maps
    unapplied, as measured data would.

    If *rng* is provided, Poisson noise is added to the intensity patterns.

    Raises:
        ValueError: when a near-field product places the detector at the focus, where
            the magnification and hence the equivalent geometry are undefined, or when
            a map does not match the pattern grid.
    """
    object_ = product.object_
    probe_geometry = product.probes.get_geometry()
    metadata = product.metadata

    if metadata.far_field:
        magnification = 1.0
        propagation_distance_m = metadata.detector_distance_m
    else:
        magnification = metadata.magnification

        if magnification == 0.0:
            raise ValueError(
                'Near-field propagation requires a nonzero magnification; the detector '
                'sits at the focus, where the equivalent parallel-beam geometry is '
                'undefined.'
            )

        propagation_distance_m = metadata.detector_distance_m / magnification

    propagator_parameters = PropagatorParameters(
        wavelength_m=metadata.photon_wavelength_m,
        width_px=probe_geometry.width_px,
        height_px=probe_geometry.height_px,
        pixel_width_m=probe_geometry.pixel_width_m,
        pixel_height_m=probe_geometry.pixel_height_m,
        propagation_distance_m=propagation_distance_m,
    )

    if metadata.far_field:
        # Pin the output grid rather than letting choose_propagator pick it: the
        # Fresnel transform already lands on the reciprocal grid, and a propagator
        # that preserved the source pitch instead would move the reported detector
        # plane without changing the array shape.
        propagator: Propagator = FresnelTransformPropagator(propagator_parameters)
        equivalent_pixel_geometry = compute_far_field_pixel_geometry(
            PixelGeometry(
                width_m=probe_geometry.pixel_width_m, height_m=probe_geometry.pixel_height_m
            ),
            ImageExtent(width_px=probe_geometry.width_px, height_px=probe_geometry.height_px),
            wavelength_m=metadata.photon_wavelength_m,
            propagation_distance_m=propagation_distance_m,
        )
    else:
        propagator, equivalent_pixel_geometry = choose_propagator(propagator_parameters)

    # One angular-spectrum propagator per inter-layer gap
    interlayer_propagators = [
        AngularSpectrumPropagator(
            PropagatorParameters(
                wavelength_m=metadata.photon_wavelength_m,
                width_px=probe_geometry.width_px,
                height_px=probe_geometry.height_px,
                pixel_width_m=probe_geometry.pixel_width_m,
                pixel_height_m=probe_geometry.pixel_height_m,
                propagation_distance_m=spacing_m,
            )
        )
        for spacing_m in object_.layer_spacing_m
    ]

    num_positions = len(product.probe_positions)
    indexes: DiffractionIndexes = numpy.zeros(num_positions, dtype=int)
    patterns: DiffractionPatterns = numpy.zeros(
        (num_positions, probe_geometry.height_px, probe_geometry.width_px),
        dtype=float,
    )
    # Scaling-theorem coordinate map back onto the true detector plane; the identity
    # for far field and for a parallel-beam near-field geometry.
    pixel_geometry = PixelGeometry(
        width_m=equivalent_pixel_geometry.width_m * magnification,
        height_m=equivalent_pixel_geometry.height_m * magnification,
    )
    bad_pixels: BadPixels = numpy.full((probe_geometry.height_px, probe_geometry.width_px), False)

    validate_pixel_map(background, bad_pixels.shape, name='Background')
    validate_pixel_map(flat_field, bad_pixels.shape, name='Flat field')

    for index, probe_patch in enumerate(iter_probe_patches(product)):
        bounds = probe_patch.bounds

        # Extract patches from all layers at the same integer position
        object_patches = [
            object_.get_layer(ilayer)[bounds.y_slice, bounds.x_slice]
            for ilayer in range(object_.num_layers)
        ]

        for wavefield in probe_patch.modes:
            # Multislice: apply each layer then propagate to the next; last layer has no propagation
            for object_patch, interlayer_propagator in zip(object_patches, interlayer_propagators):
                wavefield = wavefield * object_patch
                wavefield = interlayer_propagator.propagate(wavefield)
            wavefield = wavefield * object_patches[-1]

            wavefield = propagator.propagate(wavefield)
            patterns[index] += numpy.square(numpy.abs(wavefield))

        indexes[index] = probe_patch.position.index

    if flat_field is not None:
        numpy.multiply(patterns, flat_field, out=patterns)

    if background is not None:
        numpy.add(patterns, background, out=patterns)

    if rng is not None:
        # NOTE: object and probe scaling influence how much noise is added
        patterns = rng.poisson(patterns).astype(float)

    return AssembledDiffractionData(
        indexes,
        patterns,
        pixel_geometry,
        bad_pixels,
        background=background,
        flat_field=flat_field,
    )


def _solve_surface_points(
    geometry: ScatteringGeometry, u_m: RealArrayType, v_m: RealArrayType
) -> RealArrayType:
    """The surface point each exit-frame coordinate is the projection of, as ``(..., 3)``.

    A point ``r`` on the surface projects along the exit beam to ``(e1 · r, e2 · r)`` and
    satisfies ``n · r = 0``; the three equations are solved for ``r``.
    """
    column_axis, row_axis = geometry.get_detector_axes()
    system = numpy.vstack((column_axis, row_axis, geometry.get_surface_normal()))
    inverse = numpy.linalg.inv(system)
    return u_m[..., numpy.newaxis] * inverse[:, 0] + v_m[..., numpy.newaxis] * inverse[:, 1]


def _sample_complex(
    array: ComplexArrayType, row: RealArrayType, column: RealArrayType, *, order: int
) -> ComplexArrayType:
    """Interpolate a 2D complex array at fractional indexes, zero beyond its edge."""
    real = map_coordinates(array.real, (row, column), order=order, mode='grid-constant')
    imaginary = map_coordinates(array.imag, (row, column), order=order, mode='grid-constant')
    return real + 1j * imaginary


def generate_scattering_data(
    product: Product,
    geometry: ScatteringGeometry,
    rng: numpy.random.Generator | None = None,
    *,
    interpolation_order: int = 3,
) -> tuple[AssembledDiffractionData, Product]:
    """Simulate a far-field Bragg or Laue scan of a thin sample, kinematically.

    *product* states each part in its physical frame. The object is the sample surface,
    with its x along :meth:`ScatteringGeometry.get_surface_axes`' in-plane trace and y
    along y, at any rectangular pitch. The probe and the probe positions are across the
    incident beam, the probe on the detector's far-field pitch. Off the forward direction
    the detector sees the sample projected along the exit beam, so each part is carried
    into that exit frame and the patterns are those :func:`generate_diffraction_data`
    simulates there:

    * positions through :meth:`AffineTransform.from_scattering_geometry`;
    * object and probe by solving, for every exit-frame pixel, for the surface point it
      is the projection of, and sampling the surface object at that point's surface
      coordinates and the probe at its coordinates across the incident beam. This path
      does not go through the transform, so the two cross-check.

    Beyond the surface object's edge nothing scatters, as for a finite crystal. The
    resampled probe is rescaled so the first coherent mode keeps its ``sum(abs(P)**2)``:
    the incident photon count is conserved, and how efficiently the sample scatters them
    is the surface object's amplitude.

    Args:
        product: Surface object, incident probe and scan positions, with far-field
            metadata.
        geometry: Exit-beam and surface directions.
        rng: Adds Poisson noise to the patterns when supplied.
        interpolation_order: Spline order of the object and probe resampling. Lowering it
            is faster and blurs fine structure.

    Returns:
        The patterns, and the exit-frame product they were simulated from: the ground
        truth a reconstruction of them is compared with.

    Raises:
        ValueError: for a multislice object, which the thin-sample model excludes, for a
            near-field product, or for a geometry in which a beam grazes the surface.
    """
    if not product.metadata.far_field:
        raise ValueError('The scattering forward model is defined for far-field geometry only.')

    surface_object = product.object_

    if surface_object.num_layers != 1:
        raise ValueError(
            f'The scattering forward model takes a single-layer object; got '
            f'{surface_object.num_layers} layers.'
        )

    transform = AffineTransform.from_scattering_geometry(geometry)
    positions = ProbePositionSequence(
        list(transform_probe_positions(product.probe_positions, transform))
    )

    probes = product.probes
    probe_geometry = probes.get_geometry()
    probe_coordinates = probe_geometry.get_transverse_coordinates()
    probe_points = _solve_surface_points(geometry, probe_coordinates.x_m, probe_coordinates.y_m)
    probe_column = (
        probe_points[..., 0] / probe_geometry.pixel_width_m + (probe_geometry.width_px - 1) / 2
    )
    probe_row = (
        probe_points[..., 1] / probe_geometry.pixel_height_m + (probe_geometry.height_px - 1) / 2
    )
    probe_array = probes.get_array()
    exit_probe_array = numpy.empty_like(probe_array)

    for index in numpy.ndindex(probe_array.shape[:-2]):
        exit_probe_array[index] = _sample_complex(
            probe_array[index], probe_row, probe_column, order=interpolation_order
        )

    exit_power = numpy.sum(numpy.square(numpy.abs(exit_probe_array[0])))

    if exit_power > 0.0:
        incident_power = numpy.sum(numpy.square(numpy.abs(probe_array[0])))
        exit_probe_array *= numpy.sqrt(incident_power / exit_power)

    exit_geometry = compute_object_geometry(positions, probe_geometry)
    exit_coordinates = exit_geometry.get_transverse_coordinates()
    object_points = _solve_surface_points(
        geometry,
        exit_geometry.center_x_m + exit_coordinates.x_m,
        exit_geometry.center_y_m + exit_coordinates.y_m,
    )
    surface_axis_x, surface_axis_y = geometry.get_surface_axes()
    surface_geometry = surface_object.get_geometry()
    object_column = (
        object_points @ surface_axis_x - surface_geometry.center_x_m
    ) / surface_geometry.pixel_width_m + (surface_geometry.width_px - 1) / 2
    object_row = (
        object_points @ surface_axis_y - surface_geometry.center_y_m
    ) / surface_geometry.pixel_height_m + (surface_geometry.height_px - 1) / 2

    exit_product = Product(
        metadata=product.metadata,
        probe_positions=positions,
        probes=ProbeEnsemble(
            array=exit_probe_array,
            opr_weights=probes.get_opr_weights_or_none(),
            pixel_geometry=probes.get_pixel_geometry(),
        ),
        object_=Object(
            array=_sample_complex(
                surface_object.get_layer(0), object_row, object_column, order=interpolation_order
            ),
            pixel_geometry=exit_geometry.get_pixel_geometry(),
            center=exit_geometry.get_center(),
        ),
        losses=[],
    )
    return generate_diffraction_data(exit_product, rng), exit_product
