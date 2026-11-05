"""Forward models for simulating diffraction patterns from a ptychography data product."""

from dataclasses import replace

import numpy

from ..affine import transform_probe_positions
from ..diffraction import (
    BadPixels,
    DetectorBackground,
    DiffractionIndexes,
    DiffractionPatterns,
    FlatField,
    validate_pixel_map,
)
from ..geometry import ImageExtent, PixelGeometry
from ..interpolate import resample_complex_at
from ..object import Object, compute_object_geometry
from ..patch import iter_probe_patches
from ..assemble import AssembledDiffractionData
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
from ..scattering.frame import compute_exit_frame_transform, transform_probe_to_exit_frame
from ..scattering.geometry import ScatteringGeometry

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

    * positions through :func:`~ptychodus.api.scattering.frame.compute_exit_frame_transform`;
    * the probe through :func:`ptychodus.api.scattering.frame.transform_probe_to_exit_frame`;
    * the object by sampling the surface at the point each exit-frame pixel is the
      projection of (:meth:`ScatteringGeometry.project_onto_surface`).

    The last two do not go through the transform, so the two paths cross-check. Beyond
    the surface object's edge nothing scatters, as for a finite crystal. The incident
    photon count is conserved, and how efficiently the sample scatters them is the surface
    object's amplitude.

    The geometry is *geometry*, not the metadata's own ``scattering_geometry``, which is
    ignored: *product* is stated in the lab and surface frames, not the exit frame a
    product's geometry declares. The returned truth carries *geometry* in its metadata.

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

    transform = compute_exit_frame_transform(geometry)
    positions = ProbePositionSequence(
        list(transform_probe_positions(product.probe_positions, transform))
    )

    exit_probes = transform_probe_to_exit_frame(
        product.probes, geometry, far_field=True, interpolation_order=interpolation_order
    )
    probe_geometry = exit_probes.get_geometry()

    exit_geometry = compute_object_geometry(positions, probe_geometry)
    exit_coordinates = exit_geometry.get_transverse_coordinates()
    object_points = geometry.project_onto_surface(
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
        metadata=replace(product.metadata, scattering_geometry=geometry),
        probe_positions=positions,
        probes=exit_probes,
        object_=Object(
            array=resample_complex_at(
                surface_object.get_layer(0), object_row, object_column, order=interpolation_order
            ),
            pixel_geometry=exit_geometry.get_pixel_geometry(),
            center=exit_geometry.get_center(),
        ),
        losses=[],
    )
    return generate_diffraction_data(exit_product, rng), exit_product
