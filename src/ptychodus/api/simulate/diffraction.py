"""Forward model for simulating diffraction patterns from a ptychography data product."""

import numpy

from ..diffraction import BadPixels, DiffractionIndexes, DiffractionPatterns
from ..fourier import fourier_shift_2d
from ..geometry import ImageExtent, PixelGeometry
from ..assemble import AssembledDiffractionData
from ..product import Product
from ..propagate import (
    AngularSpectrumPropagator,
    FresnelTransformPropagator,
    Propagator,
    PropagatorParameters,
    choose_propagator,
    compute_far_field_pixel_geometry,
)


def generate_diffraction_data(
    product: Product, rng: numpy.random.Generator | None = None
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

    If *rng* is provided, Poisson noise is added to the intensity patterns.

    Raises:
        ValueError: when a near-field product places the detector at the focus, where
            the magnification and hence the equivalent geometry are undefined.
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

    object_geometry = object_.get_geometry()

    for index, (probe_position, probe) in enumerate(product.iter_position_probes()):
        object_position = object_geometry.map_coordinates_probe_to_object(probe_position)
        bounds = probe_geometry.resolve_patch_bounds(object_position.x_px, object_position.y_px)

        # Extract patches from all layers at the same integer position
        object_patches = [
            object_.get_layer(ilayer)[bounds.y_slice, bounds.x_slice]
            for ilayer in range(object_.num_layers)
        ]

        shifted_modes = fourier_shift_2d(probe.get_array(), dx=bounds.dx, dy=bounds.dy)

        for wavefield in shifted_modes:
            # Multislice: apply each layer then propagate to the next; last layer has no propagation
            for object_patch, interlayer_propagator in zip(object_patches, interlayer_propagators):
                wavefield = wavefield * object_patch
                wavefield = interlayer_propagator.propagate(wavefield)
            wavefield = wavefield * object_patches[-1]

            wavefield = propagator.propagate(wavefield)
            patterns[index] += numpy.square(numpy.abs(wavefield))

        indexes[index] = object_position.index

    if rng is not None:
        # NOTE: object and probe scaling influence how much noise is added
        patterns = rng.poisson(patterns).astype(float)

    return AssembledDiffractionData(indexes, patterns, pixel_geometry, bad_pixels)
