import numpy

from .diffraction import BadPixels, DiffractionIndexes, DiffractionPatterns
from .geometry import PixelGeometry
from .interpolate import BarycentricArrayInterpolator
from .io import AssembledDiffractionData
from .product import Product
from .propagator import FraunhoferPropagator, PropagatorParameters


def generate_diffraction_data(
    product: Product, rng: numpy.random.Generator | None = None
) -> AssembledDiffractionData:
    # TODO also support multislice
    interpolator = BarycentricArrayInterpolator(product.object_.get_layer(0))
    probe_geometry = product.probes.get_geometry()

    propagator_parameters = PropagatorParameters(
        wavelength_m=product.metadata.probe_wavelength_m,
        width_px=probe_geometry.width_px,
        height_px=probe_geometry.height_px,
        pixel_width_m=probe_geometry.pixel_width_m,
        pixel_height_m=probe_geometry.pixel_height_m,
        propagation_distance_m=product.metadata.detector_distance_m,
    )

    # TODO also support near-field propagation
    propagator = FraunhoferPropagator(propagator_parameters)

    num_positions = len(product.probe_positions)
    indexes: DiffractionIndexes = numpy.array(num_positions, dtype=int)
    patterns: DiffractionPatterns = numpy.zeros(
        (num_positions, probe_geometry.height_px, probe_geometry.width_px),
        dtype=float,
    )
    lambda_z_m2 = propagator_parameters.wavelength_m * propagator_parameters.propagation_distance_m
    pixel_geometry = PixelGeometry(
        width_m=lambda_z_m2 / probe_geometry.width_m,
        height_m=lambda_z_m2 / probe_geometry.height_m,
    )
    bad_pixels: BadPixels = numpy.full((probe_geometry.height_px, probe_geometry.width_px), False)

    for index, (probe_position, probe) in enumerate(zip(product.probe_positions, product.probes)):
        object_geometry = product.object_.get_geometry()
        object_position = object_geometry.map_coordinates_probe_to_object(probe_position)

        probe_geometry = product.probes.get_geometry()
        object_patch = interpolator.get_patch(
            object_position.coordinate_x_px,
            object_position.coordinate_y_px,
            probe_geometry.width_px,
            probe_geometry.height_px,
        )

        for imode in range(probe.num_incoherent_modes):
            exit_wave = probe.get_incoherent_mode(imode) * object_patch
            wavefield = propagator.propagate(exit_wave)
            patterns[index, :, :] += numpy.square(numpy.abs(wavefield))

        indexes[index] = object_position.index

    # FIXME rescale & add poisson noise

    return AssembledDiffractionData(indexes, patterns, pixel_geometry, bad_pixels)
