import numpy

from scipy.fft import fft2, fftfreq, ifft2

from .diffraction import BadPixels, DiffractionIndexes, DiffractionPatterns
from .geometry import PixelGeometry
from .io import AssembledDiffractionData
from .product import Product
from .propagator import AngularSpectrumPropagator, FraunhoferPropagator, PropagatorParameters


def generate_diffraction_data(
    product: Product, rng: numpy.random.Generator | None = None
) -> AssembledDiffractionData:
    object_ = product.object_
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

    # One angular-spectrum propagator per inter-layer gap
    interlayer_propagators = [
        AngularSpectrumPropagator(
            PropagatorParameters(
                wavelength_m=product.metadata.probe_wavelength_m,
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
    lambda_z_m2 = propagator_parameters.wavelength_m * propagator_parameters.propagation_distance_m
    pixel_geometry = PixelGeometry(
        width_m=lambda_z_m2 / probe_geometry.width_m,
        height_m=lambda_z_m2 / probe_geometry.height_m,
    )
    bad_pixels: BadPixels = numpy.full((probe_geometry.height_px, probe_geometry.width_px), False)

    # Precompute frequency grids for Fourier shifting the probe
    freq_y = fftfreq(probe_geometry.height_px)
    freq_x = fftfreq(probe_geometry.width_px)
    fy, fx = numpy.meshgrid(freq_y, freq_x, indexing='ij')

    object_geometry = object_.get_geometry()

    for index, (probe_position, probe) in enumerate(zip(product.probe_positions, product.probes)):
        object_position = object_geometry.map_coordinates_probe_to_object(probe_position)

        cx = object_position.coordinate_x_px
        cy = object_position.coordinate_y_px

        x_lower = int(cx - probe_geometry.width_px / 2)
        y_lower = int(cy - probe_geometry.height_px / 2)

        # Extract patches from all layers at the same integer position
        object_patches = [
            object_.get_layer(ilayer)[
                y_lower : y_lower + probe_geometry.height_px,
                x_lower : x_lower + probe_geometry.width_px,
            ]
            for ilayer in range(object_.num_layers)
        ]

        # Subpixel offsets between the true position and the integer extraction position
        dx = cx - (x_lower + probe_geometry.width_px / 2)
        dy = cy - (y_lower + probe_geometry.height_px / 2)

        # Shift all incoherent modes at once via a single batched FFT pair
        shift_phase = numpy.exp(-2j * numpy.pi * (fy * dy + fx * dx))
        shifted_modes = ifft2(fft2(probe.get_array()) * shift_phase)

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
