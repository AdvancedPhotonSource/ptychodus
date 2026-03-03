from collections.abc import Iterable, Sequence
import logging

from scipy.fft import fftfreq, ifft2
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy

from ._phase_unwrapper import PhaseUnwrapper
from .common import RealArrayType, lerp
from .object import Object, ObjectGeometry
from .probe_positions import ProbePosition
from .reconstructor import AssembledDiffractionData

logger = logging.getLogger(__name__)


def generate_stxm_object(
    geometry: ObjectGeometry,
    assembled_data: AssembledDiffractionData,
    probe_positions: Iterable[ProbePosition],
) -> Object:
    coordinates_px: list[float] = list()
    values: list[float] = list()

    assembled_indexes = assembled_data.get_indexes().tolist()
    assembled_pattern_counts = assembled_data.get_pattern_counts().tolist()
    pattern_counts_lut = dict(zip(assembled_indexes, assembled_pattern_counts))

    for scan_point in probe_positions:
        try:
            value = pattern_counts_lut[scan_point.index]
        except KeyError:
            logger.debug(f'Skipping missing scan point index={scan_point.index}!')
        else:
            object_point = geometry.map_coordinates_probe_to_object(scan_point)
            coordinates_px.append(object_point.coordinate_y_px)
            coordinates_px.append(object_point.coordinate_x_px)
            values.append(value)

    points = numpy.reshape(coordinates_px, (-1, 2))
    YY, XX = numpy.mgrid[: geometry.height_px, : geometry.width_px]  # noqa: N806
    query_points = numpy.transpose((YY.flat, XX.flat))
    intensity = griddata(points, values, query_points, method='linear', fill_value=0.0).reshape(
        XX.shape
    )

    return Object(
        array=numpy.sqrt(intensity[numpy.newaxis, :, :]).astype('complex'),
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def generate_random_object(
    rng: numpy.random.Generator,
    geometry: ObjectGeometry,
    *,
    amplitude_mean: float,
    amplitude_deviation: float,
    phase_mean: float,
    phase_deviation_tr: float,
    blur_deviation_px: float,
) -> Object:
    object_shape = (1, geometry.height_px, geometry.width_px)

    amplitude = rng.normal(
        amplitude_mean,
        amplitude_deviation,
        object_shape,
    )
    phase = rng.normal(
        0.0,
        phase_deviation_tr,
        object_shape,
    )

    if blur_deviation_px > 0.0:
        # TODO account for pixel aspect ratio in blur calculation
        amplitude = gaussian_filter(amplitude, sigma=blur_deviation_px)
        phase = gaussian_filter(phase, sigma=blur_deviation_px)

    array = amplitude * numpy.exp(2j * numpy.pi * phase)

    return Object(
        array=array.astype('complex'),
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def generate_gaussian_random_field_object(
    rng: numpy.random.Generator, geometry: ObjectGeometry, *, correlation_length_px: float
) -> Object:
    """Generate a complex-valued Gaussian random field via spectral synthesis."""
    # white noise in Fourier space
    object_shape = (1, geometry.height_px, geometry.width_px)
    noise = rng.normal(size=object_shape) + 1j * rng.normal(size=object_shape)

    # frequency grid
    kx = fftfreq(geometry.width_px)
    ky = fftfreq(geometry.height_px)
    KX, KY = numpy.meshgrid(kx, ky)  # noqa: N806
    K2 = numpy.square(KX) + numpy.square(KY)  # noqa: N806

    # power spectrum: Gaussian envelope to control correlation length
    power_spectrum = numpy.exp(-0.5 * (2 * numpy.pi * correlation_length_px) ** 2 * K2)
    power_spectrum[0, 0] = 0  # zero mean

    # shape noise with power spectrum and back-transform
    # FIXME fftshift?
    field = ifft2(noise * numpy.sqrt(power_spectrum))

    return Object(
        array=field,
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def generate_dead_leaves_object(
    rng: numpy.random.Generator,
    geometry: ObjectGeometry,
    *,
    leaf_radius_lower_px: float,
    leaf_radius_upper_px: float,
    leaf_radius_power_law_exponent: float,
    leaf_amplitude_lower: float,
    leaf_amplitude_upper: float,
    leaf_phase_lower_tr: float,
    leaf_phase_upper_tr: float,
) -> Object:
    # TODO consider using Poisson disk sampling or CVT for sample positions
    # TODO include pixel aspect ratio in sample position calculation
    _sample_positions = [
        (1.0 / 3, 1.0 / 3),
        (1.0 / 3, 2.0 / 3),
        (2.0 / 3, 1.0 / 3),
        (2.0 / 3, 2.0 / 3),
    ]

    if leaf_radius_lower_px >= leaf_radius_upper_px:
        raise ValueError('leaf_radius_lower_px must be less than leaf_radius_upper_px')

    if leaf_amplitude_lower >= leaf_amplitude_upper:
        raise ValueError('leaf_amplitude_lower must be less than leaf_amplitude_upper')

    if leaf_phase_lower_tr >= leaf_phase_upper_tr:
        raise ValueError('leaf_phase_lower_tr must be less than leaf_phase_upper_tr')

    object_shape = (geometry.height_px, geometry.width_px)
    is_covered = numpy.zeros(object_shape, dtype=bool)
    amplitude: RealArrayType = numpy.zeros(object_shape, dtype=float)
    phase_tr: RealArrayType = numpy.zeros(object_shape, dtype=float)

    position_y_px, position_x_px = numpy.indices(object_shape)
    num_covered_pixels = 0

    beta = 1.0 - leaf_radius_power_law_exponent
    coef = numpy.power(leaf_radius_upper_px / leaf_radius_lower_px, beta) - 1.0

    for leaf in range(100000):  # large value eliminates possiblity of infinite loop
        leaf_radius_px = leaf_radius_lower_px * numpy.power(1.0 + rng.uniform() * coef, 1.0 / beta)
        leaf_position_x_px = rng.uniform(-leaf_radius_px, geometry.width_px + leaf_radius_px)
        leaf_position_y_px = rng.uniform(-leaf_radius_px, geometry.height_px + leaf_radius_px)
        leaf_phase_tr = rng.uniform(
            leaf_phase_lower_tr,
            leaf_phase_upper_tr,
            size=(1, 1),
        )
        leaf_amplitude = numpy.sqrt(
            rng.uniform(
                leaf_amplitude_lower**2,
                leaf_amplitude_upper**2,
                size=(1, 1),
            )
        )

        leaf_counts = numpy.zeros_like(is_covered, dtype=int)
        dx = position_x_px - leaf_position_x_px
        dy = position_y_px - leaf_position_y_px

        # multi-sample anti-aliasing
        for u_y, u_x in _sample_positions:
            sample_distance_px = numpy.hypot(dy + u_y, dx + u_x)
            leaf_counts[sample_distance_px <= leaf_radius_px] += 1

        num_samples = len(_sample_positions)
        leaf_coverage = leaf_counts / num_samples
        amplitude = lerp(amplitude, leaf_amplitude, leaf_coverage)
        phase_tr = lerp(phase_tr, leaf_phase_tr, leaf_coverage)

        is_covered |= leaf_counts == num_samples
        num_covered_pixels = numpy.count_nonzero(is_covered).item()

        covered_pct = 100 * num_covered_pixels / is_covered.size
        logger.info(f'leaves = {leaf}, covered = {covered_pct:.2f}%')

        if num_covered_pixels == is_covered.size:
            break

    array = amplitude * numpy.exp(2j * numpy.pi * phase_tr)

    return Object(
        array=array.astype('complex'),
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def generate_layers(object_: Object, layer_spacing_m: Sequence[float]) -> Object:
    """Create an object from an existing object with a potentially
    different number of slices.

    If the new object is supposed to be a multislice object with a
    different number of slices than the existing object, the object is
    created as
    `abs(o) ** (1 / nSlices) * exp(i * unwrapPhase(o) / nSlices)`.
    Otherwise, the object is copied as is.
    """
    num_slices = 1 + len(layer_spacing_m)
    array = object_.get_array()

    if num_slices < array.shape[0]:
        array = array[:num_slices]
    elif num_slices > array.shape[0]:
        amplitude = numpy.absolute(array[:1]) ** (1.0 / num_slices)
        amplitude = amplitude.repeat(num_slices, axis=0)
        phase = PhaseUnwrapper().unwrap(array[0])[numpy.newaxis, ...] / num_slices
        phase = phase.repeat(num_slices, axis=0)
        array = amplitude * numpy.exp(1j * phase)

    return Object(
        array=array,
        pixel_geometry=object_.get_pixel_geometry(),
        center=object_.get_center(),
        layer_spacing_m=layer_spacing_m,
    )


def pad_object(object_: Object, pad_x: int, pad_y: int) -> Object:
    pad_width = [(0, 0), (pad_y, pad_y), (pad_x, pad_x)]

    return Object(
        array=numpy.pad(object_.get_array(), pad_width),
        pixel_geometry=object_.get_pixel_geometry(),
        center=object_.get_center(),
        layer_spacing_m=object_.layer_spacing_m,
    )
