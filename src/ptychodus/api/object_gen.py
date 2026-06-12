"""Object (transmission function) generation functions: STXM, random fields, simplex/fractal noise, and dead-leaves models."""

from collections.abc import Iterable, Sequence
from typing import Final
import logging

from scipy.fft import fft2, fftfreq, ifft2
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy

from ._phase_unwrapper import PhaseUnwrapper
from .common import IntegerArrayType, RealArrayType, lerp
from .object import Object, ObjectGeometry
from .probe_positions import ProbePosition
from .reconstructor import AssembledDiffractionData

logger = logging.getLogger(__name__)


def generate_stxm_object(
    geometry: ObjectGeometry,
    assembled_data: AssembledDiffractionData,
    probe_positions: Iterable[ProbePosition],
) -> Object:
    """Generate an STXM-like object by interpolating per-position diffraction counts onto the object grid."""
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


def generate_paganin_object(
    geometry: ObjectGeometry,
    assembled_data: AssembledDiffractionData,
    probe_positions: Iterable[ProbePosition],
    *,
    probe_wavelength_m: float,
    propagation_distance_m: float,
    delta_over_beta: float,
    small_value: float = 1.0e-12,
) -> Object:
    """Generate a complex object via Paganin single-material phase retrieval on an STXM-like intensity image.

    See D. Paganin et al., J. Microsc. 206, 33 (2002). The intensity image
    I(x,y) is built from per-position diffraction-pattern counts, normalized
    by its mean to approximate I/I_0, then low-pass filtered to recover the
    projected thickness T(x,y) of a homogeneous, weakly-absorbing sample with
    refractive-index decrement delta and extinction coefficient beta. The
    transmission function follows as ``filtered ** ((1 + i * delta/beta) / 2)``.
    """
    if propagation_distance_m <= 0.0:
        raise ValueError('Propagation distance must be strictly positive!')

    if delta_over_beta <= 0.0:
        raise ValueError('delta/beta ratio must be strictly positive!')

    stxm = generate_stxm_object(geometry, assembled_data, probe_positions)
    intensity = numpy.square(numpy.abs(stxm.get_array()[0]))

    mean_intensity = float(numpy.mean(intensity))

    if mean_intensity <= 0.0:
        raise ValueError('Mean STXM intensity must be positive!')

    intensity_normalized = intensity / mean_intensity

    pixel_geometry = geometry.get_pixel_geometry()
    kx = 2 * numpy.pi * fftfreq(geometry.width_px, d=pixel_geometry.width_m)
    ky = 2 * numpy.pi * fftfreq(geometry.height_px, d=pixel_geometry.height_m)
    KY, KX = numpy.meshgrid(ky, kx, indexing='ij')  # noqa: N806
    K2 = numpy.square(KX) + numpy.square(KY)  # noqa: N806

    filter_denominator = (
        1.0 + delta_over_beta * propagation_distance_m * probe_wavelength_m * K2 / (4 * numpy.pi)
    )
    filtered = numpy.real(numpy.asarray(ifft2(fft2(intensity_normalized) / filter_denominator)))
    filtered = numpy.clip(filtered, small_value, None)

    exponent = 0.5 * (1.0 + 1j * delta_over_beta)
    array = numpy.power(filtered.astype(complex), exponent)[numpy.newaxis, :, :]

    return Object(
        array=array.astype('complex'),
        pixel_geometry=pixel_geometry,
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
    """Generate a random complex object from Gaussian amplitude and phase distributions, with optional Gaussian blur."""
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
    KY, KX = numpy.meshgrid(ky, kx, indexing='ij')  # noqa: N806
    K2 = numpy.square(KX) + numpy.square(KY)  # noqa: N806

    # power spectrum: Gaussian envelope to control correlation length
    power_spectrum = numpy.exp(-0.5 * (2 * numpy.pi * correlation_length_px) ** 2 * K2)
    power_spectrum[0, 0] = 0  # zero mean

    # shape noise with power spectrum and back-transform
    field = numpy.asarray(ifft2(noise * numpy.sqrt(power_spectrum)))

    return Object(
        array=field,
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def _map_simplex_to_cartesian(
    xx: RealArrayType, yy: RealArrayType, grid_scale_px: float
) -> tuple[RealArrayType, RealArrayType]:
    SQRT3: Final[float] = numpy.sqrt(3)  # noqa: N806
    SQRT6: Final[float] = numpy.sqrt(6)  # noqa: N806

    c = 1 / (grid_scale_px * SQRT6)
    zs = xx + yy
    zd = xx - yy

    ii = c * (+zs + SQRT3 * zd)
    jj = c * (-zs + SQRT3 * zd)

    return ii, jj


def _map_cartesian_to_simplex(
    ii: RealArrayType, jj: RealArrayType, grid_scale_px: float
) -> tuple[RealArrayType, RealArrayType]:
    SQRT3: Final[float] = numpy.sqrt(3)  # noqa: N806
    SQRT8: Final[float] = numpy.sqrt(8)  # noqa: N806

    c = grid_scale_px / SQRT8
    ks = ii + jj
    kd = ii - jj

    xx = c * (ks + SQRT3 * kd)
    yy = c * (-ks + SQRT3 * kd)

    return xx, yy


def _calculate_vertex_noise_contribution(
    xx: RealArrayType,
    yy: RealArrayType,
    vertex_i: IntegerArrayType,
    vertex_j: IntegerArrayType,
    vertex_grad_x: RealArrayType,
    vertex_grad_y: RealArrayType,
    grid_scale_px: float,
) -> RealArrayType:
    vertex_x, vertex_y = _map_cartesian_to_simplex(
        vertex_i.astype(float), vertex_j.astype(float), grid_scale_px
    )
    displacement_x = (xx - vertex_x) / grid_scale_px
    displacement_y = (yy - vertex_y) / grid_scale_px
    distancesq_normalized = numpy.square(displacement_x) + numpy.square(displacement_y)
    kernel = numpy.maximum(0.0, 0.5 - distancesq_normalized) ** 4
    grad_dir_x = vertex_grad_x[vertex_j, vertex_i] * displacement_x
    grad_dir_y = vertex_grad_y[vertex_j, vertex_i] * displacement_y
    return kernel * (grad_dir_x + grad_dir_y)


def _generate_simplex_noise(
    rng: numpy.random.Generator,
    width_px: int,
    height_px: int,
    grid_scale_px: float,
) -> RealArrayType:
    # generate coordinate grid
    yy, xx = numpy.mgrid[:height_px, :width_px]

    # generate random direction vectors
    geometry_coefficient = (1 + numpy.sqrt(3)) / (grid_scale_px * numpy.sqrt(6))
    grad_shape_x = numpy.ceil(geometry_coefficient * width_px).astype(int) + 1
    grad_shape_y = numpy.ceil(geometry_coefficient * height_px).astype(int) + 1
    grad_shape = grad_shape_y, grad_shape_x
    logger.debug(f'{geometry_coefficient=} {grad_shape=}')
    angle_rad = 2 * numpy.pi * rng.uniform(size=grad_shape)
    vertex_grad_x = numpy.cos(angle_rad)
    vertex_grad_y = numpy.sin(angle_rad)

    # locate containing cell
    ii, jj = _map_simplex_to_cartesian(xx, yy, grid_scale_px)
    cell_origin_i = numpy.floor(ii).astype(int)
    cell_origin_j = numpy.floor(jj).astype(int)
    cell_origin_x, cell_origin_y = _map_cartesian_to_simplex(
        cell_origin_i.astype(float), cell_origin_j.astype(float), grid_scale_px
    )

    # vertex indexes
    is_lower_triangle = (ii - cell_origin_i) > (jj - cell_origin_j)
    vertex0_i = cell_origin_i
    vertex0_j = cell_origin_j
    vertex1_i = cell_origin_i + numpy.where(is_lower_triangle, 1, 0)
    vertex1_j = cell_origin_j + numpy.where(is_lower_triangle, 0, 1)
    vertex2_i = cell_origin_i + 1
    vertex2_j = cell_origin_j + 1

    # vertex noise contributions
    noise0 = _calculate_vertex_noise_contribution(
        xx, yy, vertex0_i, vertex0_j, vertex_grad_x, vertex_grad_y, grid_scale_px
    )
    noise1 = _calculate_vertex_noise_contribution(
        xx, yy, vertex1_i, vertex1_j, vertex_grad_x, vertex_grad_y, grid_scale_px
    )
    noise2 = _calculate_vertex_noise_contribution(
        xx, yy, vertex2_i, vertex2_j, vertex_grad_x, vertex_grad_y, grid_scale_px
    )

    # accumulate vertex contributions to noise
    noise = noise0 + noise1 + noise2

    noise_max = noise.max()
    noise_min = noise.min()
    return 2 * (noise - noise_min) / (noise_max - noise_min) - 1


def generate_simplex_noise_object(
    rng: numpy.random.Generator,
    geometry: ObjectGeometry,
    *,
    grid_scale_px: float = 30.0,
) -> Object:
    """Generate a complex object from independent simplex-noise realizations for real and imaginary parts."""
    pixel_geometry = geometry.get_pixel_geometry()

    if not pixel_geometry.is_square:
        raise ValueError('Non-square pixels are unsupported!')

    if grid_scale_px <= 0.0:
        raise ValueError('Grid scale must be strictly positive!')

    re = _generate_simplex_noise(
        rng=rng,
        width_px=geometry.width_px,
        height_px=geometry.height_px,
        grid_scale_px=grid_scale_px,
    )
    im = _generate_simplex_noise(
        rng=rng,
        width_px=geometry.width_px,
        height_px=geometry.height_px,
        grid_scale_px=grid_scale_px,
    )
    array = re + 1j * im

    return Object(
        array=array,
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def generate_fractal_noise_object(
    rng: numpy.random.Generator,
    geometry: ObjectGeometry,
    *,
    grid_scale_px: float,
    num_octaves: int = 1,
    gain: float = 0.5,
    lacunarity: float = 2.0,
) -> Object:
    """Generate a complex fractal-noise object by summing multiple octaves of simplex noise."""
    object_shape = (1, geometry.height_px, geometry.width_px)
    array = numpy.zeros(object_shape, dtype=complex)
    amplitude = 0.5

    for octave in range(num_octaves):
        noise = generate_simplex_noise_object(rng, geometry, grid_scale_px=grid_scale_px)
        array += amplitude * noise.get_array()
        amplitude *= gain
        grid_scale_px /= lacunarity

    return Object(
        array=array,
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
    """Generate a complex object by layering randomly-sized and positioned disks (dead-leaves model)."""
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

    The new slice count is ``1 + len(layer_spacing_m)``. If it is greater
    than the existing slice count, the first layer is split as
    ``abs(o) ** (1 / nSlices) * exp(i * unwrapPhase(o) / nSlices)`` and
    repeated. If it is less, the existing layers are truncated to the new
    count. If equal, the array is reused as is.
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
        array=array.astype('complex'),
        pixel_geometry=object_.get_pixel_geometry(),
        center=object_.get_center(),
        layer_spacing_m=layer_spacing_m,
    )


def pad_object(object_: Object, pad_x: int, pad_y: int) -> Object:
    """Return a zero-padded copy of *object_* with *pad_x* / *pad_y* pixels added on each side."""
    pad_width = [(0, 0), (pad_y, pad_y), (pad_x, pad_x)]

    return Object(
        array=numpy.pad(object_.get_array(), pad_width),
        pixel_geometry=object_.get_pixel_geometry(),
        center=object_.get_center(),
        layer_spacing_m=object_.layer_spacing_m,
    )
