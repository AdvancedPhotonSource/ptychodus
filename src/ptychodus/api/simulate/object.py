"""Object (transmission function) generation functions: STXM differential phase contrast, Paganin phase retrieval, random fields, simplex/fractal noise, dead-leaves models, and Siemens star resolution targets."""

from __future__ import annotations
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final
import logging
import math

from scipy.fft import fft2, fftfreq, ifft2
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import numpy

from ..accumulate import DiffractionMoments, compute_diffraction_moments
from ..assemble import AssembledDiffractionData
from ..constants import TWO_PI_J
from ..geometry import ImageExtent, PixelGeometry
from ..interpolate import interpolate_at_indexes, lerp
from ..object import Object, ObjectGeometry, resample_object_layers
from ..patch import accumulate_probe_intensity_on, iter_probe_patches_on
from ..probe import Probe, ProbeEnsemble
from ..probe_positions import ProbePosition, ProbePositionSequence
from ..typing import BooleanArrayType, IntegerArrayType, RealArrayType

__all__ = [
    'STXMDPCSignals',
    'compute_stxm_dpc_signals',
    'generate_dead_leaves_object',
    'generate_fractal_noise_object',
    'generate_gaussian_random_field_object',
    'generate_layers',
    'generate_paganin_object',
    'generate_random_object',
    'generate_siemens_star_object',
    'generate_simplex_noise_object',
    'generate_stxm_dpc_object',
    'generate_uniform_object',
    'interpolate_scan_values',
    'pad_object',
]

logger = logging.getLogger(__name__)


def interpolate_scan_values(
    geometry: ObjectGeometry,
    positions: ProbePositionSequence,
    values: RealArrayType,
    *,
    probes: ProbeEnsemble | None = None,
    support_fraction: float = 0.05,
) -> tuple[RealArrayType, BooleanArrayType]:
    """Map per-position values onto the object grid, with the region the scan supports.

    *values* holds one row of ``k`` finite values per position, shape ``(n,)`` or
    ``(n, k)``; the grids come back ``(k, height, width)``. Linear interpolation holds
    inside the convex hull of the positions, and beyond it every pixel takes its nearest
    position's value, so the border continues the measured edge rather than stepping to
    a level nothing observed.

    Interpolation is the right map for a value measured under a probe: such a value is
    already the probe-averaged field sampled at its position, so interpolating the
    samples recovers that once-blurred field. Weighting the map by the probe as well
    would blur it a second time.

    The returned support is where the scan measured anything. Without *probes* it is
    the hull. With *probes*, placed as the forward model places them and with their
    incoherent modes summed, it is where the illumination reaches *support_fraction* of
    its peak, which follows the probe's footprint past the outermost positions and into
    the gaps of a sparse scan rather than the outline of the positions alone.

    Raises:
        ValueError: if *support_fraction* is outside ``(0, 1]``, the probe illuminates
            nothing on the canvas, or a probe's footprint leaves the canvas.
    """
    rows = numpy.reshape(values, (len(positions), -1)).astype(numpy.float64)
    points_yx_px = numpy.column_stack(
        (
            geometry.map_probe_positions_to_object_y_px(positions),
            geometry.map_probe_positions_to_object_x_px(positions),
        )
    )
    YY, XX = numpy.mgrid[: geometry.height_px, : geometry.width_px]  # noqa: N806
    query_points = numpy.transpose((YY.flat, XX.flat))

    # griddata fills with NaN outside the hull unless told otherwise, which is what
    # distinguishes the two regions here.
    interpolated = griddata(points_yx_px, rows, query_points, method='linear')
    inside_hull = numpy.logical_not(numpy.isnan(interpolated[:, 0]))

    if not inside_hull.all():
        nearest = griddata(points_yx_px, rows, query_points, method='nearest')
        interpolated = numpy.where(inside_hull[:, numpy.newaxis], interpolated, nearest)

    grids = numpy.moveaxis(interpolated.reshape(*XX.shape, rows.shape[1]), -1, 0)

    if probes is None:
        return grids, inside_hull.reshape(XX.shape)

    if not 0.0 < support_fraction <= 1.0:
        raise ValueError('Support fraction must lie in (0, 1]!')

    illumination = accumulate_probe_intensity_on(probes, positions, geometry)
    peak = float(illumination.max())

    if not peak > 0.0:
        raise ValueError('The probe illuminates nothing on the object canvas!')

    return grids, illumination >= support_fraction * peak


@dataclass(frozen=True)
class STXMDPCSignals:
    """Per-pattern scanning signals: the transmitted total and the intensity centroid.

    Rows pair with probe positions by `indexes`, never by order. Centroids are in
    detector pixels, x along columns and y along rows, about the detector midpoint. A
    pattern with no signal keeps its total and has a NaN centroid.
    """

    indexes: IntegerArrayType
    total_counts: RealArrayType
    center_of_mass_x_px: RealArrayType
    center_of_mass_y_px: RealArrayType

    @classmethod
    def from_moments(cls, moments: DiffractionMoments) -> STXMDPCSignals:
        """The per-pattern rows of accumulated diffraction moments."""
        return cls(
            indexes=moments.indexes.copy(),
            total_counts=moments.total_counts.copy(),
            center_of_mass_x_px=moments.center_of_mass_x_px.copy(),
            center_of_mass_y_px=moments.center_of_mass_y_px.copy(),
        )


def compute_stxm_dpc_signals(
    assembled_data: AssembledDiffractionData,
    *,
    max_patterns_per_chunk: int = 256,
    monitor_counts_by_index: Mapping[int, float] | None = None,
) -> STXMDPCSignals:
    """Measure each pattern's transmitted total and its intensity centroid.

    One pass of :func:`ptychodus.api.accumulate.compute_diffraction_moments`, so the
    same signals accumulate online, a chunk at a time, from a stream. A detector
    background or flat field the patterns still carry is applied first, as ``(pattern -
    background) / flat_field`` clipped at zero, and the pixels those maps make unusable
    join the bad pixels.

    Bad pixels are inpainted in each frame from that frame's own neighbours, so a shifted
    disk edge next to a hole carries into the fill. A bad region touching the detector
    border is zeroed instead and so acts as dropped, and a wide gap across the edge of
    the bright field is poorly constrained either way.

    The total sums the whole inpainted frame; it is the absorption signal. Summing only
    the good pixels instead would let a beam that walks across a dead region with the
    sample's phase gradient read as absorption. When measured incident photon counts are
    attached, each total is scaled by ``mean(M) / M_j`` so beam decay does not read as
    absorption; a count at or below zero cannot normalize anything and leaves its total
    unscaled. *monitor_counts_by_index* supplies those counts keyed by scan index -- as
    :func:`ptychodus.api.assemble.compute_probe_photon_counts_by_index` merges them from
    both sides with its fallback off -- in place of the assembled data's own.

    The centroid is also taken over the whole inpainted frame. The centre-of-mass
    theorem then holds up to whatever scatters off the detector, so the crop sets both
    how much deflected signal is lost and how much high-angle shot noise enters: the
    centroid's noise variance grows as ``sum(I_p * r_p**2)``, dominated by sparse pixels
    far from the bright field. For photon-counting detectors, whose dark pixels hold
    mostly zeros after the value filter, that noise is small; data from an integrating
    detector with read noise should be cropped tightly, as the crop suggestion does.

    `max_patterns_per_chunk` bounds the working memory and sets how many frames share
    one biharmonic fill. It changes the result by rounding alone, since every per-pattern
    reduction runs within one pattern.

    Raises:
        ValueError: if a tunable is out of range.
    """
    if max_patterns_per_chunk < 1:
        raise ValueError('Chunks must hold at least one pattern!')

    indexes = numpy.asarray(assembled_data.get_indexes())
    moments = compute_diffraction_moments(
        indexes,
        assembled_data.get_patterns(),
        bad_pixels=assembled_data.get_bad_pixels(),
        background=assembled_data.get_background(),
        flat_field=assembled_data.get_flat_field(),
        chunk_size=max_patterns_per_chunk,
    )

    if moments is None:
        empty = numpy.zeros(0, dtype=numpy.float64)
        return STXMDPCSignals(
            indexes=indexes.copy(),
            total_counts=empty,
            center_of_mass_x_px=empty.copy(),
            center_of_mass_y_px=empty.copy(),
        )

    if monitor_counts_by_index:
        moments = moments.normalize_by_monitor(
            numpy.fromiter(monitor_counts_by_index.keys(), dtype=numpy.intp),
            numpy.fromiter(monitor_counts_by_index.values(), dtype=numpy.float64),
        )
    elif assembled_data.has_measured_probe_photon_counts():
        moments = moments.normalize_by_monitor(
            indexes, assembled_data.get_probe_photon_counts().astype(numpy.float64)
        )

    return STXMDPCSignals.from_moments(moments)


def _pair_signals_with_positions(
    signals: STXMDPCSignals, probe_positions: Iterable[ProbePosition]
) -> tuple[BooleanArrayType, ProbePositionSequence]:
    """The signals a position pairs with, and those positions, in signal order.

    Signals pair with positions by index: duplicate positions average, a signal between
    positions takes an interpolated one, and a signal outside the range of position
    indexes is left out.

    Raises:
        ValueError: when no position pairs with any signal.
    """
    positions = list(probe_positions)
    position_indexes = numpy.array([position.index for position in positions], dtype=int)
    position_xy_m = numpy.array(
        [(position.x_m, position.y_m) for position in positions], dtype=float
    ).reshape(-1, 2)
    paired, xy_m = interpolate_at_indexes(position_indexes, position_xy_m, signals.indexes)
    num_skipped = int(numpy.count_nonzero(~paired))

    if num_skipped > 0:
        logger.debug(f'Skipping {num_skipped} STXM-DPC signals outside the position indexes.')

    if not paired.any():
        raise ValueError('No probe position pairs with any STXM-DPC signal index!')

    paired_positions = ProbePositionSequence(
        [
            ProbePosition(index=int(index), x_m=float(x_m), y_m=float(y_m))
            for index, (x_m, y_m) in zip(signals.indexes[paired], xy_m)
        ]
    )
    return paired, paired_positions


def _compute_probe_shadow_gradient(
    transmission: RealArrayType,
    probes: ProbeEnsemble,
    positions: ProbePositionSequence,
    geometry: ObjectGeometry,
) -> RealArrayType:
    """The probe-phase gradient each position's centroid sees, in rad/m, as ``(n, 2)`` (x, y).

    A centroid measures the ``|psi|**2``-weighted mean of the exit-wave phase gradient,
    which includes the probe's own. Under transmission ``T`` that share is
    ``sum(T * J) / sum(T * I)`` over the probe's footprint at that position, where
    ``I = sum_m |P_m|**2`` and ``J = sum_m Im(conj(P_m) grad P_m)`` --
    ``|P|**2 grad(phase P)`` without unwrapping -- summed over the incoherent modes as
    the centre-of-mass theorem requires of a mixed state.

    Each footprint is placed as the forward model places it, an exact-width slice of
    the canvas with the probe Fourier-shifted by the residual sub-pixel offset.
    Derivatives are spectral, which a far-field probe supports because it is periodic on
    its FFT grid by construction.
    """
    probe_geometry = probes.get_geometry()
    pixel_geometry = probes.get_pixel_geometry()
    kx = 2 * numpy.pi * fftfreq(probe_geometry.width_px, d=pixel_geometry.width_m)
    ky = 2 * numpy.pi * fftfreq(probe_geometry.height_px, d=pixel_geometry.height_m)
    shadow = numpy.zeros((len(positions), 2))

    for index, patch in enumerate(iter_probe_patches_on(probes, positions, geometry)):
        modes = patch.modes
        spectrum = fft2(modes)
        d_dx = numpy.asarray(ifft2(1j * kx[numpy.newaxis, numpy.newaxis, :] * spectrum))
        d_dy = numpy.asarray(ifft2(1j * ky[numpy.newaxis, :, numpy.newaxis] * spectrum))
        conjugate = numpy.conj(modes)
        local = transmission[patch.bounds.y_slice, patch.bounds.x_slice]
        weight = numpy.sum(local * numpy.sum(numpy.square(numpy.abs(modes)), axis=0))

        if weight > 0.0:
            shadow[index, 0] = numpy.sum(local * numpy.sum(numpy.imag(conjugate * d_dx), axis=0))
            shadow[index, 1] = numpy.sum(local * numpy.sum(numpy.imag(conjugate * d_dy), axis=0))
            shadow[index] /= weight

    return shadow


def _integrate_gradient(
    gradient_x: RealArrayType,
    gradient_y: RealArrayType,
    pixel_geometry: PixelGeometry,
    *,
    regularization_per_m2: float,
) -> RealArrayType:
    """Least-squares phase from its gradient by an FFT Poisson solve.

    The canvas is mirrored to twice its size, negating each component in the halves
    mirrored along it, which makes the phase even and periodic so the FFT introduces no
    seam. A positive regularization adds a Tikhonov term to ``|k|**2`` that damps
    angular frequencies below ``sqrt(regularization_per_m2)``.
    """
    height_px, width_px = gradient_x.shape
    extended_x = numpy.block(
        [
            [gradient_x, -gradient_x[:, ::-1]],
            [gradient_x[::-1, :], -gradient_x[::-1, ::-1]],
        ]
    )
    extended_y = numpy.block(
        [
            [gradient_y, gradient_y[:, ::-1]],
            [-gradient_y[::-1, :], -gradient_y[::-1, ::-1]],
        ]
    )
    kx = 2 * numpy.pi * fftfreq(2 * width_px, d=pixel_geometry.width_m)
    ky = 2 * numpy.pi * fftfreq(2 * height_px, d=pixel_geometry.height_m)
    KY, KX = numpy.meshgrid(ky, kx, indexing='ij')  # noqa: N806
    denominator = numpy.square(KX) + numpy.square(KY) + regularization_per_m2
    denominator[0, 0] = 1.0
    phase_spectrum = -1j * (KX * fft2(extended_x) + KY * fft2(extended_y)) / denominator
    phase_spectrum[0, 0] = 0.0
    phase = numpy.real(numpy.asarray(ifft2(phase_spectrum)))
    return phase[:height_px, :width_px]


def generate_stxm_dpc_object(
    geometry: ObjectGeometry,
    signals: STXMDPCSignals,
    probe_positions: Iterable[ProbePosition],
    *,
    detector_extent: ImageExtent,
    probe: Probe | None = None,
    correct_shadow: bool = True,
    incident_counts: float | None = None,
    regularization_fov_fraction: float = 0.0,
    reference_quantile: float = 0.95,
) -> Object:
    """Generate a coarse complex object from far-field data by STXM differential phase contrast.

    The amplitude is the STXM one: per-position totals mapped onto the object grid. With
    *incident_counts*, the unattenuated incident count, they are divided by it, so the
    modulus is the sample's amplitude transmission. Without it they are divided by their
    mean over the supported region, a dimensionless contrast map that a position
    recording the unobstructed beam brings to unit modulus once a caller scales it to the
    sample's mean opacity.

    The totals and the gradients are mapped by :func:`interpolate_scan_values`, linearly
    between the positions. A given `probe`, whose incoherent modes are summed, sets the
    supported region -- where the gradient is integrated and the transmission is
    referenced -- to its illumination rather than the hull of the positions.

    The phase comes from the far-field centre-of-mass theorem: a pattern's intensity
    centroid is the ``|psi|**2``-weighted mean of the exit wave's phase gradient under
    the probe. The data must be far field, and its centroids must already be in the
    frame :func:`ptychodus.api.simulate.diffraction.generate_diffraction_data`
    simulates, with x along columns and y along rows; a detector rotated or flipped
    relative to the scan has to be oriented beforehand. Each centroid has the mean
    centroid subtracted, which removes the beam centre's offset and, with it, any net
    phase wedge across the sample.

    A shift of ``u`` detector pixels is the gradient ``2 * pi * u / (N * dx)`` in rad/m,
    where ``N`` is the `detector_extent` along that axis and ``dx`` the object pitch:
    one detector pixel is one frequency step of the ``N``-pixel window the patterns
    were recorded on. That holds when the object grid has the far-field pitch paired
    with the patterns, ``wavelength * z_d / (N * dx_detector)``, which is the grid the
    forward model reconstructs on. The conversion then needs neither the distance nor
    the wavelength, and a miscalibration of either cannot rescale the phase apart from
    the grid it lands on.

    Signals pair with positions by index: duplicate positions average, a signal between
    positions takes an interpolated one, and a signal outside the range of position
    indexes is skipped. The gradient is interpolated onto the grid, zeroed beyond the
    scan, and integrated by a least-squares FFT Poisson solve. The phase is referenced
    to zero mean over the scanned pixels whose transmission reaches the
    `reference_quantile`: the empty beam when the scan includes one, and otherwise the
    most transparent region.

    Integrating gradient noise piles its error up at low spatial frequency, as drift
    across the field of view. A positive `regularization_fov_fraction` adds a Tikhonov
    term that damps phase variations with wavelengths longer than that fraction of the
    canvas's larger dimension, halving them at exactly that wavelength. It has little
    effect beyond about 2, the extent of the mirrored canvas the solve runs on.

    Resolution is the probe's and the scan step's, because each centroid averages the
    gradient under ``|P|**2``. The theorem needs all of the scattered intensity, so
    gradients that deflect signal beyond the detector frame are underestimated.

    A centroid also carries the probe's own phase gradient, weighted by the sample's
    transmission. Without a `probe`, that shadow term is ignored, which assumes a probe
    of flat phase or a sample whose absorption contrast is weak where the probe's phase
    varies. For a single material of decrement ratio ``beta / delta`` under a probe of
    RMS intensity width ``sigma`` defocused by ``dz``, ignoring it rescales the phase by
    ``1 +/- 4 * pi * (beta / delta) * sigma**2 / (wavelength * |dz|)``: of order one for
    micron-scale defocused hard X-ray probes at moderate ``beta / delta``, and nothing
    for a pure phase object. Given a `probe` and *correct_shadow*, that term is
    estimated against the STXM transmission and subtracted. The estimate is first order,
    because that transmission is itself blurred by the probe.

    Raises:
        ValueError: for a detector extent that is not positive, a negative
            regularization, a reference quantile outside [0, 1], a probe whose pixels
            differ from the object's, an incident count that is not positive, when no
            probe position pairs with any signal, or when the mean STXM intensity over
            the supported region is not positive.
    """
    if detector_extent.width_px <= 0 or detector_extent.height_px <= 0:
        raise ValueError(f'Detector extent must be positive, not {detector_extent}!')

    if regularization_fov_fraction < 0.0:
        raise ValueError('Regularization must not be negative!')

    if not 0.0 <= reference_quantile <= 1.0:
        raise ValueError('Reference quantile must lie in [0, 1]!')

    pixel_geometry = geometry.get_pixel_geometry()

    if probe is not None:
        probe_pixel_geometry = probe.get_pixel_geometry()

        if not (
            math.isclose(probe_pixel_geometry.width_m, pixel_geometry.width_m, rel_tol=1e-6)
            and math.isclose(probe_pixel_geometry.height_m, pixel_geometry.height_m, rel_tol=1e-6)
        ):
            raise ValueError(
                f'Probe pixels ({probe_pixel_geometry}) differ from object pixels'
                f' ({pixel_geometry})!'
            )

    if incident_counts is not None and not incident_counts > 0.0:
        raise ValueError(f'Incident counts must be positive, not {incident_counts}!')

    paired, positions = _pair_signals_with_positions(signals, probe_positions)
    probes = None if probe is None else ProbeEnsemble.from_probe(probe)

    total_counts = signals.total_counts[paired]
    centroid_x_px = signals.center_of_mass_x_px[paired]
    centroid_y_px = signals.center_of_mass_y_px[paired]
    finite = numpy.isfinite(centroid_x_px) & numpy.isfinite(centroid_y_px)
    deviation_x_px = numpy.zeros_like(centroid_x_px)
    deviation_y_px = numpy.zeros_like(centroid_y_px)

    if finite.any():
        deviation_x_px[finite] = centroid_x_px[finite] - numpy.mean(centroid_x_px[finite])
        deviation_y_px[finite] = centroid_y_px[finite] - numpy.mean(centroid_y_px[finite])

    radians_per_px_x = 2 * numpy.pi / (detector_extent.width_px * geometry.pixel_width_m)
    radians_per_px_y = 2 * numpy.pi / (detector_extent.height_px * geometry.pixel_height_m)
    grids, scanned = interpolate_scan_values(
        geometry,
        positions,
        numpy.column_stack(
            (
                total_counts,
                radians_per_px_x * deviation_x_px,
                radians_per_px_y * deviation_y_px,
            )
        ),
        probes=probes,
    )
    intensity, gradient_x, gradient_y = grids

    if incident_counts is None:
        incident_counts = float(numpy.mean(intensity[scanned])) if scanned.any() else 0.0

        if incident_counts <= 0.0:
            raise ValueError('Mean STXM intensity must be positive!')

    transmission = intensity / incident_counts

    if probes is not None and correct_shadow:
        shadow = _compute_probe_shadow_gradient(transmission, probes, positions, geometry)
        shadow -= numpy.mean(shadow, axis=0)
        shadow_grids, _ = interpolate_scan_values(geometry, positions, shadow, probes=probes)
        gradient_x = gradient_x - shadow_grids[0]
        gradient_y = gradient_y - shadow_grids[1]

    if regularization_fov_fraction > 0.0:
        cutoff_wavelength_m = regularization_fov_fraction * max(geometry.width_m, geometry.height_m)
        regularization_per_m2 = (2 * numpy.pi / cutoff_wavelength_m) ** 2
    else:
        regularization_per_m2 = 0.0

    phase = _integrate_gradient(
        numpy.where(scanned, gradient_x, 0.0),
        numpy.where(scanned, gradient_y, 0.0),
        pixel_geometry,
        regularization_per_m2=regularization_per_m2,
    )

    if scanned.any():
        threshold = numpy.quantile(transmission[scanned], reference_quantile)
        reference = scanned & (transmission >= threshold)
        phase -= numpy.mean(phase[reference if reference.any() else scanned])

    array = numpy.sqrt(transmission) * numpy.exp(1j * phase)

    return Object(
        array=array[numpy.newaxis, :, :].astype('complex'),
        pixel_geometry=pixel_geometry,
        center=geometry.get_center(),
    )


def generate_paganin_object(
    geometry: ObjectGeometry,
    signals: STXMDPCSignals,
    probe_positions: Iterable[ProbePosition],
    *,
    probe: Probe | None = None,
    photon_wavelength_m: float,
    propagation_distance_m: float,
    delta_over_beta: float,
    small_value: float = 1.0e-12,
) -> Object:
    """Generate a complex object via Paganin single-material phase retrieval on an STXM-like intensity image.

    See D. Paganin et al., J. Microsc. 206, 33 (2002). The intensity image
    I(x,y) is the per-position totals of `signals`, mapped onto the object grid as
    :func:`generate_stxm_dpc_object` maps them, and normalized by its mean over the
    supported region -- the illumination of a given `probe`, or the hull of the
    positions -- to approximate I/I_0. It is then low-pass filtered to recover the projected thickness
    T(x,y) of a homogeneous, weakly-absorbing sample with refractive-index decrement
    delta and extinction coefficient beta. The transmission function follows as
    ``filtered ** ((1 + i * delta/beta) / 2)``.

    The mean is taken over the supported region alone. Pixels beyond it are
    extrapolated from the edge of the scan rather than measured, so including
    them would set I_0 partly from values nothing observed.
    """
    if propagation_distance_m <= 0.0:
        raise ValueError('Propagation distance must be strictly positive!')

    if delta_over_beta <= 0.0:
        raise ValueError('delta/beta ratio must be strictly positive!')

    paired, positions = _pair_signals_with_positions(signals, probe_positions)
    grids, scanned = interpolate_scan_values(
        geometry,
        positions,
        signals.total_counts[paired],
        probes=None if probe is None else ProbeEnsemble.from_probe(probe),
    )
    intensity = grids[0]
    mean_intensity = float(numpy.mean(intensity[scanned])) if scanned.any() else 0.0

    if mean_intensity <= 0.0:
        raise ValueError('Mean STXM intensity must be positive!')

    intensity_normalized = intensity / mean_intensity

    pixel_geometry = geometry.get_pixel_geometry()
    kx = 2 * numpy.pi * fftfreq(geometry.width_px, d=pixel_geometry.width_m)
    ky = 2 * numpy.pi * fftfreq(geometry.height_px, d=pixel_geometry.height_m)
    KY, KX = numpy.meshgrid(ky, kx, indexing='ij')  # noqa: N806
    K2 = numpy.square(KX) + numpy.square(KY)  # noqa: N806

    filter_denominator = (
        1.0 + delta_over_beta * propagation_distance_m * photon_wavelength_m * K2 / (4 * numpy.pi)
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


def generate_uniform_object(geometry: ObjectGeometry, *, amplitude: float = 1.0) -> Object:
    """Generate a featureless object of uniform *amplitude* and zero phase.

    The starting guess for a reconstruction that asserts nothing about the structure of
    the sample, as distinct from :func:`generate_random_object` with its deviations set
    to zero -- which computes the same array, but reads as though the result were random.

    The default is a transparent sample. An amplitude below one asserts uniform
    absorption instead, which is what a measured mean transmission states: the exit wave
    then predicts the pattern totals the detector recorded rather than the unobstructed
    beam.
    """
    array = numpy.full((1, geometry.height_px, geometry.width_px), amplitude, dtype=complex)

    return Object(
        array=array,
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def generate_random_object(
    rng: numpy.random.Generator,
    geometry: ObjectGeometry,
    *,
    amplitude_mean: float,
    amplitude_deviation: float,
    phase_mean_tr: float,
    phase_deviation_tr: float,
    blur_deviation_px: float,
) -> Object:
    """Generate a random complex object from Gaussian amplitude and phase distributions, with optional Gaussian blur.

    Phase is in turns, as the ``_tr`` suffixes say: the array is
    ``amplitude * exp(2*pi*j * phase)``, so a `phase_mean_tr` of 0.5 is half a turn.
    """
    object_shape = (1, geometry.height_px, geometry.width_px)

    amplitude = rng.normal(
        amplitude_mean,
        amplitude_deviation,
        object_shape,
    )
    phase = rng.normal(
        phase_mean_tr,
        phase_deviation_tr,
        object_shape,
    )

    if blur_deviation_px > 0.0:
        # TODO account for pixel aspect ratio in blur calculation
        amplitude = gaussian_filter(amplitude, sigma=blur_deviation_px)
        phase = gaussian_filter(phase, sigma=blur_deviation_px)

    array = amplitude * numpy.exp(TWO_PI_J * phase)

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
        num_covered_pixels = int(numpy.count_nonzero(is_covered))

        covered_pct = 100 * num_covered_pixels / is_covered.size
        logger.info(f'leaves = {leaf}, covered = {covered_pct:.2f}%')

        if num_covered_pixels == is_covered.size:
            break

    array = amplitude * numpy.exp(TWO_PI_J * phase_tr)

    return Object(
        array=array.astype('complex'),
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def generate_siemens_star_object(
    geometry: ObjectGeometry,
    *,
    num_spokes: int,
    outer_radius_fraction: float,
    spoke_amplitude: float,
    background_amplitude: float,
    spoke_phase_tr: float,
    background_phase_tr: float,
) -> Object:
    """Generate a Siemens star resolution target with alternating angular wedges.

    ``2 * num_spokes`` wedges alternate between the spoke and background
    values inside a disk of radius
    ``outer_radius_fraction * min(width_px, height_px) / 2`` centered on the
    canvas; pixels outside the disk take the background values. Amplitude and
    phase are averaged over a 4-point sub-pixel pattern to soften wedge edges
    and the singular center.
    """
    if num_spokes < 2:
        raise ValueError('num_spokes must be at least 2!')

    if outer_radius_fraction <= 0.0:
        raise ValueError('outer_radius_fraction must be strictly positive!')

    _sample_positions = [
        (1.0 / 3, 1.0 / 3),
        (1.0 / 3, 2.0 / 3),
        (2.0 / 3, 1.0 / 3),
        (2.0 / 3, 2.0 / 3),
    ]

    height_px = geometry.height_px
    width_px = geometry.width_px
    outer_radius_px = outer_radius_fraction * min(width_px, height_px) / 2.0
    center_y_px = height_px / 2.0
    center_x_px = width_px / 2.0

    position_y_px, position_x_px = numpy.indices((height_px, width_px)).astype(float)
    amplitude = numpy.zeros((height_px, width_px), dtype=float)
    phase_tr = numpy.zeros((height_px, width_px), dtype=float)

    for u_y, u_x in _sample_positions:
        dy = position_y_px + u_y - center_y_px
        dx = position_x_px + u_x - center_x_px
        radius_px = numpy.hypot(dy, dx)
        angle = numpy.mod(numpy.arctan2(dy, dx), 2.0 * numpy.pi)
        wedge_index = numpy.floor(angle * num_spokes / numpy.pi).astype(int)
        is_spoke = (radius_px <= outer_radius_px) & (wedge_index % 2 == 0)
        amplitude += numpy.where(is_spoke, spoke_amplitude, background_amplitude)
        phase_tr += numpy.where(is_spoke, spoke_phase_tr, background_phase_tr)

    num_samples = len(_sample_positions)
    amplitude /= num_samples
    phase_tr /= num_samples

    array = amplitude * numpy.exp(TWO_PI_J * phase_tr)

    return Object(
        array=array[numpy.newaxis, :, :].astype('complex'),
        pixel_geometry=geometry.get_pixel_geometry(),
        center=geometry.get_center(),
    )


def generate_layers(object_: Object, layer_spacing_m: Sequence[float]) -> Object:
    """Reslice an object onto the depth grid given by *layer_spacing_m*.

    The new slice count is ``1 + len(layer_spacing_m)``. Every layer of the input
    contributes: :func:`resample_object_layers` interpolates the cumulative optical path
    and preserves the layer product, so growing no longer discards all but the first
    layer and shrinking no longer drops the trailing ones. An object with a single layer
    still has its transmission spread evenly over the requested slices, which is the one
    behaviour the previous implementation got right.
    """
    return resample_object_layers(object_, layer_spacing_m)


def pad_object(object_: Object, pad_x: int, pad_y: int) -> Object:
    """Return a zero-padded copy of *object_* with *pad_x* / *pad_y* pixels added on each side."""
    pad_width = [(0, 0), (pad_y, pad_y), (pad_x, pad_x)]

    return Object(
        array=numpy.pad(object_.get_array(), pad_width),
        pixel_geometry=object_.get_pixel_geometry(),
        center=object_.get_center(),
        layer_spacing_m=object_.layer_spacing_m,
    )
