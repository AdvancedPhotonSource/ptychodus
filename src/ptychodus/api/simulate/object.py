"""Object (transmission function) generation functions: STXM differential phase contrast, Paganin phase retrieval, spectral refinement, random fields, simplex/fractal noise, dead-leaves models, and Siemens star resolution targets.

Also home to :class:`ObjectGenerator`, the interface an object generator is
offered through. It sits beside the functions because it is how they are reached: a generator
states what it accepts and turns one set of those values into an object, and the
functions below are what it turns them with.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from typing import Final
import logging
import math

from scipy.fft import fft2, fftfreq, ifft2, ifftshift, set_workers
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import LinearOperator
import numpy
from pydantic import BaseModel

from ..accumulate import DiffractionMoments, compute_diffraction_moments, compute_i0_scale
from ..assemble import AssembledDiffractionData, DetectorCorrections
from ..constants import PI, TWO_PI
from ..diffraction import DiffractionPatterns
from ..geometry import ImageExtent, PixelGeometry
from ..object import Object, ObjectGeometry, ObjectGeometryProvider, resample_object_layers
from ..patch import (
    accumulate_patches_on,
    accumulate_probe_intensity_on,
    compute_patch_pixel_indexes,
    iter_probe_patches_on,
    resolve_probe_patch_bounds_on,
    shift_probe_onto_patches,
)
from ..probe import PatchBounds, Probe, ProbeEnsemble
from ..probe_positions import ProbePosition, ProbePositionSequence
from ..propagate import (
    AngularSpectrumPropagator,
    FresnelTransformPropagator,
    Propagator,
    PropagatorParameters,
    choose_propagator,
)
from ..typing import BooleanArrayType, ComplexArrayType, IntegerArrayType, RealArrayType

__all__ = [
    'ObjectGenerator',
    'STXMDPCSignals',
    'SpectralModulus',
    'SpectralPreprocessing',
    'compute_stxm_dpc_signals',
    'generate_dead_leaves_object',
    'generate_fractal_noise_object',
    'generate_gaussian_random_field_object',
    'generate_layers',
    'generate_paganin_object',
    'generate_random_object',
    'generate_siemens_star_object',
    'generate_simplex_noise_object',
    'generate_spectral_object',
    'generate_stxm_dpc_object',
    'generate_uniform_object',
    'interpolate_position_values',
    'pad_object',
]

logger = logging.getLogger(__name__)


class ObjectGenerator(ABC):
    """An object generator, and the options it accepts.

    Stateless and configured per call: a generator holds no settings of its own, so it
    costs nothing to construct and two callers can drive the same one with different
    values at the same time. It does not name itself either; the name a person selects
    it by is the one it was registered under.

    A generator is a complete description of one kind of object and knows nothing about
    where its option values come from. Its options may state what a field *is* -- its
    unit, its bounds -- but never where a value for it is found. The same holds for its
    other inputs: the random draws and the probe are handed to it.

    What it returns is raw: a single layer on the provider's object grid, unpadded, at
    whatever modulus the model produces. Padding the canvas and scaling to the sample's
    measured transmission are separate concerns from describing a sample, and a
    generator that applied either would be applying it twice.
    """

    @abstractmethod
    def get_default_options(self) -> BaseModel:
        """Return this generator's options, populated with their defaults.

        A description, not live state: plain pydantic carrying values, types, ranges,
        enums and the words to put beside each field. A caller edits a copy and passes
        it back as ``options``, and its type is what an incoming ``options`` is
        validated against.

        An instance rather than the class, because a field declared with a
        ``default_factory`` reports its class-level default as undefined. Only an
        instance carries the value such a field actually starts at.
        """

    @abstractmethod
    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        """Return the object *options* describe, sampled on the provider's object grid.

        *rng* is the source of every random draw; a generator that draws none ignores
        it. *probe* is the illumination on the object's pixels, or None where there is
        none; it is supplied only to a generator that :attr:`reads_probe`.

        Raises:
            GeometryNotDefinedError: while something the object depends on is not yet
                available, which a caller should treat as "not yet" rather than as a
                failure.
        """

    @property
    def reads_diffraction_data(self) -> bool:
        """Whether :meth:`generate` reads the diffraction data, not only the geometry.

        Such an object goes stale when the data changes under an unchanged detector
        geometry, which the geometry alone does not announce.
        """
        return False

    @property
    def reads_probe(self) -> bool:
        """Whether :meth:`generate` reads the probe it is handed."""
        return False


def interpolate_position_values(
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
    detector_sums: RealArrayType
    center_of_mass_x_px: RealArrayType
    center_of_mass_y_px: RealArrayType

    @classmethod
    def from_moments(cls, moments: DiffractionMoments) -> STXMDPCSignals:
        """The per-pattern rows of accumulated diffraction moments."""
        return cls(
            indexes=moments.indexes.copy(),
            detector_sums=moments.detector_sums.copy(),
            center_of_mass_x_px=moments.center_of_mass_x_px.copy(),
            center_of_mass_y_px=moments.center_of_mass_y_px.copy(),
        )


def compute_stxm_dpc_signals(
    assembled_data: AssembledDiffractionData,
    *,
    max_patterns_per_chunk: int = 256,
    i0_by_index: Mapping[int, float] | None = None,
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
    sample's phase gradient read as absorption. *i0_by_index* holds the I0
    readings keyed by scan index, as
    :meth:`ptychodus.api.probe_positions.ProbePositionSequence.get_i0_by_index`
    gives them; each total is then scaled by ``mean(I0) / I0_j`` so beam decay does not
    read as absorption, and a reading at or below zero cannot correct anything and leaves
    its total unscaled.

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
            detector_sums=empty,
            center_of_mass_x_px=empty.copy(),
            center_of_mass_y_px=empty.copy(),
        )

    if i0_by_index:
        moments = moments.normalize_by_i0(
            numpy.fromiter(i0_by_index.keys(), dtype=numpy.intp),
            numpy.fromiter(i0_by_index.values(), dtype=numpy.float64),
        )

    return STXMDPCSignals.from_moments(moments)


def _pair_indexes_with_positions(
    indexes: IntegerArrayType, probe_positions: Iterable[ProbePosition], *, what: str
) -> tuple[BooleanArrayType, ProbePositionSequence]:
    """The *indexes* a position pairs with, and those positions, in index order.

    Rows pair with positions by index: duplicate positions average, a row between
    positions takes an interpolated one, and a row outside the range of position
    indexes is left out. *what* names the rows in messages.

    Raises:
        ValueError: when no position pairs with any index.
    """
    paired, paired_positions = ProbePositionSequence(list(probe_positions)).interpolate_at_indexes(
        indexes
    )
    num_skipped = int(numpy.count_nonzero(~paired))

    if num_skipped > 0:
        logger.debug(f'Skipping {num_skipped} {what}s outside the position indexes.')

    if not paired.any():
        raise ValueError(f'No probe position pairs with any {what} index!')

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
    kx = TWO_PI * fftfreq(probe_geometry.width_px, d=pixel_geometry.width_m)
    ky = TWO_PI * fftfreq(probe_geometry.height_px, d=pixel_geometry.height_m)
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
    kx = TWO_PI * fftfreq(2 * width_px, d=pixel_geometry.width_m)
    ky = TWO_PI * fftfreq(2 * height_px, d=pixel_geometry.height_m)
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
    incident_photon_count: float | None = None,
    regularization_fov_fraction: float = 0.0,
    reference_quantile: float = 0.95,
) -> Object:
    """Generate a coarse complex object from far-field data by STXM differential phase contrast.

    The amplitude is the STXM one: per-position detector sums mapped onto the object grid. With
    *incident_photon_count*, the unattenuated incident photon count, they are divided by it, so the
    modulus is the sample's amplitude transmission. Without it they are divided by their
    mean over the supported region, a dimensionless contrast map that a position
    recording the unobstructed beam brings to unit modulus once a caller scales it to the
    sample's mean opacity.

    The detector sums and the gradients are mapped by :func:`interpolate_position_values`, linearly
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
            differ from the object's, an incident photon count that is not positive, when no
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

    if incident_photon_count is not None and not incident_photon_count > 0.0:
        raise ValueError(
            f'The incident photon count must be positive, not {incident_photon_count}!'
        )

    paired, positions = _pair_indexes_with_positions(
        signals.indexes, probe_positions, what='STXM-DPC signal'
    )
    probes = None if probe is None else ProbeEnsemble.from_probe(probe)

    detector_sums = signals.detector_sums[paired]
    centroid_x_px = signals.center_of_mass_x_px[paired]
    centroid_y_px = signals.center_of_mass_y_px[paired]
    finite = numpy.isfinite(centroid_x_px) & numpy.isfinite(centroid_y_px)
    deviation_x_px = numpy.zeros_like(centroid_x_px)
    deviation_y_px = numpy.zeros_like(centroid_y_px)

    if finite.any():
        deviation_x_px[finite] = centroid_x_px[finite] - numpy.mean(centroid_x_px[finite])
        deviation_y_px[finite] = centroid_y_px[finite] - numpy.mean(centroid_y_px[finite])

    radians_per_px_x = TWO_PI / (detector_extent.width_px * geometry.pixel_width_m)
    radians_per_px_y = TWO_PI / (detector_extent.height_px * geometry.pixel_height_m)
    grids, scanned = interpolate_position_values(
        geometry,
        positions,
        numpy.column_stack(
            (
                detector_sums,
                radians_per_px_x * deviation_x_px,
                radians_per_px_y * deviation_y_px,
            )
        ),
        probes=probes,
    )
    intensity, gradient_x, gradient_y = grids

    if incident_photon_count is None:
        incident_photon_count = float(numpy.mean(intensity[scanned])) if scanned.any() else 0.0

        if incident_photon_count <= 0.0:
            raise ValueError('Mean STXM intensity must be positive!')

    transmission = intensity / incident_photon_count

    if probes is not None and correct_shadow:
        shadow = _compute_probe_shadow_gradient(transmission, probes, positions, geometry)
        shadow -= numpy.mean(shadow, axis=0)
        shadow_grids, _ = interpolate_position_values(geometry, positions, shadow, probes=probes)
        gradient_x = gradient_x - shadow_grids[0]
        gradient_y = gradient_y - shadow_grids[1]

    if regularization_fov_fraction > 0.0:
        cutoff_wavelength_m = regularization_fov_fraction * max(geometry.width_m, geometry.height_m)
        regularization_per_m2 = (TWO_PI / cutoff_wavelength_m) ** 2
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
    I(x,y) is the per-position detector sums of `signals`, mapped onto the object grid as
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

    paired, positions = _pair_indexes_with_positions(
        signals.indexes, probe_positions, what='STXM-DPC signal'
    )
    grids, scanned = interpolate_position_values(
        geometry,
        positions,
        signals.detector_sums[paired],
        probes=None if probe is None else ProbeEnsemble.from_probe(probe),
    )
    intensity = grids[0]
    mean_intensity = float(numpy.mean(intensity[scanned])) if scanned.any() else 0.0

    if mean_intensity <= 0.0:
        raise ValueError('Mean STXM intensity must be positive!')

    intensity_normalized = intensity / mean_intensity

    pixel_geometry = geometry.get_pixel_geometry()
    kx = TWO_PI * fftfreq(geometry.width_px, d=pixel_geometry.width_m)
    ky = TWO_PI * fftfreq(geometry.height_px, d=pixel_geometry.height_m)
    KY, KX = numpy.meshgrid(ky, kx, indexing='ij')  # noqa: N806
    K2 = numpy.square(KX) + numpy.square(KY)  # noqa: N806

    filter_denominator = (
        1.0 + delta_over_beta * propagation_distance_m * photon_wavelength_m * K2 / (4 * PI)
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


class SpectralPreprocessing(StrEnum):
    """How :func:`generate_spectral_object` weights each normalized intensity ``y``."""

    OPTIMAL = 'optimal'
    """``max(-beta, 1 - 1 / y)``: the preprocessing W. Luo, W. Alghamdi and Y. M. Lu,
    IEEE Trans. Signal Process. 67, 2347 (2019), prove uniformly optimal for noiseless
    intensities, floored at ``-beta`` as L. Valzania, J. Dong and S. Gigan, Opt. Lett. 46,
    1357 (2021), floor it."""
    THRESHOLD = 'threshold'
    """One for the brightest fraction of good pixels and zero for the rest; S. Marchesini,
    Y.-C. Tu and H.-T. Wu, Appl. Comput. Harmon. Anal. 41, 815 (2016)."""
    IDENTITY = 'identity'
    """The intensity itself; E. J. Candès, X. Li and M. Soltanolkotabi, IEEE Trans. Inf.
    Theory 61, 1985 (2015)."""


class SpectralModulus(StrEnum):
    """Where :func:`generate_spectral_object` takes the estimate's modulus from."""

    START = 'start'
    """The start's, smoothed over the probe's RMS radius; uniform without a start."""
    ESTIMATE = 'estimate'
    """The refinement's own, which also measures how strongly the sample scatters."""


def _build_propagator_pair(
    parameters: PropagatorParameters, *, far_field: bool
) -> tuple[Propagator, Propagator]:
    """The detector propagator of the multislice forward model, and its adjoint.

    Far field is the Fresnel transform onto the reciprocal grid; near field is the
    propagator :func:`ptychodus.api.propagate.choose_propagator` samples correctly. The
    adjoint is the same propagator over the negated distance: the exact inverse of a
    single-FFT propagator, which is unitary, and the conjugate transfer function of the
    angular spectrum, which is its adjoint with or without the evanescent cutoff.
    """
    backward_parameters = replace(
        parameters, propagation_distance_m=-parameters.propagation_distance_m
    )

    if far_field:
        return FresnelTransformPropagator(parameters), FresnelTransformPropagator(
            backward_parameters
        )

    forward, _ = choose_propagator(parameters)

    if isinstance(forward, AngularSpectrumPropagator):
        return forward, AngularSpectrumPropagator(backward_parameters)

    return forward, FresnelTransformPropagator(backward_parameters)


def _compute_spectral_weights(
    normalized: RealArrayType,
    *,
    preprocessing: SpectralPreprocessing,
    lower_bound: float,
    threshold: float,
) -> RealArrayType:
    """Map intensities normalized to unit mean through the preprocessing ``T(y)``.

    *lower_bound* is the floor of :attr:`SpectralPreprocessing.OPTIMAL` and *threshold*
    the normalized cut of :attr:`SpectralPreprocessing.THRESHOLD`; each preprocessing
    ignores the other's. Always returns a new array.
    """
    match preprocessing:
        case SpectralPreprocessing.OPTIMAL:
            # 1 - 1/y reaches -beta at y = 1/(1 + beta), so clamping y there first applies
            # the floor without dividing by the zeros photon counting records.
            floor = 1.0 / (1.0 + lower_bound)
            return 1.0 - 1.0 / numpy.maximum(normalized, floor)
        case SpectralPreprocessing.THRESHOLD:
            return (normalized > threshold).astype(numpy.float64)
        case SpectralPreprocessing.IDENTITY:
            return normalized.copy()


@dataclass(frozen=True)
class _IntensityScale:
    """What one pass over the corrected intensities measures."""

    mean: float
    """The mean over the good pixels of every pattern."""
    threshold: float
    """The normalized threshold, or zero without a threshold fraction."""
    mean_pattern: RealArrayType
    """The mean pattern, bad pixels included."""


def _measure_intensity_scale(
    intensity_chunks: Iterable[RealArrayType],
    good_pixels: BooleanArrayType,
    *,
    threshold_fraction: float | None,
) -> _IntensityScale:
    """Measure the mean over the good pixels of every chunk, the mean pattern, and a threshold.

    With *threshold_fraction*, the threshold is the intensity, divided by the mean, that
    the brightest fraction of good pixels exceeds, found over a float32 copy of every good
    pixel; without one it is zero and nothing is copied.

    Raises:
        ValueError: when the good pixels hold no signal.
    """
    total = 0.0
    num_good = 0
    num_patterns = 0
    pattern_sum = numpy.zeros(good_pixels.shape)
    good_values: list[RealArrayType] = []

    for intensities in intensity_chunks:
        values = intensities[:, good_pixels]
        total += float(numpy.sum(values))
        num_good += values.size
        num_patterns += intensities.shape[0]
        pattern_sum += numpy.sum(intensities, axis=0)

        if threshold_fraction is not None:
            good_values.append(values.astype(numpy.float32).ravel())

    mean = total / num_good if num_good > 0 else 0.0

    if not mean > 0.0:
        raise ValueError('The diffraction patterns hold no signal on their good pixels!')

    threshold = (
        float(numpy.quantile(numpy.concatenate(good_values), 1.0 - threshold_fraction)) / mean
        if threshold_fraction is not None
        else 0.0
    )
    return _IntensityScale(mean, threshold, pattern_sum / num_patterns)


def _correlate(a: RealArrayType, b: RealArrayType) -> float:
    """Pearson's correlation of *a* and *b*; zero when either is constant."""
    a = a - numpy.mean(a)
    b = b - numpy.mean(b)
    denominator = float(numpy.linalg.norm(a) * numpy.linalg.norm(b))
    return float(numpy.sum(a * b)) / denominator if denominator > 0.0 else 0.0


def _measure_corrected_intensities(
    chunk: slice,
    *,
    patterns: DiffractionPatterns,
    pattern_rows: IntegerArrayType,
    corrections: DetectorCorrections,
    i0_scale: RealArrayType,
) -> RealArrayType:
    """The paired patterns of *chunk*, corrected for the detector maps and scaled by I0."""
    corrected = corrections.apply(patterns[pattern_rows[chunk]])
    return corrected * i0_scale[chunk, numpy.newaxis, numpy.newaxis]


class _PropagatorSandwich:
    """``D^H diag(w) D`` through the detector propagator and its adjoint, any regime."""

    def __init__(self, forward: Propagator, backward: Propagator) -> None:
        self._forward = forward
        self._backward = backward

    def prepare_probes(self, shifted: ComplexArrayType) -> ComplexArrayType:
        return shifted

    def prepare_weights(self, weights: RealArrayType) -> RealArrayType:
        return weights

    def apply(self, exit_waves: ComplexArrayType, weights: RealArrayType) -> ComplexArrayType:
        return self._backward.propagate(weights * self._forward.propagate(exit_waves))


class _FourierSandwich:
    """``D^H diag(w) D`` for a Fresnel-transform ``D``, as one FFT pair per wave.

    The output-plane phase cancels against the real weights and the shifts commute with
    the convolution left between them, so only the input-plane phase ``B`` remains: it
    is folded into the probes once, as the weights are into FFT order.
    """

    def __init__(self, propagator: FresnelTransformPropagator) -> None:
        self._input_plane_phase = propagator.get_input_plane_phase()

    def prepare_probes(self, shifted: ComplexArrayType) -> ComplexArrayType:
        return shifted * self._input_plane_phase

    def prepare_weights(self, weights: RealArrayType) -> RealArrayType:
        return ifftshift(weights, axes=(-2, -1))

    def apply(self, exit_waves: ComplexArrayType, weights: RealArrayType) -> ComplexArrayType:
        return ifft2(weights * fft2(exit_waves))


class _ShiftedProbeStack:
    """The probe shifted onto every placement, computed once and kept or per request.

    Chunks bound the working memory of the shift either way. *prepare* maps every
    shifted stack before it is kept or returned, in single precision, which costs eight
    bytes a pixel and mode for every placement kept and halves the cost of every wave
    computed from it.
    """

    def __init__(
        self,
        probe_modes: ComplexArrayType,
        bounds: Sequence[PatchBounds],
        chunks: Sequence[slice],
        *,
        prepare: Callable[[ComplexArrayType], ComplexArrayType],
        cached: bool,
    ) -> None:
        self._probe_modes = probe_modes
        self._bounds = bounds
        self._prepare = prepare
        self._cached = (
            numpy.concatenate([self._shift(chunk) for chunk in chunks]) if cached else None
        )

    def _shift(self, chunk: slice) -> ComplexArrayType:
        shifted = shift_probe_onto_patches(self._probe_modes, self._bounds[chunk])
        return self._prepare(shifted).astype(numpy.complex64)

    def get(self, chunk: slice) -> ComplexArrayType:
        """The shifted modes of the placements in *chunk*, ``(n, modes, height, width)``."""
        if self._cached is None:
            return self._shift(chunk)
        else:
            return self._cached[chunk]


class _SpectralCovariance(LinearOperator):
    """The weighted covariance ``sum_j sum_k S_jk^H diag(T(y_j) + c) S_jk``, matrix-free.

    ``S_jk`` places mode ``k`` of the shifted probe at placement ``j``, multiplies it into
    the object and propagates it to the detector. The intensities are normalized to unit
    mean over the good pixels when the operator is built; their weights, offset by
    `weight_offset` ``c`` and zero at every bad pixel, are kept as float32 when
    `cache_weights` asks and recomputed per product otherwise. ``Z`` is Hermitian, so its
    adjoint is itself. `num_products` counts the products taken, and `mean_pattern` is
    the mean of the corrected intensities.
    """

    def __init__(
        self,
        *,
        intensities: Callable[[slice], RealArrayType],
        chunks: Sequence[slice],
        bounds: Sequence[PatchBounds],
        geometry: ObjectGeometry,
        probes: _ShiftedProbeStack,
        sandwich: _PropagatorSandwich | _FourierSandwich,
        good_pixels: BooleanArrayType,
        preprocessing: SpectralPreprocessing,
        lower_bound: float,
        threshold_fraction: float,
        weight_offset: float,
        cache_weights: bool,
    ) -> None:
        num_pixels = geometry.height_px * geometry.width_px
        super().__init__(numpy.complex128, (num_pixels, num_pixels))
        self._intensities = intensities
        self._chunks = chunks
        self._bounds = bounds
        self._geometry = geometry
        self._probes = probes
        self._sandwich = sandwich
        self._bad_pixels = numpy.logical_not(good_pixels)
        self._preprocessing = preprocessing
        self._lower_bound = lower_bound
        self._weight_offset = weight_offset
        scale = _measure_intensity_scale(
            (intensities(chunk) for chunk in chunks),
            good_pixels,
            threshold_fraction=(
                threshold_fraction if preprocessing == SpectralPreprocessing.THRESHOLD else None
            ),
        )
        self._mean = scale.mean
        self._threshold = scale.threshold
        self.mean_pattern = scale.mean_pattern
        self._cached_weights = (
            [self._compute_weights(chunk).astype(numpy.float32) for chunk in chunks]
            if cache_weights
            else None
        )
        self.num_products = 0

    def _compute_weights(self, chunk: slice) -> RealArrayType:
        weights = _compute_spectral_weights(
            self._intensities(chunk) / self._mean,
            preprocessing=self._preprocessing,
            lower_bound=self._lower_bound,
            threshold=self._threshold,
        )
        weights += self._weight_offset
        weights[:, self._bad_pixels] = 0.0
        return self._sandwich.prepare_weights(weights)

    def _weights(self, chunk_index: int) -> RealArrayType:
        if self._cached_weights is None:
            return self._compute_weights(self._chunks[chunk_index])
        else:
            return self._cached_weights[chunk_index]

    def _matvec(self, x: ComplexArrayType) -> ComplexArrayType:
        self.num_products += 1
        canvas = numpy.ravel(x)
        result = numpy.zeros(
            (self._geometry.height_px, self._geometry.width_px), dtype=numpy.complex128
        )

        for chunk_index, chunk in enumerate(self._chunks):
            pixel_indexes = compute_patch_pixel_indexes(self._bounds[chunk], self._geometry)
            shifted = self._probes.get(chunk)
            weights = self._weights(chunk_index)[:, numpy.newaxis]
            values = canvas[pixel_indexes].astype(numpy.complex64)
            exit_waves = shifted * values[:, numpy.newaxis]
            back = self._sandwich.apply(exit_waves, weights)
            update = numpy.sum(numpy.conj(shifted) * back, axis=1)
            result += accumulate_patches_on(update, pixel_indexes, self._geometry)

        return result.ravel()

    def _rmatvec(self, x: ComplexArrayType) -> ComplexArrayType:
        return self._matvec(x)


def _iterate_power(
    covariance: _SpectralCovariance, *, start: ComplexArrayType, num_iterations: int
) -> ComplexArrayType:
    """*num_iterations* products of *covariance*, each normalized, from *start*."""
    vector = start / numpy.linalg.norm(start)

    for iteration in range(num_iterations):
        product = covariance.matvec(vector)
        rayleigh = float(numpy.vdot(vector, product).real)
        vector = product / numpy.linalg.norm(product)
        logger.debug(f'Spectral iteration {iteration + 1}: Rayleigh quotient {rayleigh:.6g}.')

    return vector


def _compute_rms_radius_px(probe_modes: ComplexArrayType) -> float:
    """The RMS radius of the probe's summed intensity, in pixels."""
    intensity = numpy.sum(numpy.square(numpy.abs(probe_modes)), axis=0)
    total = float(numpy.sum(intensity))
    y, x = numpy.indices(intensity.shape)
    center_y = numpy.sum(y * intensity) / total
    center_x = numpy.sum(x * intensity) / total
    radius2 = numpy.square(y - center_y) + numpy.square(x - center_x)
    return math.sqrt(float(numpy.sum(radius2 * intensity)) / total)


def generate_spectral_object(
    geometry: ObjectGeometry,
    assembled_data: AssembledDiffractionData,
    probe_positions: Iterable[ProbePosition],
    *,
    probe: Probe,
    photon_wavelength_m: float,
    propagation_distance_m: float,
    far_field: bool,
    start: Object | None = None,
    num_iterations: int = 3,
    preprocessing: SpectralPreprocessing = SpectralPreprocessing.IDENTITY,
    lower_bound: float = 0.25,
    threshold_fraction: float = 0.02,
    modulus: SpectralModulus = SpectralModulus.START,
    i0_by_index: Mapping[int, float] | None = None,
    support_fraction: float = 0.05,
    max_patterns_per_chunk: int = 64,
    max_cached_bytes: int = 1 << 30,
    workers: int = -1,
    min_probe_consistency: float = 0.5,
) -> Object:
    """Generate a complex object by spectral refinement of *start* with every pixel of the patterns.

    The refinement takes `num_iterations` power-method steps with the weighted covariance

        Z = sum_j sum_k S_jk^H diag(T(y_j) + c) S_jk,    S_jk psi = D(P_jk * psi[patch_j]),

    where ``P_jk`` is incoherent mode ``k`` of the `probe`, placed at position ``j`` as
    the multislice forward model places it, ``D`` the detector propagator, ``T`` the
    `preprocessing` of the normalized intensities ``y_j``, and ``c`` the offset that makes
    every weight non-negative (``lower_bound`` for :attr:`SpectralPreprocessing.OPTIMAL`,
    zero otherwise), so the steps follow the largest eigenvalues rather than the most
    negative. ``Z`` is the operator of L. Valzania, J. Dong and S. Gigan, Opt. Lett. 46,
    1357 (2021), Appendix C (reference code https://github.com/laboGigan/ptychography-spectral),
    applied matrix-free: a forward pass, a weight and an adjoint pass per pattern, batched
    over `max_patterns_per_chunk` patterns at once. A Fresnel-transform ``D`` needs one FFT
    pair per pattern and mode, its quadratic phases folded into the probe and the weights.

    The spectral estimate proper is ``Z``'s leading eigenvector, and on measured data it
    is the wrong target. With a focused probe on a canvas many probe widths across,
    ``Z`` acts locally and its leading eigenvalues lie within about a percent of each
    other; each of their eigenvectors concentrates on a different patch of the most
    strongly scattering features, and its modulus measures scattering strength rather
    than transmission. A few steps from a start spread over the whole field resolve the
    local structure, whose gaps are large, before the global competition, whose gaps are
    small, can concentrate the estimate; every further step sharpens detail where the
    probe is right and amplifies its error where it is not. `start`, an object on
    *geometry*'s canvas such as the STXM-DPC estimate, supplies the low-resolution
    object; without one the steps start from the illuminated region, uniformly.

    The estimate's phase is the refinement's. Its modulus is the start's smoothed over
    the probe's RMS radius with :attr:`SpectralModulus.START`, which keeps the
    low-resolution transmission the start measured and drops detail finer than the
    probe that a noisy start carries, and uniform without a start; with
    :attr:`SpectralModulus.ESTIMATE` it is the refinement's own, which carries an
    absorbing sample's transmission but also the scattering-strength bias above.

    The intensities are corrected for any detector background or flat field the data
    still carries, as ``(pattern - background) / flat_field`` clipped at zero, and their
    bad pixels widened by the pixels those maps make unusable take zero weight, so a gap
    or masked beamstop leaves ``Z`` exactly. With `i0_by_index`, keyed by scan
    index, each pattern is scaled by ``mean(I0) / I0_j`` as
    :func:`ptychodus.api.accumulate.compute_i0_scale` gives it. The stack is then divided
    by its mean over every good pixel of every pattern, the normalization under which
    W. Luo, W. Alghamdi and Y. M. Lu, IEEE Trans. Signal Process. 67, 2347 (2019), find
    ``1 - 1/y`` uniformly optimal; the absorption contrast between positions survives,
    which dividing each pattern by its own mean would discard.

    :attr:`SpectralPreprocessing.IDENTITY` weights every pixel by its intensity, so the
    dark field, where the detail finer than the probe is recorded, takes part.
    :attr:`SpectralPreprocessing.OPTIMAL` floors every intensity below ``1 / (1 +
    lower_bound)`` of the mean at ``-lower_bound``, the shot-noise-limited and zero
    counts of a photon-counting detector among them; offset to be non-negative, the
    floored pixels take no weight, which slows the refinement as more of the detector
    lies in the dark field.
    :attr:`SpectralPreprocessing.THRESHOLD` sets the brightest `threshold_fraction` of
    good pixels to one, found over a float32 copy of every good pixel, the one temporary
    the size of the data. Patterns pair with positions by index; one outside the range
    of position indexes is left out. The estimate is only as good as the probe it
    assumes, its defocus above all; a warning is logged when the correlation over the
    good pixels between the mean corrected pattern and the far-field intensity the probe
    alone predicts, ``sum_k |D P_k|^2``, falls below `min_probe_consistency`.

    The phase is fixed so the sum over the support -- where the summed illumination
    reaches `support_fraction` of its peak -- is real and positive, and the modulus is
    scaled to a mean of one there; pixels beyond the support are one.

    Up to `max_cached_bytes`, what every product reuses is computed once and kept: the
    weights first, at four bytes a detector pixel, then the probe shifted onto every
    position, at eight a pixel and mode; each pattern's waves are single precision, and
    the products accumulate in double. What does not fit is recomputed for each chunk
    of every product, which changes the cost but not the result. `workers` is the number
    of FFT threads, as :func:`scipy.fft.set_workers` takes it.

    Raises:
        ValueError: if a tunable is out of range, the probe's pixels differ from the
            object's or its extent from the patterns', the start is not on the canvas,
            no position pairs with any pattern, or the good pixels hold no signal.
    """
    if num_iterations < 1:
        raise ValueError('The refinement needs at least one iteration!')

    if not 0.0 <= lower_bound <= 1.0:
        raise ValueError('The lower bound must lie in [0, 1]!')

    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError('The threshold fraction must lie in (0, 1]!')

    if not 0.0 < support_fraction <= 1.0:
        raise ValueError('Support fraction must lie in (0, 1]!')

    if max_patterns_per_chunk < 1:
        raise ValueError('Chunks must hold at least one pattern!')

    if max_cached_bytes < 0:
        raise ValueError('The cache budget must be non-negative!')

    if far_field and propagation_distance_m == 0.0:
        raise ValueError('Far-field propagation requires a nonzero distance!')

    pixel_geometry = geometry.get_pixel_geometry()
    probe_pixel_geometry = probe.get_pixel_geometry()

    if not (
        math.isclose(probe_pixel_geometry.width_m, pixel_geometry.width_m, rel_tol=1e-6)
        and math.isclose(probe_pixel_geometry.height_m, pixel_geometry.height_m, rel_tol=1e-6)
    ):
        raise ValueError(
            f'Probe pixels ({probe_pixel_geometry}) differ from object pixels'
            f' ({pixel_geometry}); resample the probe onto the object grid first.'
        )

    canvas_shape = (geometry.height_px, geometry.width_px)
    start_array = None if start is None else start.get_layers_flattened()

    if start_array is not None and start_array.shape != canvas_shape:
        raise ValueError(
            f'The {start_array.shape} start is not on the {canvas_shape} object canvas!'
        )

    patterns = assembled_data.get_patterns()
    modes = probe.get_array()
    probe_extent_px = modes.shape[-2:]

    if patterns.shape[-2:] != probe_extent_px:
        raise ValueError(
            f'Patterns of shape {patterns.shape[-2:]} do not match the {probe_extent_px} probe!'
        )

    indexes = numpy.asarray(assembled_data.get_indexes())
    paired, positions = _pair_indexes_with_positions(
        indexes, probe_positions, what='diffraction pattern'
    )
    pattern_rows = numpy.flatnonzero(paired)
    chunks = [
        slice(begin, min(begin + max_patterns_per_chunk, pattern_rows.size))
        for begin in range(0, pattern_rows.size, max_patterns_per_chunk)
    ]

    probe_ensemble = ProbeEnsemble.from_probe(probe)
    patch_bounds = resolve_probe_patch_bounds_on(probe_ensemble.get_geometry(), positions, geometry)
    illumination = accumulate_probe_intensity_on(probe_ensemble, positions, geometry)
    peak = float(illumination.max())

    if not peak > 0.0:
        raise ValueError('The probe illuminates nothing on the object canvas!')

    support = illumination >= support_fraction * peak

    corrections = DetectorCorrections(
        background=assembled_data.get_background(),
        flat_field=assembled_data.get_flat_field(),
        subtract_background=True,
        divide_flat_field=True,
    )
    good_pixels = numpy.logical_not(
        corrections.apply_to_bad_pixels(assembled_data.get_bad_pixels())
    )
    i0_scale = (
        compute_i0_scale(
            indexes[pattern_rows],
            numpy.fromiter(i0_by_index.keys(), dtype=numpy.intp),
            numpy.fromiter(i0_by_index.values(), dtype=numpy.float64),
        )
        if i0_by_index
        else numpy.ones(pattern_rows.size)
    )
    forward, backward = _build_propagator_pair(
        PropagatorParameters(
            wavelength_m=photon_wavelength_m,
            width_px=probe_extent_px[1],
            height_px=probe_extent_px[0],
            pixel_width_m=probe_pixel_geometry.width_m,
            pixel_height_m=probe_pixel_geometry.height_m,
            propagation_distance_m=propagation_distance_m,
        ),
        far_field=far_field,
    )
    sandwich = (
        _FourierSandwich(forward)
        if isinstance(forward, FresnelTransformPropagator)
        else _PropagatorSandwich(forward, backward)
    )

    weight_bytes = (
        pattern_rows.size * math.prod(probe_extent_px) * numpy.dtype(numpy.float32).itemsize
    )
    caches_weights = weight_bytes <= max_cached_bytes
    probe_budget = max_cached_bytes - weight_bytes if caches_weights else max_cached_bytes

    with set_workers(workers):
        covariance = _SpectralCovariance(
            intensities=partial(
                _measure_corrected_intensities,
                patterns=patterns,
                pattern_rows=pattern_rows,
                corrections=corrections,
                i0_scale=i0_scale,
            ),
            chunks=chunks,
            bounds=patch_bounds,
            geometry=geometry,
            probes=_ShiftedProbeStack(
                modes,
                patch_bounds,
                chunks,
                prepare=sandwich.prepare_probes,
                cached=(
                    pattern_rows.size * modes.size * numpy.dtype(numpy.complex64).itemsize
                    <= probe_budget
                ),
            ),
            sandwich=sandwich,
            good_pixels=good_pixels,
            preprocessing=preprocessing,
            lower_bound=lower_bound,
            threshold_fraction=threshold_fraction,
            weight_offset=(lower_bound if preprocessing == SpectralPreprocessing.OPTIMAL else 0.0),
            cache_weights=caches_weights,
        )
        estimate = _iterate_power(
            covariance,
            start=(
                support.astype(numpy.complex128)
                if start_array is None
                else start_array.astype(numpy.complex128)
            ).ravel(),
            num_iterations=num_iterations,
        ).reshape(canvas_shape)
        predicted = numpy.sum(numpy.square(numpy.abs(forward.propagate(modes))), axis=0)

    consistency = _correlate(covariance.mean_pattern[good_pixels], predicted[good_pixels])
    logger.debug(f'The probe predicts the mean pattern with correlation {consistency:.3f}.')

    if consistency < min_probe_consistency:
        logger.warning(
            f'The probe predicts the mean diffraction pattern poorly (correlation'
            f' {consistency:.2f}), so the spectral refinement follows its errors; an'
            ' average-pattern probe at the measured defocus reproduces it.'
        )

    match modulus:
        case SpectralModulus.START:
            magnitude = (
                numpy.ones(canvas_shape)
                if start_array is None
                else gaussian_filter(numpy.abs(start_array), _compute_rms_radius_px(modes))
            )
            estimate = magnitude * numpy.exp(1j * numpy.angle(estimate))
        case SpectralModulus.ESTIMATE:
            pass

    supported = estimate[support]
    supported_sum = numpy.sum(supported)

    if abs(supported_sum) > 0.0:
        estimate = estimate * (abs(supported_sum) / supported_sum)

    mean_modulus = float(numpy.mean(numpy.abs(estimate[support])))

    if not mean_modulus > 0.0:
        raise ValueError('The spectral estimate vanishes on the illuminated region!')

    array = numpy.where(support, estimate / mean_modulus, 1.0 + 0.0j)

    return Object(
        array=array[numpy.newaxis, :, :].astype(complex),
        pixel_geometry=pixel_geometry,
        center=geometry.get_center(),
    )


def generate_uniform_object(geometry: ObjectGeometry, *, amplitude: float = 1.0) -> Object:
    """Generate a featureless object of uniform *amplitude* and zero phase.

    The initial guess for a reconstruction that asserts nothing about the structure of
    the sample, as distinct from :func:`generate_random_object` with its deviations set
    to zero -- which computes the same array, but reads as though the result were random.

    The default is a transparent sample. An amplitude below one asserts uniform
    absorption instead, which is what a measured mean transmission states: the exit wave
    then predicts the detector sums the detector recorded rather than the unobstructed
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
    phase_mean_rad: float,
    phase_deviation_rad: float,
    blur_deviation_px: float,
) -> Object:
    """Generate a random complex object from Gaussian amplitude and phase distributions, with optional Gaussian blur.

    Phase is in radians, as the ``_rad`` suffixes say: the array is
    ``amplitude * exp(1j * phase)``, so a `phase_mean_rad` of pi is half a cycle.
    """
    object_shape = (1, geometry.height_px, geometry.width_px)

    amplitude = rng.normal(
        amplitude_mean,
        amplitude_deviation,
        object_shape,
    )
    phase = rng.normal(
        phase_mean_rad,
        phase_deviation_rad,
        object_shape,
    )

    if blur_deviation_px > 0.0:
        # TODO account for pixel aspect ratio in blur calculation
        amplitude = gaussian_filter(amplitude, sigma=blur_deviation_px)
        phase = gaussian_filter(phase, sigma=blur_deviation_px)

    array = amplitude * numpy.exp(1j * phase)

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
    power_spectrum = numpy.exp(-0.5 * (TWO_PI * correlation_length_px) ** 2 * K2)
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
    angle_rad = TWO_PI * rng.uniform(size=grad_shape)
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


def _integrate_power_law(lower: float, upper: float, exponent: float) -> float:
    """Return the integral of ``r**-exponent`` over ``[lower, upper]``."""
    if math.isclose(exponent, 1.0):
        return math.log(upper / lower)

    power = 1.0 - exponent
    return (upper**power - lower**power) / power


def _sample_truncated_power_law(
    rng: numpy.random.Generator, lower: float, upper: float, exponent: float, size: int
) -> RealArrayType:
    """Draw *size* values with density proportional to ``r**-exponent`` on ``[lower, upper]``.

    Inverse-CDF sampling, valid for any real exponent; an exponent of one is log-uniform.
    """
    uniform = rng.uniform(size=size)

    if math.isclose(exponent, 1.0):
        return lower * numpy.power(upper / lower, uniform)

    power = 1.0 - exponent
    lower_power = lower**power
    upper_power = upper**power
    return numpy.power(lower_power + uniform * (upper_power - lower_power), 1.0 / power)


def _sample_window_leaves(
    rng: numpy.random.Generator,
    lower: float,
    upper: float,
    exponent: float,
    width: float,
    height: float,
    size: int,
) -> tuple[RealArrayType, RealArrayType, RealArrayType]:
    """Draw *size* leaves of a Poisson dead-leaves process that can reach a window.

    The window is ``[0, width) x [0, height)`` and leaf radii have density proportional to
    ``r**-exponent`` on ``[lower, upper]``. A leaf of radius r is kept when its center
    lies in the window widened by r, so the kept radii have density proportional to
    ``r**-exponent * (width + 2r) * (height + 2r)``: a three-term mixture of truncated
    power laws, sampled exactly. The centers are then uniform in the widened window.

    Returns:
        The center x, center y and radius of each leaf.
    """
    weights = numpy.array(
        [
            width * height * _integrate_power_law(lower, upper, exponent),
            2.0 * (width + height) * _integrate_power_law(lower, upper, exponent - 1.0),
            4.0 * _integrate_power_law(lower, upper, exponent - 2.0),
        ]
    )
    component = rng.choice(len(weights), size=size, p=weights / weights.sum())
    radius: RealArrayType = numpy.empty(size)

    for degree in range(len(weights)):
        is_component = component == degree
        radius[is_component] = _sample_truncated_power_law(
            rng, lower, upper, exponent - degree, int(numpy.count_nonzero(is_component))
        )

    center_x = rng.uniform(-radius, width + radius)
    center_y = rng.uniform(-radius, height + radius)
    return center_x, center_y, radius


def _propose_leaf_hitting_points(
    rng: numpy.random.Generator,
    points_x: RealArrayType,
    points_y: RealArrayType,
    lower: float,
    upper: float,
    exponent: float,
) -> tuple[float, float, float, BooleanArrayType, int]:
    """Draw the next leaf of a Poisson dead-leaves process that covers one of the points.

    Karp-Luby-Madras union sampling: a radius weighted by leaf area, a point chosen
    uniformly, and a center uniform in the disk of that radius about it propose a leaf
    with density proportional to ``p(r) * k``, where k counts the points it covers.
    Accepting with probability ``1 / k`` leaves exactly the process's leaves that cover
    at least one point, so no draw is ever spent on a leaf that covers none.

    Returns:
        The accepted leaf's center x, center y and radius, the mask of the points it
        covers (never empty), and the number of proposals it took.
    """
    num_proposals = 0

    while True:
        num_proposals += 1
        radius = float(_sample_truncated_power_law(rng, lower, upper, exponent - 2.0, 1)[0])
        anchor = int(rng.integers(len(points_x)))
        distance = radius * math.sqrt(rng.uniform())
        angle = TWO_PI * rng.uniform()
        center_x = float(points_x[anchor]) + distance * math.cos(angle)
        center_y = float(points_y[anchor]) + distance * math.sin(angle)
        is_covered = numpy.hypot(points_x - center_x, points_y - center_y) <= radius
        num_covered = int(numpy.count_nonzero(is_covered))

        # num_covered is zero only when rounding puts the anchor just outside the disk
        if num_covered > 0 and rng.uniform() * num_covered < 1.0:
            return center_x, center_y, radius, is_covered, num_proposals


@dataclass(frozen=True)
class _LeafMarks:
    """The transmission each leaf carries: uniform phase, amplitude uniform in intensity."""

    amplitude_lower: float
    amplitude_upper: float
    phase_lower_rad: float
    phase_upper_rad: float

    def draw(self, rng: numpy.random.Generator, size: int) -> ComplexArrayType:
        phase_rad = rng.uniform(self.phase_lower_rad, self.phase_upper_rad, size=size)
        amplitude = numpy.sqrt(
            rng.uniform(self.amplitude_lower**2, self.amplitude_upper**2, size=size)
        )
        return amplitude * numpy.exp(1j * phase_rad)


class _DeadLeavesCanvas:
    """A dead-leaves field painted front to back onto jittered pixel subsamples.

    Coordinates are in pixel widths, with y scaled by the pixel aspect ratio so leaves are
    disks in physical space. Each pixel holds a stratified n x n grid of subsamples, one
    uniformly jittered point per cell. A leaf lies beneath every leaf painted before it,
    so it claims only subsamples still uncovered, and a pixel's value is the mean
    transmission of its subsamples. The field is complete, and exact, once every
    subsample is claimed.
    """

    _BATCH_SIZE: Final[int] = 1024
    _TAIL_SIZE: Final[int] = 2048
    """Uncovered subsamples at which painting switches to drawing only leaves that cover one."""

    def __init__(
        self,
        rng: numpy.random.Generator,
        *,
        width_px: int,
        height_px: int,
        aspect: float,
        num_subsamples_per_axis: int,
    ) -> None:
        num_subsamples = num_subsamples_per_axis**2
        shape = (height_px, width_px, num_subsamples)
        cell_y, cell_x = numpy.divmod(numpy.arange(num_subsamples), num_subsamples_per_axis)
        jitter = rng.uniform(size=(2, *shape))

        self._width_px = width_px
        self._height_px = height_px
        self._aspect = aspect
        self._num_subsamples = num_subsamples
        self._offset_x: RealArrayType = ((cell_x + jitter[0]) / num_subsamples_per_axis).astype(
            numpy.float32
        )
        self._offset_y: RealArrayType = ((cell_y + jitter[1]) / num_subsamples_per_axis).astype(
            numpy.float32
        )
        self._is_uncovered: BooleanArrayType = numpy.ones(shape, dtype=bool)
        self._num_uncovered = self._is_uncovered.size
        self._pixel_sum: ComplexArrayType = numpy.zeros((height_px, width_px), dtype=complex)
        self.num_leaves = 0
        self.num_tail_leaves = 0
        self.num_tail_proposals = 0

    def fill(
        self,
        rng: numpy.random.Generator,
        marks: _LeafMarks,
        *,
        radius_lower_px: float,
        radius_upper_px: float,
        radius_exponent: float,
    ) -> None:
        """Paint leaves until every subsample is covered."""
        width = float(self._width_px)
        height = self._height_px * self._aspect

        while self._num_uncovered > self._TAIL_SIZE:
            centers_x, centers_y, radii = _sample_window_leaves(
                rng,
                radius_lower_px,
                radius_upper_px,
                radius_exponent,
                width,
                height,
                self._BATCH_SIZE,
            )
            values = marks.draw(rng, self._BATCH_SIZE)

            for center_x, center_y, radius, value in zip(centers_x, centers_y, radii, values):
                self._paint(float(center_x), float(center_y), float(radius), complex(value))
                self.num_leaves += 1

                if self._num_uncovered <= self._TAIL_SIZE:
                    break

        rows, columns, subsamples = numpy.nonzero(self._is_uncovered)
        points_x = columns + self._offset_x[rows, columns, subsamples]
        points_y = (rows + self._offset_y[rows, columns, subsamples]) * self._aspect
        pixel_indexes = rows * self._width_px + columns
        flat_pixel_sum = self._pixel_sum.reshape(-1)

        while points_x.size > 0:
            _, _, _, is_covered, num_proposals = _propose_leaf_hitting_points(
                rng, points_x, points_y, radius_lower_px, radius_upper_px, radius_exponent
            )
            value = marks.draw(rng, 1)[0]
            numpy.add.at(flat_pixel_sum, pixel_indexes[is_covered], value)
            is_kept = ~is_covered
            points_x = points_x[is_kept]
            points_y = points_y[is_kept]
            pixel_indexes = pixel_indexes[is_kept]
            self.num_leaves += 1
            self.num_tail_leaves += 1
            self.num_tail_proposals += num_proposals

        self._is_uncovered[...] = False
        self._num_uncovered = 0

    def _paint(self, center_x: float, center_y: float, radius: float, value: complex) -> None:
        """Lay one leaf beneath those already painted, within its clipped bounding box."""
        column_start = max(math.floor(center_x - radius), 0)
        column_stop = min(math.floor(center_x + radius) + 1, self._width_px)
        row_start = max(math.floor((center_y - radius) / self._aspect), 0)
        row_stop = min(math.floor((center_y + radius) / self._aspect) + 1, self._height_px)

        if column_start >= column_stop or row_start >= row_stop:
            return

        box = (slice(row_start, row_stop), slice(column_start, column_stop))
        is_uncovered = self._is_uncovered[box]

        if not is_uncovered.any():
            return

        dx = numpy.arange(column_start, column_stop)[None, :, None] + self._offset_x[box]
        dx -= center_x
        dy = (numpy.arange(row_start, row_stop)[:, None, None] + self._offset_y[box]) * self._aspect
        dy -= center_y
        is_claimed = is_uncovered & (dx * dx + dy * dy <= radius * radius)
        num_claimed = numpy.count_nonzero(is_claimed, axis=-1)
        self._pixel_sum[box] += value * num_claimed
        is_uncovered &= ~is_claimed
        self._num_uncovered -= int(num_claimed.sum())

    def to_array(self) -> ComplexArrayType:
        return self._pixel_sum / self._num_subsamples


def generate_dead_leaves_object(
    rng: numpy.random.Generator,
    geometry: ObjectGeometry,
    *,
    leaf_radius_lower_px: float,
    leaf_radius_upper_px: float,
    leaf_radius_power_law_exponent: float,
    leaf_amplitude_lower: float,
    leaf_amplitude_upper: float,
    leaf_phase_lower_rad: float,
    leaf_phase_upper_rad: float,
    num_subsamples_per_axis: int = 4,
) -> Object:
    """Generate a complex object from the stationary dead-leaves model.

    Leaves are disks dropped by a Poisson process, each hiding whatever lies beneath it.
    Radii have density proportional to ``r**-leaf_radius_power_law_exponent`` on
    ``[leaf_radius_lower_px, leaf_radius_upper_px]``, in pixel widths, and an exponent of
    three makes the field scale invariant. Leaves are circular in physical space, also on
    non-square pixels. Each leaf's transmission has a phase uniform on ``[leaf_phase_lower_rad,
    leaf_phase_upper_rad)`` and an amplitude whose square is uniform on
    ``[leaf_amplitude_lower**2, leaf_amplitude_upper**2)``.

    The sample is exact, not an approximation that improves with more leaves: leaves are
    painted front to back, so the field seen through the window is final as soon as the
    window is covered, and once few points remain uncovered only leaves that cover one
    are drawn. Each pixel is the mean transmission over ``num_subsamples_per_axis**2``
    stratified, jittered points, the box-filtered sample of the continuous field. Memory
    grows as about ``9 * num_subsamples_per_axis**2`` bytes per pixel.
    """
    if leaf_radius_lower_px <= 0.0:
        raise ValueError('leaf_radius_lower_px must be strictly positive')

    if leaf_radius_lower_px >= leaf_radius_upper_px:
        raise ValueError('leaf_radius_lower_px must be less than leaf_radius_upper_px')

    if leaf_amplitude_lower >= leaf_amplitude_upper:
        raise ValueError('leaf_amplitude_lower must be less than leaf_amplitude_upper')

    if leaf_phase_lower_rad >= leaf_phase_upper_rad:
        raise ValueError('leaf_phase_lower_rad must be less than leaf_phase_upper_rad')

    if num_subsamples_per_axis < 1:
        raise ValueError('num_subsamples_per_axis must be at least one')

    pixel_geometry = geometry.get_pixel_geometry()
    canvas = _DeadLeavesCanvas(
        rng,
        width_px=geometry.width_px,
        height_px=geometry.height_px,
        aspect=pixel_geometry.height_m / pixel_geometry.width_m,
        num_subsamples_per_axis=num_subsamples_per_axis,
    )
    canvas.fill(
        rng,
        _LeafMarks(
            amplitude_lower=leaf_amplitude_lower,
            amplitude_upper=leaf_amplitude_upper,
            phase_lower_rad=leaf_phase_lower_rad,
            phase_upper_rad=leaf_phase_upper_rad,
        ),
        radius_lower_px=leaf_radius_lower_px,
        radius_upper_px=leaf_radius_upper_px,
        radius_exponent=leaf_radius_power_law_exponent,
    )
    logger.debug(
        f'Dead leaves: {canvas.num_leaves} leaves, the last {canvas.num_tail_leaves} '
        f'from {canvas.num_tail_proposals} proposals.'
    )

    return Object(
        array=canvas.to_array(),
        pixel_geometry=pixel_geometry,
        center=geometry.get_center(),
    )


def generate_siemens_star_object(
    rng: numpy.random.Generator,
    geometry: ObjectGeometry,
    *,
    num_spokes: int,
    outer_radius_fraction: float,
    spoke_amplitude: float,
    background_amplitude: float,
    spoke_phase_rad: float,
    background_phase_rad: float,
    num_subsamples_per_axis: int = 4,
) -> Object:
    """Generate a Siemens star resolution target with alternating angular wedges.

    ``2 * num_spokes`` wedges alternate between the spoke and background values inside a
    disk whose radius is ``outer_radius_fraction`` of the canvas's smaller physical
    half-extent, centered on the canvas; outside the disk is background. The star is
    circular in physical space, also on non-square pixels. Each pixel is the mean
    transmission over ``num_subsamples_per_axis**2`` stratified points, one uniformly
    jittered point per cell, so edges and the singular center are box filtered without
    a fixed sampling pattern; the result is reproducible for a given *rng*.
    """
    if num_spokes < 2:
        raise ValueError('num_spokes must be at least 2!')

    if outer_radius_fraction <= 0.0:
        raise ValueError('outer_radius_fraction must be strictly positive!')

    if num_subsamples_per_axis < 1:
        raise ValueError('num_subsamples_per_axis must be at least one!')

    pixel_geometry = geometry.get_pixel_geometry()
    aspect = pixel_geometry.height_m / pixel_geometry.width_m
    width = float(geometry.width_px)
    height = geometry.height_px * aspect
    outer_radius = outer_radius_fraction * min(width, height) / 2.0
    spoke_value = spoke_amplitude * numpy.exp(1j * spoke_phase_rad)
    background_value = background_amplitude * numpy.exp(1j * background_phase_rad)

    shape = (geometry.height_px, geometry.width_px)
    rows, columns = numpy.indices(shape, dtype=float)
    pixel_sum: ComplexArrayType = numpy.zeros(shape, dtype=complex)

    for cell in range(num_subsamples_per_axis**2):
        cell_y, cell_x = divmod(cell, num_subsamples_per_axis)
        jitter = rng.uniform(size=(2, *shape))
        dx = columns + (cell_x + jitter[0]) / num_subsamples_per_axis - 0.5 * width
        dy = (rows + (cell_y + jitter[1]) / num_subsamples_per_axis) * aspect - 0.5 * height
        angle = numpy.mod(numpy.arctan2(dy, dx), TWO_PI)
        wedge_index = numpy.floor(angle * num_spokes / PI).astype(int)
        is_spoke = (numpy.hypot(dx, dy) <= outer_radius) & (wedge_index % 2 == 0)
        pixel_sum += numpy.where(is_spoke, spoke_value, background_value)

    return Object(
        array=pixel_sum[numpy.newaxis] / num_subsamples_per_axis**2,
        pixel_geometry=pixel_geometry,
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
