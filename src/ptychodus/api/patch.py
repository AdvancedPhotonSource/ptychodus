"""Placement of a probe-sized patch on the object canvas.

There are two ways to put a probe-sized window on an object canvas at a scan
position that falls between pixels, and they are not interchangeable.

**Shift the probe** -- :func:`iter_probe_patches` and
:func:`accumulate_probe_intensity`. The window is an exact-width integer slice of
the object and the residual sub-pixel offset is applied to the probe by a Fourier
shift, so the object array is read or accumulated on its own grid and is never
interpolated. This is the right convention whenever the object is the quantity
being solved for or integrated over: a forward model, an illumination canvas, an
error map splatted back into object space.

**Resample the object** -- :class:`ptychodus.api.interpolate.BarycentricArrayInterpolator`.
The object itself is bilinearly resampled at the fractional position, so the patch
lands on the probe's own grid. This is the right convention when the patch must be
a fixed-size array in its own frame, or when an operator's adjoint has to be the
exact transpose of its gather. The cost is that bilinear resampling is a low-pass
filter: it is not invertible, and it damps the phase excursions of a transmission
function that a Fourier shift carries exactly.

Both place the probe's beam axis, pixel ``N // 2``, on the scan position, as
:meth:`ptychodus.api.probe.ProbeGeometry.resolve_patch_bounds` does, and both
refuse a window that leaves the canvas rather than returning a truncated or wrapped
one.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload

import numpy

from .fourier import fourier_shift_2d
from .object import ObjectGeometry
from .probe import PatchBounds, ProbeEnsemble, ProbeGeometry
from .probe_positions import ProbePosition
from .product import Product
from .typing import ComplexArrayType, IntegerArrayType, RealArrayType

__all__ = [
    'ProbePatch',
    'accumulate_patches_on',
    'accumulate_probe_intensity',
    'accumulate_probe_intensity_on',
    'compute_patch_pixel_indexes',
    'iter_probe_patches',
    'iter_probe_patches_on',
    'resolve_probe_patch_bounds_on',
    'shift_probe_onto_patches',
]


@dataclass(frozen=True)
class ProbePatch:
    """One scan position's object-canvas placement and its sub-pixel-shifted probe."""

    position: ProbePosition
    bounds: PatchBounds
    modes: ComplexArrayType
    """Shape ``(num_incoherent_modes, height_px, width_px)``, Fourier-shifted by the
    residual sub-pixel offset so the modes sit on the object pixel grid."""

    def get_intensity(self) -> RealArrayType:
        """Incoherent-mode intensity sum, shape ``(height_px, width_px)``."""
        return numpy.sum(numpy.abs(self.modes) ** 2, axis=0)

    def get_normalized_intensity(self) -> RealArrayType:
        """:meth:`get_intensity` scaled to sum to one.

        The patch becomes a distribution over canvas pixels rather than a photon
        count. A patch carrying no intensity is returned unchanged.
        """
        intensity = self.get_intensity()
        total = intensity.sum()
        return intensity / total if total > 0.0 else intensity


def _validate_patch_bounds(bounds: PatchBounds, geometry: ObjectGeometry, index: int) -> None:
    rows_fit = 0 <= bounds.y_slice.start and bounds.y_slice.stop <= geometry.height_px
    columns_fit = 0 <= bounds.x_slice.start and bounds.x_slice.stop <= geometry.width_px

    if not (rows_fit and columns_fit):
        raise ValueError(
            f'Probe patch for scan index {index} falls outside the object canvas!'
            f' It needs rows [{bounds.y_slice.start}, {bounds.y_slice.stop}) and columns'
            f' [{bounds.x_slice.start}, {bounds.x_slice.stop}) of a'
            f' {geometry.height_px} x {geometry.width_px} canvas.'
        )


def _resolve_patch_bounds(
    probe_geometry: ProbeGeometry, scan_point: ProbePosition, object_geometry: ObjectGeometry
) -> PatchBounds:
    object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
    bounds = probe_geometry.resolve_patch_bounds(object_point.x_px, object_point.y_px)
    _validate_patch_bounds(bounds, object_geometry, scan_point.index)
    return bounds


def resolve_probe_patch_bounds_on(
    probe_geometry: ProbeGeometry,
    positions: Sequence[ProbePosition],
    object_geometry: ObjectGeometry,
) -> list[PatchBounds]:
    """Return the object-canvas placement of every scan position, in order.

    The placement :func:`iter_probe_patches_on` yields, without shifting a probe: each
    bounds holds exact-width integer slices and the sub-pixel residual a Fourier shift
    of the probe must apply.

    Raises:
        ValueError: for a position whose footprint leaves the canvas.
    """
    return [_resolve_patch_bounds(probe_geometry, point, object_geometry) for point in positions]


def shift_probe_onto_patches(
    probe_modes: ComplexArrayType, bounds: Sequence[PatchBounds]
) -> ComplexArrayType:
    """Return the probe shifted onto every placement at once, ``(n, modes, height, width)``.

    Each placement's modes carry its sub-pixel residual, as :func:`iter_probe_patches_on`
    shifts them one position at a time; *probe_modes* is ``(modes, height, width)``.
    """
    dx = numpy.array([placement.dx for placement in bounds])[:, None, None, None]
    dy = numpy.array([placement.dy for placement in bounds])[:, None, None, None]
    return fourier_shift_2d(probe_modes[numpy.newaxis], dx, dy)


def compute_patch_pixel_indexes(
    bounds: Sequence[PatchBounds], object_geometry: ObjectGeometry
) -> IntegerArrayType:
    """Return the row-major canvas index of every patch pixel, ``(n, height, width)``.

    ``canvas.ravel()[indexes]`` gathers every patch at once, and
    :func:`accumulate_patches_on` scatters patches back. The bounds must share one
    extent, as those of one probe do.
    """
    if not bounds:
        return numpy.zeros((0, 0, 0), dtype=numpy.intp)

    first = bounds[0]
    height_px = first.y_slice.stop - first.y_slice.start
    width_px = first.x_slice.stop - first.x_slice.start
    starts = numpy.array(
        [
            placement.y_slice.start * object_geometry.width_px + placement.x_slice.start
            for placement in bounds
        ],
        dtype=numpy.intp,
    )
    offsets = (
        numpy.arange(height_px, dtype=numpy.intp)[:, None] * object_geometry.width_px
        + numpy.arange(width_px, dtype=numpy.intp)[None, :]
    )
    return starts[:, None, None] + offsets


@overload
def accumulate_patches_on(
    patches: RealArrayType, pixel_indexes: IntegerArrayType, object_geometry: ObjectGeometry
) -> RealArrayType: ...


@overload
def accumulate_patches_on(
    patches: ComplexArrayType, pixel_indexes: IntegerArrayType, object_geometry: ObjectGeometry
) -> ComplexArrayType: ...


def accumulate_patches_on(
    patches: RealArrayType | ComplexArrayType,
    pixel_indexes: IntegerArrayType,
    object_geometry: ObjectGeometry,
) -> RealArrayType | ComplexArrayType:
    """Sum *patches* onto a zero canvas at *pixel_indexes*, ``(height_px, width_px)``.

    The scatter that undoes the gather :func:`compute_patch_pixel_indexes` describes:
    overlapping pixels add. Real patches give a float64 canvas, complex ones complex128.
    """
    num_pixels = object_geometry.height_px * object_geometry.width_px
    flat_indexes = pixel_indexes.ravel()
    flat_patches = patches.ravel()
    # Weighted, bincount sums in float64; the stubs promise only its unweighted counts.
    real = numpy.asarray(
        numpy.bincount(flat_indexes, weights=flat_patches.real, minlength=num_pixels),
        dtype=numpy.float64,
    )
    canvas_shape = (object_geometry.height_px, object_geometry.width_px)

    if numpy.iscomplexobj(patches):
        imaginary = numpy.bincount(flat_indexes, weights=flat_patches.imag, minlength=num_pixels)
        return (real + 1j * imaginary).reshape(canvas_shape)
    else:
        return real.reshape(canvas_shape)


def iter_probe_patches_on(
    probes: ProbeEnsemble,
    positions: Sequence[ProbePosition],
    object_geometry: ObjectGeometry,
) -> Iterator[ProbePatch]:
    """Yield the object-canvas placement and sub-pixel-shifted probe of every scan position.

    The bounds are exact-width integer slices, so a patch can be read or accumulated
    with a plain assignment and the object array is never interpolated. The probe at
    each position is paired as :meth:`ProbeEnsemble.iter_position_probes` pairs it.

    Raises:
        ValueError: for a position whose footprint leaves the canvas. The slices are
            unclipped, and an unchecked negative lower bound is a valid Python slice
            that wraps around to the far edge of the array.
    """
    probe_geometry = probes.get_geometry()

    for scan_point, probe in probes.iter_position_probes(positions):
        bounds = _resolve_patch_bounds(probe_geometry, scan_point, object_geometry)

        yield ProbePatch(
            position=scan_point,
            bounds=bounds,
            modes=fourier_shift_2d(probe.get_array(), dx=bounds.dx, dy=bounds.dy),
        )


def iter_probe_patches(product: Product) -> Iterator[ProbePatch]:
    """:func:`iter_probe_patches_on` over the product's probe, positions and object canvas."""
    return iter_probe_patches_on(
        product.probes, product.probe_positions, product.object_.get_geometry()
    )


def _resolve_weight_rows(
    weights: RealArrayType | Sequence[RealArrayType] | None, num_positions: int
) -> tuple[RealArrayType | None, bool]:
    """Normalize the weight argument to a ``(K, N)`` array plus whether to unwrap."""
    if weights is None:
        return None, True

    rows = numpy.asarray(weights, dtype=float)
    is_single = rows.ndim == 1

    if is_single:
        rows = rows[numpy.newaxis, :]

    if rows.ndim != 2:
        raise ValueError(f'Weights must be 1-D or 2-D; got shape {rows.shape}!')

    if rows.shape[-1] != num_positions:
        raise ValueError(
            f'Weights describe {rows.shape[-1]} position(s) but the product has {num_positions}!'
        )

    return rows, is_single


def accumulate_probe_intensity_on(
    probes: ProbeEnsemble,
    positions: Sequence[ProbePosition],
    object_geometry: ObjectGeometry,
    *,
    weights: RealArrayType | Sequence[RealArrayType] | None = None,
    normalize: bool = False,
) -> RealArrayType:
    """Sum every scan position's probe intensity onto the object canvas.

    This is the probe intensity cross-correlated with the scan positions, so it is
    expressed as one accumulation rather than as per-position patches: the result
    depends on the positions only through where each contribution lands.

    Args:
        probes: The probe, paired with each position as for :func:`iter_probe_patches_on`.
        positions: The scan positions.
        object_geometry: The canvas.
        weights: Per-position scale factors, ``(N,)`` for one canvas or ``(K, N)`` for
            ``K`` canvases accumulated in a single pass. The default weights every
            position equally.
        normalize: Scale each position's intensity to sum to one before weighting, so
            a contribution is a distribution over canvas pixels rather than a photon
            count.

    Returns:
        ``(height_px, width_px)``, or ``(K, height_px, width_px)`` when ``weights`` is
        two-dimensional.

    Raises:
        ValueError: when ``weights`` does not describe every position, or when a
            position's footprint leaves the canvas.
    """
    canvas_shape = (object_geometry.height_px, object_geometry.width_px)
    weight_rows, is_single = _resolve_weight_rows(weights, len(positions))
    num_canvases = 1 if weight_rows is None else weight_rows.shape[0]
    canvases: RealArrayType = numpy.zeros((num_canvases, *canvas_shape))

    for index, probe_patch in enumerate(iter_probe_patches_on(probes, positions, object_geometry)):
        patch = probe_patch.get_normalized_intensity() if normalize else probe_patch.get_intensity()
        bounds = probe_patch.bounds

        if weight_rows is None:
            canvases[0, bounds.y_slice, bounds.x_slice] += patch
        else:
            for canvas, weight in zip(canvases, weight_rows[:, index]):
                canvas[bounds.y_slice, bounds.x_slice] += weight * patch

    return canvases[0] if is_single else canvases


def accumulate_probe_intensity(
    product: Product,
    *,
    weights: RealArrayType | Sequence[RealArrayType] | None = None,
    normalize: bool = False,
) -> RealArrayType:
    """:func:`accumulate_probe_intensity_on` over the product's probe, positions and canvas."""
    return accumulate_probe_intensity_on(
        product.probes,
        product.probe_positions,
        product.object_.get_geometry(),
        weights=weights,
        normalize=normalize,
    )
