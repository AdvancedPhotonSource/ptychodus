"""Sub-pixel array interpolation utilities."""

from typing import Final, Generic, TypeVar, overload
import math

import numpy.typing
import numpy
from scipy.ndimage import affine_transform, map_coordinates

from ptychodus.api.typing import (
    BooleanArrayType,
    ComplexArrayType,
    InexactArrayType,
    IntegerArrayType,
    RealArrayType,
)

__all__ = [
    'BarycentricArrayInterpolator',
    'interpolate_at_indexes',
    'lerp',
    'resample_complex_at',
    'resample_complex_on_grid',
]


_InexactT = TypeVar('_InexactT', bound=numpy.inexact)

# An offset this close to a whole pixel is treated as landing on it. A coordinate that
# is an exact integer in real arithmetic reaches here a few ULP away from one, and
# keeping that residue would demand an interpolation sample the array is not required
# to hold. Five orders above worst-case double round-off on a canvas of ~1e4 pixels and
# five below any physically meaningful sub-pixel shift, so the exact value is immaterial;
# it would narrow on a canvas beyond ~1e7 pixels.
_INTEGER_SNAP_TOLERANCE_PX: Final[float] = 1.0e-9


@overload
def lerp(lower: float, upper: float, frac: float) -> float:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: complex, upper: complex, frac: float) -> complex:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: RealArrayType, upper: RealArrayType, frac: float) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: RealArrayType, upper: RealArrayType, frac: RealArrayType) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


@overload
def lerp(lower: float, upper: float, frac: RealArrayType) -> RealArrayType:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    ...


def lerp(
    lower: InexactArrayType | complex,
    upper: InexactArrayType | complex,
    frac: RealArrayType | float,
) -> InexactArrayType | complex:
    """Linearly interpolate between *lower* and *upper* by fraction *frac* in [0, 1]."""
    return (1.0 - frac) * lower + frac * upper


def _grid_scale_and_offset(indexes: RealArrayType) -> tuple[float, float]:
    """The scale and offset of a uniform 1-D index grid, ``indexes[i] = scale * i + offset``."""
    scale = float(indexes[1] - indexes[0]) if indexes.size > 1 else 1.0
    return scale, float(indexes[0])


def resample_complex_on_grid(
    array: ComplexArrayType,
    rows: RealArrayType,
    columns: RealArrayType,
    *,
    order: int,
    fill_value: complex = 0j,
) -> ComplexArrayType:
    """Resample the last two axes of a complex array onto an axis-aligned grid.

    *rows* and *columns* are the fractional source indexes each target row and column
    samples. Both must be uniform -- a pitch change, a shift, a crop or a pad -- so the
    target is a tensor product of two 1-D grids and every leading slice (layers, modes) is
    one diagonal :func:`scipy.ndimage.affine_transform`, with no coordinate arrays built.
    The axis-aligned counterpart of :func:`resample_complex_at`, which takes scattered
    points.

    The real and imaginary parts are each a B-spline of *order*, so the result is linear in
    the field: it commutes with a global phase, and resampled probe modes still mix by
    their OPR weights as the originals do. A cubic spline overshoots by a few percent at a
    hard edge, which a band-limited field does not have.

    The spline is fitted with the source mirrored at its edge, and a target point whose
    source index lies outside ``[0, N - 1]`` on either axis takes *fill_value* exactly. A
    spline's prefilter is not local, so fitting it through the fill instead would leak the
    fill several pixels into the field; mirroring keeps the interior from ever seeing it.

    A gather only: it has no adjoint.
    """
    row_scale, row_offset = _grid_scale_and_offset(rows)
    column_scale, column_offset = _grid_scale_and_offset(columns)
    output_shape = (rows.size, columns.size)
    outside = ~numpy.outer(
        (rows >= 0.0) & (rows <= array.shape[-2] - 1),
        (columns >= 0.0) & (columns <= array.shape[-1] - 1),
    )
    resampled = numpy.empty((*array.shape[:-2], *output_shape), dtype=numpy.result_type(array, 1j))

    for index in numpy.ndindex(array.shape[:-2]):
        plane = array[index]
        parts = []

        for values in (plane.real, plane.imag):
            parts.append(
                affine_transform(
                    values,
                    numpy.array([row_scale, column_scale]),
                    offset=(row_offset, column_offset),
                    output_shape=output_shape,
                    order=order,
                    mode='mirror',
                )
            )

        resampled[index] = numpy.where(outside, fill_value, parts[0] + 1j * parts[1])

    return resampled


def resample_complex_at(
    array: ComplexArrayType,
    row: RealArrayType,
    column: RealArrayType,
    *,
    order: int,
    mode: str = 'grid-constant',
) -> ComplexArrayType:
    """Sample a 2D complex array at fractional ``(row, column)`` indexes, as zero beyond its edge.

    The real and imaginary parts are each a B-spline of *order* through
    :func:`scipy.ndimage.map_coordinates`. That makes the result linear in the field: it
    commutes with a global phase, so it does not depend on the arbitrary phase a
    reconstruction settles on, and resampled probe modes still mix by their OPR weights as
    the originals do. *row* and *column* may have any shape; the result takes it.

    *mode* is how the edge is treated. ``'grid-constant'`` interpolates toward the zeros
    beyond the array, so a field fades out across the edge -- over a pixel at order 1, and
    over a geometrically decaying tail of the spline's prefilter at higher orders.
    ``'constant'`` interpolates only within the array and returns exactly zero outside it.

    A gather only: it has no adjoint. Where an operator's transpose must be exact, use
    :class:`BarycentricArrayInterpolator`.
    """
    coordinates = (row, column)
    real = map_coordinates(array.real, coordinates, order=order, mode=mode, cval=0.0)
    imaginary = map_coordinates(array.imag, coordinates, order=order, mode=mode, cval=0.0)
    return real + 1j * imaginary


def interpolate_at_indexes(
    source_indexes: IntegerArrayType,
    values: RealArrayType,
    target_indexes: IntegerArrayType,
) -> tuple[BooleanArrayType, RealArrayType]:
    """Resample values keyed by integer index onto other indexes.

    Values sharing a source index are averaged into one anchor. A target inside the
    closed range of the anchor indexes takes the anchor value where one sits at its
    index and is linearly interpolated between the neighbouring anchors otherwise. A
    target outside that range is not extrapolated; it is left out. A NaN anchor value
    propagates into every target interpolated from it.

    *values* holds one row per source index: shape ``(n,)``, or ``(n, k)`` for ``k``
    columns resampled independently.

    Returns the mask over *target_indexes* marking those in range, and the values at
    them in target order, shaped ``(m,)`` or ``(m, k)`` to match *values*. An empty
    source leaves every target out of range.

    Raises:
        ValueError: if *values* does not have one row per source index.
    """
    if values.shape[0] != source_indexes.size:
        raise ValueError(
            f'Expected one row of values per source index; got {values.shape[0]} rows'
            f' for {source_indexes.size} indexes.'
        )

    if source_indexes.size == 0:
        return (
            numpy.zeros(target_indexes.shape, dtype=numpy.bool),
            numpy.empty((0, *values.shape[1:]), dtype=numpy.float64),
        )

    # numpy.unique sorts ascending, which is the order numpy.interp needs its anchors in.
    unique_indexes, inverse = numpy.unique(source_indexes, return_inverse=True)
    inverse = inverse.ravel()
    counts = numpy.bincount(inverse)

    in_range = (target_indexes >= unique_indexes[0]) & (target_indexes <= unique_indexes[-1])
    in_range_targets = target_indexes[in_range]

    columns = values.reshape(values.shape[0], -1)
    resampled = numpy.empty((in_range_targets.size, columns.shape[1]), dtype=numpy.float64)

    for column in range(columns.shape[1]):
        anchor_values = numpy.bincount(inverse, weights=columns[:, column]) / counts
        # numpy.interp returns the anchor value exactly at a coincident index, and the
        # range mask above rules out the extrapolation it would otherwise clamp.
        resampled[:, column] = numpy.interp(in_range_targets, unique_indexes, anchor_values)

    return in_range, resampled.reshape((in_range_targets.size, *values.shape[1:]))


def _calculate_support_frac(x: float, n: int, extent: int) -> tuple[slice, float]:
    """Return the integer support slice and sub-pixel offset for a width-``n`` window centered at ``x`` on an axis of ``extent`` samples.

    The centered-pixel convention places the window's array-center pixel at
    ``(n - 1) / 2``, matching
    :meth:`ptychodus.api.object.ObjectGeometry.map_coordinates_probe_to_object`.
    Bilinear interpolation across the window reads one sample past its far edge, so
    the support spans ``n + 1`` samples -- except at a zero offset, where that sample
    carries zero weight and the support is exactly ``n`` wide. An offset within
    ``_INTEGER_SNAP_TOLERANCE_PX`` of a whole pixel is taken as zero.

    Raises:
        ValueError: when the support leaves ``[0, extent)``. Such a window is an
            inconsistency between the array and the coordinates addressing it rather
            than something to clip: a clipped window comes back silently undersized,
            and a negative lower bound is a valid Python slice that wraps around to
            the far end of the axis.
    """
    lower = x - (n - 1) / 2
    whole = math.floor(lower)
    frac = lower - whole

    if frac > 1.0 - _INTEGER_SNAP_TOLERANCE_PX:
        whole += 1
        frac = 0.0
    elif frac < _INTEGER_SNAP_TOLERANCE_PX:
        frac = 0.0

    stop = whole + n + 1 if frac > 0.0 else whole + n

    if whole < 0 or stop > extent:
        raise ValueError(
            f'A width-{n} window centered at {x} needs samples [{whole}, {stop}) '
            f'of an axis holding {extent}!'
        )

    return slice(whole, stop), frac


def _axis_terms(frac: float) -> tuple[tuple[slice, float], ...]:
    """Bilinear scatter terms along one axis, as ``(support slice, weight)`` pairs.

    A zero offset leaves a single unit-weight term over an ``n``-wide support; the
    far-side term it drops would have carried zero weight.
    """
    if frac > 0.0:
        return (slice(None, -1), 1.0 - frac), (slice(1, None), frac)

    return ((slice(None), 1.0),)


class BarycentricArrayInterpolator(Generic[_InexactT]):
    """Extract patches from an array using bilinear (barycentric) sub-pixel interpolation."""

    def __init__(self, array: numpy.typing.NDArray[_InexactT]) -> None:
        super().__init__()
        self._array = array

    def get_patch(
        self, center_x: float, center_y: float, width: int, height: int
    ) -> numpy.typing.NDArray[_InexactT]:
        x_support, x_frac = _calculate_support_frac(center_x, width, self._array.shape[-1])
        y_support, y_frac = _calculate_support_frac(center_y, height, self._array.shape[-2])

        support = self._array[..., y_support, x_support]

        if y_frac > 0.0:
            # separable bilinear as fused in-place lerp: lower + frac * (upper - lower)
            y_interp = numpy.subtract(support[..., 1:, :], support[..., :-1, :])
            y_interp *= y_frac
            y_interp += support[..., :-1, :]
        else:
            y_interp = support

        if x_frac > 0.0:
            patch = numpy.subtract(y_interp[..., :, 1:], y_interp[..., :, :-1])
            patch *= x_frac
            patch += y_interp[..., :, :-1]
        else:
            # support is a view of the array; a gather always returns its own storage
            patch = y_interp.copy()

        return patch

    def add_patch(
        self,
        center_x: float,
        center_y: float,
        patch: numpy.typing.NDArray[_InexactT],
    ) -> None:
        """Bilinear-scatter *patch* back into the underlying array (transpose of get_patch)."""
        x_support, x_frac = _calculate_support_frac(
            center_x, patch.shape[-1], self._array.shape[-1]
        )
        y_support, y_frac = _calculate_support_frac(
            center_y, patch.shape[-2], self._array.shape[-2]
        )

        support = self._array[..., y_support, x_support]
        scratch = numpy.empty(patch.shape, dtype=patch.dtype)

        for y_term, y_weight in _axis_terms(y_frac):
            for x_term, x_weight in _axis_terms(x_frac):
                numpy.multiply(patch, y_weight * x_weight, out=scratch)
                support[..., y_term, x_term] += scratch
