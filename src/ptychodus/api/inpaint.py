"""Biharmonic inpainting of bad detector pixels, factorized once per mask."""

from __future__ import annotations
from typing import cast, overload

import numpy
from scipy import ndimage, sparse
from scipy.sparse.linalg import splu

from .diffraction import BadPixels, DiffractionPattern, DiffractionPatterns
from .typing import IntegerArrayType, RealArrayType

__all__ = [
    'BiharmonicInpainter',
]


def _biharmonic_stencil(
    shape: tuple[int, int], center: tuple[int, int]
) -> tuple[IntegerArrayType, IntegerArrayType, RealArrayType]:
    """The nonzero rows, columns and weights of the biharmonic stencil on a neighbourhood.

    The stencil is the Laplacian applied twice to a unit impulse at *center*, on a
    neighbourhood of *shape* that the image border may have truncated, which is how
    :func:`skimage.restoration.inpaint_biharmonic` builds it.
    """
    stencil = numpy.zeros(shape, dtype=numpy.float64)
    stencil[center] = 1.0
    stencil = ndimage.laplace(ndimage.laplace(stencil))
    rows, columns = numpy.nonzero(stencil)
    return rows, columns, stencil[rows, columns]


class BiharmonicInpainter:
    """Fills the bad pixels of a fixed mask by biharmonic inpainting.

    Only interior holes are inpainted. A bad region touching the array border is zeroed
    instead, because filling it would extrapolate beyond every measured pixel rather than
    interpolate between them. Regions are found with 4-connectivity.

    Calling it gives the same result as :func:`skimage.restoration.inpaint_biharmonic`
    on the interior holes, followed by zeroing the border regions: every pixel outside
    an interior hole counts as known, border regions included, and each pattern's fill is
    clipped to the range of those pixels. The sparse system depends only on the mask, so
    it is built and factorized here once. A call then solves for every pattern together,
    reading only the known pixels the stencils reach.
    """

    def __init__(self, bad_pixels: BadPixels) -> None:
        self._shape = bad_pixels.shape
        height, width = bad_pixels.shape

        labels, _ = ndimage.label(bad_pixels)
        border_labels = numpy.unique(
            numpy.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
        self._edge_pixels = numpy.isin(labels, border_labels[border_labels != 0])
        interior_pixels = numpy.logical_and(bad_pixels, numpy.logical_not(self._edge_pixels))
        self._known_pixels = numpy.logical_not(interior_pixels).reshape(-1)
        self._unknown = numpy.flatnonzero(interior_pixels)
        self._band = numpy.zeros(0, dtype=numpy.intp)
        self._band_stencils: sparse.csr_array | None = None
        self._solver: sparse.linalg.SuperLU | None = None

        if self._unknown.size == 0:
            return

        radius = 2  # The reach of the biharmonic stencil.
        point_rows, point_columns = numpy.divmod(self._unknown, width)
        near_border = (
            (point_rows < radius)
            | (point_rows >= height - radius)
            | (point_columns < radius)
            | (point_columns >= width - radius)
        )

        # Away from the border every point takes the full stencil, placed by broadcasting.
        diameter = 2 * radius + 1
        stencil_rows, stencil_columns, stencil_weights = _biharmonic_stencil(
            (diameter, diameter), (radius, radius)
        )
        full = numpy.flatnonzero(numpy.logical_not(near_border))
        entry_rows = [numpy.repeat(full, stencil_weights.size)]
        entry_columns = [
            (
                (point_rows[full, numpy.newaxis] + stencil_rows - radius) * width
                + point_columns[full, numpy.newaxis]
                + stencil_columns
                - radius
            ).reshape(-1)
        ]
        entry_weights = [numpy.tile(stencil_weights, full.size)]

        # Near the border the stencil is truncated, so each point takes its own.
        for point in numpy.flatnonzero(near_border):
            row = int(point_rows[point])
            column = int(point_columns[point])
            row_lo = max(row - radius, 0)
            column_lo = max(column - radius, 0)
            rows, columns, weights = _biharmonic_stencil(
                (
                    min(row + radius + 1, height) - row_lo,
                    min(column + radius + 1, width) - column_lo,
                ),
                (row - row_lo, column - column_lo),
            )
            entry_rows.append(numpy.full(weights.size, point))
            entry_columns.append((rows + row_lo) * width + columns + column_lo)
            entry_weights.append(weights)

        all_columns = numpy.concatenate(entry_columns)
        stencils = sparse.csr_array(
            (numpy.concatenate(entry_weights), (numpy.concatenate(entry_rows), all_columns)),
            shape=(self._unknown.size, height * width),
        )
        self._band = numpy.setdiff1d(numpy.unique(all_columns), self._unknown, assume_unique=True)
        self._band_stencils = sparse.csr_array(stencils[:, self._band])
        self._solver = splu(sparse.csc_array(stencils[:, self._unknown]), permc_spec='MMD_ATA')

    @overload
    def __call__(
        self, data: DiffractionPattern, *, overwrite_input: bool = False
    ) -> DiffractionPattern: ...

    @overload
    def __call__(
        self, data: DiffractionPatterns, *, overwrite_input: bool = False
    ) -> DiffractionPatterns: ...

    def __call__(
        self, data: DiffractionPattern | DiffractionPatterns, *, overwrite_input: bool = False
    ) -> DiffractionPattern | DiffractionPatterns:
        """Return *data* inpainted, in the shape and dtype it was given.

        Takes a single pattern or a stack sharing the mask. Integer values are rounded
        rather than truncated, which would bias every filled pixel low.

        With *overwrite_input*, a writeable C-contiguous float64 *data* is filled in
        place and returned, sparing a copy; other data is copied as usual, so the
        caller must use the return value either way.

        Raises:
            ValueError: if *data* is not 2- or 3-dimensional or its patterns do not match
                the mask.
        """
        if data.ndim not in (2, 3):
            raise ValueError('Diffraction data must be 2- or 3-dimensional ndarray.')

        if data.shape[-2:] != self._shape:
            raise ValueError(
                f'Patterns of shape {data.shape[-2:]} do not match the {self._shape} mask!'
            )

        can_overwrite = (
            overwrite_input
            and data.dtype == numpy.float64
            and data.flags.c_contiguous
            and data.flags.writeable
        )
        filled = data if can_overwrite else data.astype(numpy.float64, order='C')

        if self._solver is not None and self._band_stencils is not None:
            # `filled` is C-contiguous, so this is a view and writes land in it.
            patterns = filled.reshape(-1, self._known_pixels.size)
            values = self._solver.solve(-(self._band_stencils @ patterns[:, self._band].T)).T
            lower = numpy.min(patterns, axis=-1, where=self._known_pixels, initial=numpy.inf)
            upper = numpy.max(patterns, axis=-1, where=self._known_pixels, initial=-numpy.inf)
            patterns[:, self._unknown] = numpy.clip(
                values, lower[:, numpy.newaxis], upper[:, numpy.newaxis]
            )

        filled[..., self._edge_pixels] = 0.0
        dtype = data.dtype

        if numpy.issubdtype(dtype, numpy.integer):
            # Fills are already confined to the range of each pattern's known pixels, so
            # they are representable; the clip costs one pass and keeps the cast from
            # wrapping silently if that ever stops being true.
            # issubdtype does not narrow the pattern dtype union for the checker.
            iinfo = numpy.iinfo(cast(numpy.dtype[numpy.integer], dtype))
            filled = numpy.clip(numpy.rint(filled), iinfo.min, iinfo.max)

        return filled.astype(dtype, copy=False)
