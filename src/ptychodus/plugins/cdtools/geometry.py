"""Carry a product into the pixel frame cdtools' FancyPtycho places probes in, and back.

Pure numpy, so the mapping is testable without cdtools or torch installed.

FancyPtycho places each probe window by the **top-left corner** of the object array it
reads, ``translations_to_pixel(obj_basis, t) - min_translation`` in ``(row, column)``
pixels, cutting the window at the floor of that corner and Fourier-shifting the probe by
the fractional remainder. :meth:`ptychodus.api.probe.ProbeGeometry.resolve_patch_bounds`
places a patch the same way -- an integer lower corner, then a positive sub-pixel shift
of the probe -- from a corner at ``center - (N - 1) / 2``, so a corner computed here
puts every probe exactly where :func:`ptychodus.api.patch.iter_probe_patches` does.

Translations are passed relative to the object's center. cdtools holds them in single
precision, and an absolute stage coordinate of a few millimetres would cost a visible
fraction of a pixel there.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy

from ptychodus.api.object import Object
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product
from ptychodus.api.typing import RealArrayType

__all__ = [
    'CDToolsFrame',
    'apply_translation_offsets',
    'build_frame',
    'verify_product',
]


@dataclass(frozen=True)
class CDToolsFrame:
    """Everything FancyPtycho needs to place a product's probe on its object.

    ``obj_basis`` and ``detector_basis`` are cdtools' ``3 x 2`` real-space bases, whose
    columns are the lab-frame steps of one row and one column. ``translations`` holds one
    ``(x, y, 0)`` row per probe position, in meters from the object's center, and
    ``min_translation`` is the ``(row, column)`` pixel offset that turns their pixel
    projection into the window corner.
    """

    wavelength_m: float
    detector_distance_m: float
    obj_basis: RealArrayType
    detector_basis: RealArrayType
    translations: RealArrayType
    min_translation: RealArrayType

    def get_detector_geometry(self) -> dict[str, RealArrayType | float]:
        """The detector geometry dictionary cdtools models store."""
        return {'distance': self.detector_distance_m, 'basis': self.detector_basis}


def verify_product(product: Product, *, pixel_rtol: float = 1.0e-6) -> None:
    """Raise :class:`ValueError` unless FancyPtycho can reconstruct *product* as given.

    FancyPtycho propagates by a single far-field FFT, reads one object layer, and
    samples the probe and the object on one grid. Every probe window must also fit the
    object canvas, which FancyPtycho does not check: a window past the far edge reads a
    truncated slice, and one past the near edge wraps around.

    *pixel_rtol* is how far the probe and object pixel sizes may disagree before the
    product is refused.
    """
    if not product.metadata.far_field:
        raise ValueError('cdtools FancyPtycho reconstructs far-field diffraction data only.')

    object_ = product.object_

    if object_.num_layers != 1:
        raise ValueError(
            f'cdtools FancyPtycho reconstructs a single-layer object; this one has '
            f'{object_.num_layers} layers.'
        )

    probe_pixels = product.probes.get_pixel_geometry()
    object_pixels = object_.get_pixel_geometry()

    if not (
        math.isclose(probe_pixels.width_m, object_pixels.width_m, rel_tol=pixel_rtol)
        and math.isclose(probe_pixels.height_m, object_pixels.height_m, rel_tol=pixel_rtol)
    ):
        raise ValueError(
            'cdtools samples the probe and the object on one grid, but the probe pixel '
            f'({probe_pixels.width_m:.6g} x {probe_pixels.height_m:.6g} m) differs from '
            f'the object pixel ({object_pixels.width_m:.6g} x '
            f'{object_pixels.height_m:.6g} m). Resample one to match the other.'
        )

    _verify_windows_fit(product)


def _verify_windows_fit(product: Product) -> None:
    object_geometry = product.object_.get_geometry()
    probe_geometry = product.probes.get_geometry()
    x_px = object_geometry.map_probe_positions_to_object_x_px(product.probe_positions)
    y_px = object_geometry.map_probe_positions_to_object_y_px(product.probe_positions)
    x_lower = numpy.floor(x_px - (probe_geometry.width_px - 1) / 2)
    y_lower = numpy.floor(y_px - (probe_geometry.height_px - 1) / 2)
    outside = (
        (x_lower < 0)
        | (y_lower < 0)
        | (x_lower + probe_geometry.width_px > object_geometry.width_px)
        | (y_lower + probe_geometry.height_px > object_geometry.height_px)
    )

    if numpy.any(outside):
        first = int(numpy.flatnonzero(outside)[0])
        index = product.probe_positions[first].index
        raise ValueError(
            f'{int(numpy.count_nonzero(outside))} probe position(s) place the probe '
            f'partly outside the {object_geometry.height_px} x {object_geometry.width_px} '
            f'object canvas, the first at scan index {index}. Rebuild the object with '
            'more ExtraPaddingX / ExtraPaddingY, or regenerate it to cover the scan.'
        )


def build_frame(product: Product) -> CDToolsFrame:
    """Describe *product*'s geometry in the frame FancyPtycho places probes in.

    The detector basis is the pitch the far-field relation implies from the probe's,
    ``lambda * z / (N * dx)``. FancyPtycho reads it only to plot and to model the beam
    moving across the detector, neither of which ptychodus asks of it, so deriving it
    keeps the two consistent by construction rather than adding a quantity to check.
    """
    metadata = product.metadata
    object_: Object = product.object_
    object_geometry = object_.get_geometry()
    probe_geometry = product.probes.get_geometry()
    dx_m = object_geometry.pixel_width_m
    dy_m = object_geometry.pixel_height_m
    wavelength_m = metadata.photon_wavelength_m
    distance_m = metadata.detector_distance_m

    positions = product.probe_positions
    x_m = positions.get_coordinates_x_m() - object_geometry.center_x_m
    y_m = positions.get_coordinates_y_m() - object_geometry.center_y_m
    translations = numpy.stack([x_m, y_m, numpy.zeros_like(x_m)], axis=1)

    # translations_to_pixel(obj_basis, (x, y, 0)) is (y / dy, x / dx); subtracting this
    # leaves center_px - (N - 1) / 2, the corner resolve_patch_bounds anchors.
    min_translation = numpy.array(
        [
            -(object_geometry.height_px - 1) / 2 + (probe_geometry.height_px - 1) / 2,
            -(object_geometry.width_px - 1) / 2 + (probe_geometry.width_px - 1) / 2,
        ]
    )

    return CDToolsFrame(
        wavelength_m=wavelength_m,
        detector_distance_m=distance_m,
        obj_basis=_row_column_basis(dy_m, dx_m),
        detector_basis=_row_column_basis(
            wavelength_m * distance_m / (probe_geometry.height_px * dy_m),
            wavelength_m * distance_m / (probe_geometry.width_px * dx_m),
        ),
        translations=translations,
        min_translation=min_translation,
    )


def _row_column_basis(row_step_m: float, column_step_m: float) -> RealArrayType:
    """A cdtools basis whose rows step along lab y and whose columns step along lab x."""
    return numpy.array([[0.0, column_step_m], [row_step_m, 0.0], [0.0, 0.0]])


def apply_translation_offsets(
    positions: ProbePositionSequence,
    offsets_px: RealArrayType,
    *,
    pixel_width_m: float,
    pixel_height_m: float,
) -> ProbePositionSequence:
    """Move each position by FancyPtycho's refined ``(row, column)`` offset in pixels.

    *offsets_px* holds one row per position, already multiplied by the translation
    scale. The rows step along y and the columns along x, as the frame's basis does, so
    the move is the offset times the pixel size, and each position keeps its index and its
    photon count.
    """
    if offsets_px.shape != (len(positions), 2):
        raise ValueError(
            f'Expected {len(positions)} x 2 translation offsets, got {offsets_px.shape}.'
        )

    return ProbePositionSequence(
        [
            replace(
                position,
                x_m=position.x_m + float(offset[1]) * pixel_width_m,
                y_m=position.y_m + float(offset[0]) * pixel_height_m,
            )
            for position, offset in zip(positions, offsets_px)
        ]
    )
