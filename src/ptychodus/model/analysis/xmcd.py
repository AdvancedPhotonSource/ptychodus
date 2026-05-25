from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import logging

import numpy
from skimage.registration import phase_cross_correlation

from ptychodus.api.object import Object
from ptychodus.api.observer import Observable
from ptychodus.api.reconstructor import ReconstructionAmbiguities

from ..product import ProductRepository

__all__ = [
    'XMCDAnalyzer',
    'XMCDResult',
    'align_objects',
    'estimate_xmcd',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XMCDResult:
    structural_object: Object
    magnetic_object: Object


# FIXME move align_objects to ptychodus.api (only depends on Object + FFT, parallels FRC extraction)
def align_objects(
    reference_object: Object, moving_object: Object, *, upsample_factor: int = 100
) -> Object:
    """Sub-pixel align ``moving_object`` to ``reference_object``.

    Estimates the sub-pixel translation between the two reconstructions with
    ``skimage.registration.phase_cross_correlation`` (run on the amplitudes), then
    applies the inverse shift to the complex ``moving_object`` array via a Fourier
    phase ramp so the complex phase is preserved across the interpolation.
    """
    reference_pixel_geometry = reference_object.get_pixel_geometry()
    moving_pixel_geometry = moving_object.get_pixel_geometry()
    if reference_pixel_geometry != moving_pixel_geometry:
        raise ValueError(
            f'Object pixel geometry mismatch: reference {reference_pixel_geometry} vs moving {moving_pixel_geometry}!'
        )

    reference_array = reference_object.get_layers_flattened()
    moving_array = moving_object.get_layers_flattened()
    if reference_array.shape != moving_array.shape:
        raise ValueError(
            f'Object array shape mismatch: reference {reference_array.shape} vs moving {moving_array.shape}!'
        )

    shift_yx, _, _ = phase_cross_correlation(
        numpy.absolute(reference_array),
        numpy.absolute(moving_array),
        upsample_factor=upsample_factor,
    )
    logger.info(f'XMCD sub-pixel alignment shift (y, x) = {tuple(shift_yx)} px')

    height_px, width_px = moving_array.shape[-2:]
    ky = numpy.fft.fftfreq(height_px)
    kx = numpy.fft.fftfreq(width_px)
    ky_grid, kx_grid = numpy.meshgrid(ky, kx, indexing='ij')
    phase_ramp = numpy.exp(2j * numpy.pi * (shift_yx[0] * ky_grid + shift_yx[1] * kx_grid))

    moving_fft = numpy.fft.fft2(moving_array)
    aligned_array = numpy.fft.ifft2(moving_fft * phase_ramp).astype(moving_array.dtype)

    return Object(
        array=aligned_array,
        pixel_geometry=reference_pixel_geometry,
        center=reference_object.get_center(),
        layer_spacing_m=list(moving_object.layer_spacing_m),
    )


def estimate_xmcd(
    rcp_object: Object, lcp_object_aligned: Object, *, epsilon: float = 1.0e-12
) -> XMCDResult:
    if rcp_object.num_layers > 1 or lcp_object_aligned.num_layers > 1:
        logger.warning(
            'XMCD estimation flattens multi-layer objects; per-layer XMCD is not implemented.'
        )

    rcp_center = rcp_object.get_center()
    lcp_center = lcp_object_aligned.get_center()

    if rcp_center != lcp_center:
        raise ValueError(f'Object center mismatch: rcp {rcp_center} vs lcp {lcp_center}!')

    rcp_pixel_geometry = rcp_object.get_pixel_geometry()
    lcp_pixel_geometry = lcp_object_aligned.get_pixel_geometry()

    if rcp_pixel_geometry != lcp_pixel_geometry:
        raise ValueError(
            f'Object pixel geometry mismatch: rcp {rcp_pixel_geometry} vs lcp {lcp_pixel_geometry}!'
        )

    rcp_array = rcp_object.get_layers_flattened()
    lcp_array = lcp_object_aligned.get_layers_flattened()

    if rcp_array.shape != lcp_array.shape:
        raise ValueError(
            f'Object array shape mismatch: rcp {rcp_array.shape} vs lcp {lcp_array.shape}!'
        )

    cross_helicity_product = rcp_array * numpy.conj(lcp_array)
    parallel_helicity_product = rcp_array * lcp_array
    parallel_helicity_ratio = rcp_array / (lcp_array + epsilon)

    structural_absorption = numpy.sqrt(numpy.abs(cross_helicity_product))
    structural_phase = 0.5 * numpy.angle(parallel_helicity_product)
    magnetic_absorption = numpy.sqrt(numpy.abs(parallel_helicity_ratio))
    magnetic_phase = 0.5 * numpy.angle(cross_helicity_product)

    return XMCDResult(
        structural_object=Object(
            array=structural_absorption * numpy.exp(1j * structural_phase),
            pixel_geometry=rcp_pixel_geometry,
            center=rcp_center,
            layer_spacing_m=[],
        ),
        magnetic_object=Object(
            array=magnetic_absorption * numpy.exp(1j * magnetic_phase),
            pixel_geometry=rcp_pixel_geometry,
            center=rcp_center,
            layer_spacing_m=[],
        ),
    )


class XMCDAnalyzer(Observable):
    def __init__(self, repository: ProductRepository) -> None:
        super().__init__()
        self._repository = repository

        self._lcp_product_index = -1
        self._rcp_product_index = -1
        self._result: XMCDResult | None = None

    def set_lcp_product(self, lcirc_product_index: int) -> None:
        if self._lcp_product_index != lcirc_product_index:
            self._lcp_product_index = lcirc_product_index
            self.notify_observers()

    def get_lcp_product(self) -> int:
        return self._lcp_product_index

    def set_rcp_product(self, rcirc_product_index: int) -> None:
        if self._rcp_product_index != rcirc_product_index:
            self._rcp_product_index = rcirc_product_index
            self.notify_observers()

    def get_rcp_product(self) -> int:
        return self._rcp_product_index

    def analyze(self) -> None:
        lcp_product = self._repository[self._lcp_product_index].get_product()
        rcp_product = self._repository[self._rcp_product_index].get_product()

        aligned_lcp_object = align_objects(rcp_product.object_, lcp_product.object_)
        aligned_lcp_product = replace(lcp_product, object_=aligned_lcp_object)  # FIXME

        ambiguities = ReconstructionAmbiguities.estimate(aligned_lcp_product, reference=rcp_product)
        standardized_lcp_product = ambiguities.standardize_product(aligned_lcp_product)

        self._result = estimate_xmcd(
            rcp_object=rcp_product.object_,
            lcp_object_aligned=standardized_lcp_product.object_,
        )
        self.notify_observers()

    def get_result(self) -> XMCDResult:
        if self._result is None:
            raise ValueError('No analyzed data!')

        return self._result

    def get_save_file_filters(self) -> Sequence[str]:
        return [self.get_save_file_filter()]

    def get_save_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def save_data(self, file_path: Path) -> None:  # FIXME rethink
        if self._result is None:
            raise ValueError('No analyzed data!')

        structural_object = self._result.structural_object
        magnetic_object = self._result.magnetic_object
        pixel_geometry = structural_object.get_pixel_geometry()
        center = structural_object.get_center()

        contents: dict[str, Any] = {
            'structural_object': structural_object.get_array(),
            'magnetic_object': magnetic_object.get_array(),
            'pixel_height_m': pixel_geometry.height_m,
            'pixel_width_m': pixel_geometry.width_m,
            'center_x_m': center.coordinate_x_m,
            'center_y_m': center.coordinate_y_m,
        }

        numpy.savez_compressed(file_path, allow_pickle=False, **contents)
