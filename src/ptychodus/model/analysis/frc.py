from __future__ import annotations

import logging
import time


from ptychodus.api.illumination import compute_illumination_map
from ptychodus.api.metrics import (
    ApodizationWindow,
    FourierRingCorrelation,
    PowerSpectralDensity,
    ProbeComparison,
    ScoringRegion,
    SpectralSNRConvention,
    compute_fourier_ring_correlation,
    compute_illumination_scoring_region,
    compute_object_comparison,
    compute_power_spectral_density,
    compute_probe_comparison,
)
from ptychodus.api.typing import ComplexArrayType, RealArrayType
from ptychodus.api.product import Product

from ..product import ProductRepository

__all__ = [
    'FourierRingCorrelation',
    'FourierRingCorrelator',
    'PowerSpectralDensity',
    'ProbeComparison',
    'SpectralSNRConvention',
]

logger = logging.getLogger(__name__)


def _scoring_region(
    product: Product,
    array_shape: tuple[int, int],
    *,
    probe_erosion_fraction: int = 8,
) -> ScoringRegion | None:
    """Build an illumination-derived scoring region, or None if one cannot be derived.

    Args:
        product: Supplies the scan positions and probe that define the illuminated area.
        array_shape: Shape of the object array the region will be applied to.
        probe_erosion_fraction: Erode the illuminated mask by the probe extent divided by
            this. The outermost scan positions leave a rim that only part of the probe ever
            covered, so the photon count there is real but the reconstruction is not
            converged. Larger values erode less.
    """
    logger.info('Computing illumination map...')
    tic = time.perf_counter()
    illumination_map = compute_illumination_map(product)
    toc = time.perf_counter()
    logger.info(f'Computed illumination map in {toc - tic:.4f} seconds.')

    probe_extent_px = min(product.probes.width_px, product.probes.height_px)

    try:
        return compute_illumination_scoring_region(
            illumination_map,
            array_shape,
            erosion_px=probe_extent_px // probe_erosion_fraction,
        )
    except ValueError as exc:
        logger.warning(f'Falling back to the full object array: {exc}')
        return None


def _pair_scoring_region(reference: Product, test: Product) -> ScoringRegion | None:
    """Return the reference's scoring region on the grid the two objects share.

    :func:`compute_object_comparison` center-crops both objects to their common extent,
    so the region is built for that shape, where it both weights the ambiguity estimate
    and crops the compared arrays.
    """
    common_shape = (
        min(reference.object_.height_px, test.object_.height_px),
        min(reference.object_.width_px, test.object_.width_px),
    )
    return _scoring_region(reference, common_shape)


def _ambiguity_weights(region: ScoringRegion | None) -> RealArrayType | None:
    """Weight the ambiguity estimate by the scanned region, so the border cannot bias it."""
    return None if region is None else region.get_array_weights()


class FourierRingCorrelator:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def correlate(
        self,
        product_index_1: int,
        product_index_2: int,
        *,
        convention: SpectralSNRConvention = SpectralSNRConvention.HALF_SPLIT,
    ) -> FourierRingCorrelation:
        product1 = self._repository[product_index_1].get_product()
        product2 = self._repository[product_index_2].get_product()
        region = _pair_scoring_region(product1, product2)

        comparison = compute_object_comparison(
            reference=product1, test=product2, weights=_ambiguity_weights(region)
        )
        reference: ComplexArrayType = comparison.reference_complex
        test: ComplexArrayType = comparison.test_complex
        window = ApodizationWindow.TUKEY if region is None else region.weights

        if region is not None:
            reference = region.crop(reference)
            test = region.crop(test)

        logger.info('Computing Fourier ring correlation...')
        tic = time.perf_counter()
        result = compute_fourier_ring_correlation(
            reference,
            test,
            pixel_width_m=comparison.pixel_geometry.width_m,
            pixel_height_m=comparison.pixel_geometry.height_m,
            window=window,
            convention=convention,
        )
        toc = time.perf_counter()
        logger.info(f'Computed Fourier ring correlation in {toc - tic:.4f} seconds.')

        return result

    def compare_probes(self, product_index_1: int, product_index_2: int) -> ProbeComparison:
        """Compare the two products' probes as mixed states; see :func:`compute_probe_comparison`."""
        product1 = self._repository[product_index_1].get_product()
        product2 = self._repository[product_index_2].get_product()
        region = _pair_scoring_region(product1, product2)

        logger.info('Comparing probes...')
        tic = time.perf_counter()
        result = compute_probe_comparison(product1, product2, weights=_ambiguity_weights(region))
        toc = time.perf_counter()
        logger.info(f'Compared probes in {toc - tic:.4f} seconds.')

        return result

    def compute_power_spectrum(self, product_index: int) -> PowerSpectralDensity:
        product = self._repository[product_index].get_product()
        object_ = product.object_
        array: ComplexArrayType = object_.get_layers_flattened()
        pixel_geometry = object_.get_pixel_geometry()
        region = _scoring_region(product, array.shape)
        window = ApodizationWindow.TUKEY if region is None else region.weights

        if region is not None:
            array = region.crop(array)

        logger.info('Computing power spectral density...')
        tic = time.perf_counter()
        result = compute_power_spectral_density(
            array,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            window=window,
        )
        toc = time.perf_counter()
        logger.info(f'Computed power spectral density in {toc - tic:.4f} seconds.')

        return result
