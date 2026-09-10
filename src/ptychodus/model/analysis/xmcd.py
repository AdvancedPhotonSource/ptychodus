from __future__ import annotations
from dataclasses import replace
import logging
import time

from ptychodus.api.metrics import estimate_reconstruction_ambiguities
from ptychodus.api.object import align_objects
from ptychodus.api.xmcd import XMCDResult, estimate_xmcd

from ..product import ProductRepository

logger = logging.getLogger(__name__)


class XMCDAnalyzer:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def analyze(self, lcp_product_index: int, rcp_product_index: int) -> XMCDResult:
        lcp_product = self._repository[lcp_product_index].get_product()
        rcp_product = self._repository[rcp_product_index].get_product()

        if lcp_product.object_.num_layers > 1 or rcp_product.object_.num_layers > 1:
            logger.warning('XMCD flattens multi-layer objects; per-layer XMCD is not implemented.')

        logger.info('Computing object alignment...')
        tic = time.perf_counter()
        cropped_rcp_object, aligned_lcp_object = align_objects(
            rcp_product.object_, lcp_product.object_
        )
        toc = time.perf_counter()
        logger.info(f'Computed object alignment in {toc - tic:.4f} seconds.')

        cropped_rcp_product = replace(rcp_product, object_=cropped_rcp_object)
        aligned_lcp_product = replace(lcp_product, object_=aligned_lcp_object)

        ambiguities = estimate_reconstruction_ambiguities(
            aligned_lcp_product, reference=cropped_rcp_product
        )
        standardized_lcp_product = ambiguities.standardize_product(aligned_lcp_product)

        logger.info('Computing XMCD...')
        tic = time.perf_counter()
        result = estimate_xmcd(
            rcp_object=cropped_rcp_object,
            lcp_object_aligned=standardized_lcp_product.object_,
        )
        toc = time.perf_counter()
        logger.info(f'Computed XMCD in {toc - tic:.4f} seconds.')

        return result
