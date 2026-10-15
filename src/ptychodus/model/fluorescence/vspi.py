from __future__ import annotations
from collections.abc import Iterator
from typing import Final
import logging
import time

from scipy.sparse.linalg import lsmr

from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
    VSPILinearOperator,
)

from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)

__all__ = [
    'VSPIFluorescenceEnhancer',
]


class VSPIFluorescenceEnhancer(FluorescenceEnhancer):
    SIMPLE_NAME: Final[str] = 'VSPI'
    DISPLAY_NAME: Final[str] = 'Virtual Single Pixel Imaging'

    def __init__(self, settings: FluorescenceSettings) -> None:
        super().__init__()
        self._settings = settings

    @property
    def name(self) -> str:
        return self.DISPLAY_NAME

    def get_progress_goal(self) -> int:
        return 0

    def enhance(
        self, parameters: FluorescenceEnhancerInput
    ) -> Iterator[FluorescenceEnhancerOutput]:
        # Per-channel granularity; per-LSMR-iteration progress is a future enhancement
        # (would require wrapping LinearOperator._matvec or chunking lsmr with maxiter).
        dataset = parameters.dataset
        product = parameters.product
        object_geometry = product.object_.get_geometry()
        e_cps_shape = object_geometry.height_px, object_geometry.width_px
        element_maps: list[ElementMap] = list()
        A = VSPILinearOperator(product)  # noqa: N806

        for emap in dataset:
            logger.info(f'Enhancing "{emap.name}"...')
            tic = time.perf_counter()
            m_cps = emap.counts_per_second
            result = lsmr(
                A,
                m_cps.flatten(),
                damp=self._settings.vspi_damping_factor.get_value(),
                maxiter=self._settings.vspi_max_iterations.get_value(),
                show=True,
            )
            logger.debug(result)
            e_cps = result[0].reshape(e_cps_shape)
            emap_enhanced = ElementMap(emap.name, e_cps)
            toc = time.perf_counter()
            logger.info(f'Enhanced "{emap.name}" in {toc - tic:.4f} seconds.')

            element_maps.append(emap_enhanced)
            yield FluorescenceEnhancerOutput(
                dataset=FluorescenceDataset(
                    _element_maps=list(element_maps),
                    counts_per_second_path=dataset.counts_per_second_path,
                    channel_names_path=dataset.channel_names_path,
                ),
                progress=len(element_maps),
            )
