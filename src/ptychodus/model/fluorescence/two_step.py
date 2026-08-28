from __future__ import annotations
from collections.abc import Iterator
from typing import Final
import logging
import time

from ptychodus.api.fluorescence import (
    DeconvolutionStrategy,
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
    UpscalingStrategy,
)
from ptychodus.api.plugins import PluginChooser

logger = logging.getLogger(__name__)

__all__ = [
    'TwoStepFluorescenceEnhancer',
]


class TwoStepFluorescenceEnhancer(FluorescenceEnhancer):
    SIMPLE_NAME: Final[str] = 'TwoStep'
    DISPLAY_NAME: Final[str] = 'Upscale and Deconvolve'

    def __init__(
        self,
        upscaling_strategy_chooser: PluginChooser[UpscalingStrategy],
        deconvolution_strategy_chooser: PluginChooser[DeconvolutionStrategy],
    ) -> None:
        super().__init__()
        self._upscaling_strategy_chooser = upscaling_strategy_chooser
        self._deconvolution_strategy_chooser = deconvolution_strategy_chooser

    @property
    def name(self) -> str:
        return self.DISPLAY_NAME

    def get_progress_goal(self) -> int:
        return 0

    def enhance(
        self, parameters: FluorescenceEnhancerInput
    ) -> Iterator[FluorescenceEnhancerOutput]:
        upscaler = self._upscaling_strategy_chooser.get_current_plugin().strategy
        deconvolver = self._deconvolution_strategy_chooser.get_current_plugin().strategy
        dataset = parameters.dataset
        product = parameters.product
        element_maps: list[ElementMap] = list()

        for emap in dataset:
            logger.info(f'Enhancing "{emap.name}"...')
            tic = time.perf_counter()
            emap_upscaled = upscaler(emap, product)
            emap_enhanced = deconvolver(emap_upscaled, product)
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
