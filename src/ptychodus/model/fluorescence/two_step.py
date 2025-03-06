from __future__ import annotations
from collections.abc import Iterator
from typing import Final
import logging
import time

from ptychodus.api.fluorescence import (
    DeconvolutionStrategy,
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancingAlgorithm,
    UpscalingStrategy,
)
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product

from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)

__all__ = [
    'TwoStepFluorescenceEnhancingAlgorithm',
]


class TwoStepFluorescenceEnhancingAlgorithm(FluorescenceEnhancingAlgorithm, Observable, Observer):
    SIMPLE_NAME: Final[str] = 'TwoStep'
    DISPLAY_NAME: Final[str] = 'Upscale and Deconvolve'

    def __init__(
        self,
        settings: FluorescenceSettings,
        upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
        deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
    ) -> None:
        super().__init__()
        self._upscalingStrategyChooser = upscalingStrategyChooser
        self._deconvolutionStrategyChooser = deconvolutionStrategyChooser

        upscalingStrategyChooser.synchronize_with_parameter(settings.upscalingStrategy)
        upscalingStrategyChooser.addObserver(self)

        deconvolutionStrategyChooser.synchronize_with_parameter(settings.deconvolutionStrategy)
        deconvolutionStrategyChooser.addObserver(self)

    def enhance(self, dataset: FluorescenceDataset, product: Product) -> FluorescenceDataset:
        # FIXME OPR
        upscaler = self._upscalingStrategyChooser.get_current_plugin().strategy
        deconvolver = self._deconvolutionStrategyChooser.get_current_plugin().strategy
        element_maps: list[ElementMap] = list()

        for emap in dataset.element_maps:
            logger.info(f'Enhancing "{emap.name}"...')
            tic = time.perf_counter()
            emap_upscaled = upscaler(emap, product)
            emap_enhanced = deconvolver(emap_upscaled, product)
            toc = time.perf_counter()
            logger.info(f'Enhanced "{emap.name}" in {toc - tic:.4f} seconds.')

            element_maps.append(emap_enhanced)

        return FluorescenceDataset(
            element_maps=element_maps,
            counts_per_second_path=dataset.counts_per_second_path,
            channel_names_path=dataset.channel_names_path,
        )

    def getUpscalingStrategyList(self) -> Iterator[str]:
        for plugin in self._upscalingStrategyChooser:
            yield plugin.display_name

    def getUpscalingStrategy(self) -> str:
        return self._upscalingStrategyChooser.get_current_plugin().display_name

    def setUpscalingStrategy(self, name: str) -> None:
        self._upscalingStrategyChooser.set_current_plugin(name)

    def getDeconvolutionStrategyList(self) -> Iterator[str]:
        for plugin in self._deconvolutionStrategyChooser:
            yield plugin.display_name

    def getDeconvolutionStrategy(self) -> str:
        return self._deconvolutionStrategyChooser.get_current_plugin().display_name

    def setDeconvolutionStrategy(self, name: str) -> None:
        self._deconvolutionStrategyChooser.set_current_plugin(name)

    def update(self, observable: Observable) -> None:
        if observable is self._upscalingStrategyChooser:
            self.notifyObservers()
        elif observable is self._deconvolutionStrategyChooser:
            self.notifyObservers()
