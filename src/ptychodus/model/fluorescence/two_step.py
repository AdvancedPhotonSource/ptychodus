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
        upscaling_strategy_chooser: PluginChooser[UpscalingStrategy],
        deconvolution_strategy_chooser: PluginChooser[DeconvolutionStrategy],
    ) -> None:
        super().__init__()
        self._upscaling_strategy_chooser = upscaling_strategy_chooser
        self._deconvolution_strategy_chooser = deconvolution_strategy_chooser

        upscaling_strategy_chooser.synchronize_with_parameter(settings.upscaling_strategy)
        upscaling_strategy_chooser.add_observer(self)

        deconvolution_strategy_chooser.synchronize_with_parameter(settings.deconvolution_strategy)
        deconvolution_strategy_chooser.add_observer(self)

    def enhance(self, dataset: FluorescenceDataset, product: Product) -> FluorescenceDataset:
        upscaler = self._upscaling_strategy_chooser.get_current_plugin().strategy
        deconvolver = self._deconvolution_strategy_chooser.get_current_plugin().strategy
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

    def get_upscaling_strategies(self) -> Iterator[str]:
        for plugin in self._upscaling_strategy_chooser:
            yield plugin.display_name

    def get_upscaling_strategy(self) -> str:
        return self._upscaling_strategy_chooser.get_current_plugin().display_name

    def set_upscaling_strategy(self, name: str) -> None:
        self._upscaling_strategy_chooser.set_current_plugin(name)

    def get_deconvolution_strategies(self) -> Iterator[str]:
        for plugin in self._deconvolution_strategy_chooser:
            yield plugin.display_name

    def get_deconvolution_strategy(self) -> str:
        return self._deconvolution_strategy_chooser.get_current_plugin().display_name

    def set_deconvolution_strategy(self, name: str) -> None:
        self._deconvolution_strategy_chooser.set_current_plugin(name)

    def _update(self, observable: Observable) -> None:
        if observable is self._upscaling_strategy_chooser:
            self.notify_observers()
        elif observable is self._deconvolution_strategy_chooser:
            self.notify_observers()
