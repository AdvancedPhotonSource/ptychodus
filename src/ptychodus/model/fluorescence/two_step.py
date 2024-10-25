from __future__ import annotations
from collections.abc import Sequence
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
        reinitObservable: Observable,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._upscalingStrategyChooser = upscalingStrategyChooser
        self._deconvolutionStrategyChooser = deconvolutionStrategyChooser
        self._reinitObservable = reinitObservable

        self._syncUpscalingStrategyFromSettings()
        upscalingStrategyChooser.addObserver(self)

        self._syncDeconvolutionStrategyFromSettings()
        deconvolutionStrategyChooser.addObserver(self)

        reinitObservable.addObserver(self)

    def enhance(self, dataset: FluorescenceDataset, product: Product) -> FluorescenceDataset:
        upscaler = self._upscalingStrategyChooser.currentPlugin.strategy
        deconvolver = self._deconvolutionStrategyChooser.currentPlugin.strategy
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

    def getUpscalingStrategyList(self) -> Sequence[str]:
        return self._upscalingStrategyChooser.getDisplayNameList()

    def getUpscalingStrategy(self) -> str:
        return self._upscalingStrategyChooser.currentPlugin.displayName

    def setUpscalingStrategy(self, name: str) -> None:
        self._upscalingStrategyChooser.setCurrentPluginByName(name)
        self._settings.upscalingStrategy.setValue(
            self._upscalingStrategyChooser.currentPlugin.simpleName
        )

    def _syncUpscalingStrategyFromSettings(self) -> None:
        self.setUpscalingStrategy(self._settings.upscalingStrategy.getValue())

    def getDeconvolutionStrategyList(self) -> Sequence[str]:
        return self._deconvolutionStrategyChooser.getDisplayNameList()

    def getDeconvolutionStrategy(self) -> str:
        return self._deconvolutionStrategyChooser.currentPlugin.displayName

    def setDeconvolutionStrategy(self, name: str) -> None:
        self._deconvolutionStrategyChooser.setCurrentPluginByName(name)
        self._settings.deconvolutionStrategy.setValue(
            self._deconvolutionStrategyChooser.currentPlugin.simpleName
        )

    def _syncDeconvolutionStrategyFromSettings(self) -> None:
        self.setDeconvolutionStrategy(self._settings.deconvolutionStrategy.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncUpscalingStrategyFromSettings()
            self._syncDeconvolutionStrategyFromSettings()
        elif observable is self._upscalingStrategyChooser:
            self.notifyObservers()
        elif observable is self._deconvolutionStrategyChooser:
            self.notifyObservers()
