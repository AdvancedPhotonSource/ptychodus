from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass

import numpy

from ...api.observer import Observable
from ...api.probe import ProbeArrayType
from .settings import ProbeSettings


class ProbeInitializer(Observable, ABC):
    '''ABC for plugins that can initialize probes'''

    @abstractmethod
    def syncFromSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes initializer state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes initializer state to settings'''
        settings.initializer.value = self.simpleName

    @abstractproperty
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        return ''.join(self.displayName.split())

    def __call__(self) -> ProbeArrayType:
        '''produces an initial probe guess'''
        pass


@dataclass
class UnimodalProbeInitializerParameters:
    rng: numpy.random.Generator
    numberOfProbeModes: int = 0

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        self.numberOfProbeModes = settings.numberOfProbeModes.value

    def syncToSettings(self, settings: ProbeSettings) -> None:
        settings.numberOfProbeModes.value = self.numberOfProbeModes


class UnimodalProbeInitializer(ProbeInitializer):

    def __init__(self, parameters: UnimodalProbeInitializerParameters) -> None:
        super().__init__()
        self._parameters = parameters

    @abstractmethod
    def syncFromSettings(self, settings: ProbeSettings) -> None:
        self._parameters.syncFromSettings(settings)
        super().syncFromSettings(settings)

    @abstractmethod
    def syncToSettings(self, settings: ProbeSettings) -> None:
        self._parameters.syncToSettings(settings)
        super().syncToSettings(settings)

    def setNumberOfProbeModes(self, number: int) -> None:
        if self._parameters.numberOfProbeModes != number:
            self._parameters.numberOfProbeModes = number
            self.notifyObservers()

    def getNumberOfProbeModes(self) -> int:
        return max(self._parameters.numberOfProbeModes, 1)

    @abstractmethod
    def _createPrimaryMode(self) -> ProbeArrayType:
        pass

    def __call__(self) -> ProbeArrayType:
        primaryMode = self._createPrimaryMode()
        modeList = [primaryMode]

        while len(modeList) < self.getNumberOfProbeModes():
            # randomly shift the first mode
            pw = primaryMode.shape[-1]

            variate1 = self._parameters.rng.uniform(size=(2, 1)) - 0.5
            variate2 = (numpy.arange(0, pw) + 0.5) / pw - 0.5
            variate = variate1 * variate2
            phaseShift = numpy.exp(-2j * numpy.pi * variate)

            mode = primaryMode * phaseShift[0][numpy.newaxis] * phaseShift[1][:, numpy.newaxis]
            modeList.append(mode)

        return numpy.stack(modeList)
