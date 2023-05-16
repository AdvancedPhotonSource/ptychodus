from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import logging

import numpy

from ...api.image import ImageExtent
from ...api.observer import Observable, Observer
from ...api.probe import ProbeArrayType
from ..itemRepository import ItemRepository
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeInitializer(ABC, Observable):
    '''ABC for plugins that can initialize probes'''

    @property
    @abstractmethod
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @property
    @abstractmethod
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def syncFromSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes initializer state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes initializer state to settings'''
        pass

    @abstractmethod
    def __call__(self) -> ProbeArrayType:
        '''produces an initial probe guess'''
        pass


class ProbeRepositoryItem(Observable, Observer):
    '''container for items that can be stored in a probe repository'''

    def __init__(self,
                 rng: numpy.random.Generator,
                 nameHint: str,
                 array: Optional[ProbeArrayType] = None) -> None:
        super().__init__()
        self._rng = rng
        self._nameHint = nameHint
        self._array = numpy.zeros((0, 0, 0), dtype=complex) if array is None else array
        self._initializer: Optional[ProbeInitializer] = None
        self._numberOfProbeModes: int = 0

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
        return self._nameHint

    @property
    def canSelect(self) -> bool:
        '''indicates whether item can be selected'''
        return (self._initializer is not None)

    def reinitialize(self) -> None:
        '''reinitializes the probe array'''
        if self._initializer is None:
            logger.error('Missing probe initializer!')
            return

        try:
            primaryMode = self._initializer()
        except:
            logger.exception('Failed to reinitialize probe!')
            return

        modeList = [primaryMode]

        while len(modeList) < self.getNumberOfProbeModes():
            # randomly shift the first mode
            pw = primaryMode.shape[-1]

            variate1 = self._rng.uniform(size=(2, 1)) - 0.5
            variate2 = (numpy.arange(0, pw) + 0.5) / pw - 0.5
            variate = variate1 * variate2
            phaseShift = numpy.exp(-2j * numpy.pi * variate)

            mode = primaryMode * phaseShift[0][numpy.newaxis] * phaseShift[1][:, numpy.newaxis]
            modeList.append(mode)

        self._array = numpy.stack(modeList)

        self.notifyObservers()

    def getInitializerSimpleName(self) -> str:
        return 'FromMemory' if self._initializer is None else self._initializer.simpleName

    def getInitializer(self) -> Optional[ProbeInitializer]:
        '''returns the initializer'''
        return self._initializer

    def setInitializer(self, initializer: ProbeInitializer) -> None:
        '''sets the initializer'''
        if self._initializer is not None:
            self._initializer.removeObserver(self)

        self._initializer = initializer
        initializer.addObserver(self)
        self.reinitialize()

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes item state from settings'''
        self._numberOfProbeModes = settings.numberOfProbeModes.value
        self.notifyObservers()

    def syncToSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes item state to settings'''
        settings.numberOfProbeModes.value = self.getNumberOfProbeModes()

    def getDataType(self) -> str:
        '''returns the array data type'''
        return str(self._array.dtype)

    def getExtentInPixels(self) -> ImageExtent:
        '''returns the array width and height'''
        return ImageExtent(width=self._array.shape[-1], height=self._array.shape[-2])

    def getSizeInBytes(self) -> int:
        '''returns the array size in bytes'''
        return self._array.nbytes

    def getArray(self) -> ProbeArrayType:
        '''returns the array data'''
        return self._array

    def setNumberOfProbeModes(self, number: int) -> None:
        if self._numberOfProbeModes != number:
            self._numberOfProbeModes = number
            self.reinitialize()

    def getNumberOfProbeModes(self) -> int:
        return max(self._numberOfProbeModes, 1)

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self.reinitialize()


ProbeRepository = ItemRepository[ProbeRepositoryItem]
