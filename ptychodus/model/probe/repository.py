from __future__ import annotations
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Final, Optional
import logging

import numpy

from ...api.geometry import Interval
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
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self,
                 rng: numpy.random.Generator,
                 nameHint: str,
                 array: Optional[ProbeArrayType] = None) -> None:
        super().__init__()
        self._rng = rng
        self._nameHint = nameHint
        self._array = numpy.zeros((1, 0, 0), dtype=complex)
        self._initializer: Optional[ProbeInitializer] = None
        self._numberOfModes: int = 0

        if array is not None:
            self._setArray(array)

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
        return self._nameHint

    def getArray(self) -> ProbeArrayType:
        '''returns the array data'''
        return self._array

    def _setArray(self, array: ProbeArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Probe must be a complex-valued ndarray')

        if array.ndim == 2:
            self._array = array[numpy.newaxis, ...]
        elif array.ndim == 3:
            self._array = array
        else:
            raise ValueError('Probe must be 2- or 3-dimensional ndarray.')

        self.notifyObservers()

    def setArray(self, array: ProbeArrayType) -> None:
        self._initializer = None
        self._setArray(array)

    def reinitialize(self) -> None:
        '''reinitializes the probe array'''
        if self._initializer is None:
            logger.error('Missing probe initializer!')
            return

        try:
            initialProbe = self._initializer()
        except:
            logger.exception('Failed to reinitialize probe!')
            return

        modeList: list[ProbeArrayType] = list()

        if initialProbe.ndim == 2:
            modeList.append(initialProbe)
        elif initialProbe.ndim == 3:
            for mode in initialProbe:
                modeList.append(mode)
        else:
            raise ValueError('Probe must be 2- or 3-dimensional ndarray.')

        while len(modeList) < self._numberOfModes:
            # randomly shift the first mode
            pw = initialProbe.shape[-1]

            variate1 = self._rng.uniform(size=(2, 1)) - 0.5
            variate2 = (numpy.arange(0, pw) + 0.5) / pw - 0.5
            variate = variate1 * variate2
            phaseShift = numpy.exp(-2j * numpy.pi * variate)

            mode = modeList[0] * phaseShift[0][numpy.newaxis] * phaseShift[1][:, numpy.newaxis]
            modeList.append(mode)

        array = numpy.stack(modeList)
        # TODO array /= numpy.sqrt(numpy.sum(numpy.abs(array)**2))

        self._setArray(array)

    def getInitializerSimpleName(self) -> str:
        return 'FromMemory' if self._initializer is None else self._initializer.simpleName

    def getInitializerDisplayName(self) -> str:
        return 'From Memory' if self._initializer is None else self._initializer.displayName

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
        self._numberOfModes = settings.numberOfModes.value
        self.reinitialize()

    def syncToSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes item state to settings'''
        settings.numberOfModes.value = self.getNumberOfModes()

    def getDataType(self) -> str:
        '''returns the array data type'''
        return str(self._array.dtype)

    def getExtentInPixels(self) -> ImageExtent:
        '''returns the array width and height'''
        return ImageExtent(width=self._array.shape[-1], height=self._array.shape[-2])

    def getSizeInBytes(self) -> int:
        '''returns the array size in bytes'''
        return self._array.nbytes

    def getNumberOfModesLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberOfModes(self) -> int:
        return self._array.shape[0]

    def setNumberOfModes(self, number: int) -> None:
        if self._numberOfModes != number:
            self._numberOfModes = number
            # TODO only reinitialize as needed
            self.reinitialize()

    def getMode(self, mode: int) -> ProbeArrayType:
        return self._array[mode, :, :]

    def getModesFlattened(self) -> ProbeArrayType:
        return self._array.transpose((1, 0, 2)).reshape(self._array.shape[1], -1)

    def getModeRelativePower(self, mode: int) -> Decimal:
        if numpy.isnan(self._array).any():
            logger.error('Probe contains NaN value(s)!')
            return Decimal()

        probe = self._array
        power = numpy.sum((probe * probe.conj()).real, axis=(-2, -1))
        powersum = power.sum()

        if powersum > 0.:
            power /= powersum

        return Decimal.from_float(power[mode])

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self.reinitialize()


ProbeRepository = ItemRepository[ProbeRepositoryItem]
