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
from .modes import MultimodalProbeFactory, ProbeModeDecayType
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
    SIMPLE_NAME: Final[str] = 'FromMemory'
    DISPLAY_NAME: Final[str] = 'From Memory'

    def __init__(self,
                 modesFactory: MultimodalProbeFactory,
                 nameHint: str,
                 array: Optional[ProbeArrayType] = None) -> None:
        super().__init__()
        self._modesFactory = modesFactory
        self._nameHint = nameHint
        self._array = numpy.zeros((1, 0, 0), dtype=complex)
        self._initializer: Optional[ProbeInitializer] = None

        if array is not None:
            self._setArray(array)

        modesFactory.addObserver(self)

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
        except Exception:
            logger.exception('Failed to reinitialize probe!')
            return

        array = self._modesFactory.build(initialProbe)
        self._setArray(array)

    def getInitializerSimpleName(self) -> str:
        return self.SIMPLE_NAME if self._initializer is None else self._initializer.simpleName

    def getInitializerDisplayName(self) -> str:
        return self.DISPLAY_NAME if self._initializer is None else self._initializer.displayName

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
        self._modesFactory.syncFromSettings(settings)
        self.reinitialize()

    def syncToSettings(self, settings: ProbeSettings) -> None:
        '''synchronizes item state to settings'''
        self._modesFactory.syncToSettings(settings)

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
        return self._modesFactory.getNumberOfModesLimits()

    def getNumberOfModes(self) -> int:
        return self._array.shape[0]

    def setNumberOfModes(self, number: int) -> None:
        self._modesFactory.setNumberOfModes(number)

    @property
    def isOrthogonalizeModesEnabled(self) -> bool:
        return self._modesFactory.isOrthogonalizeModesEnabled

    def setOrthogonalizeModesEnabled(self, value: bool) -> None:
        self._modesFactory.setOrthogonalizeModesEnabled(value)

    def getModeDecayType(self) -> ProbeModeDecayType:
        return self._modesFactory.getModeDecayType()

    def setModeDecayType(self, value: ProbeModeDecayType) -> None:
        self._modesFactory.setModeDecayType(value)

    def getModeDecayRatioLimits(self) -> Interval[Decimal]:
        return self._modesFactory.getModeDecayRatioLimits()

    def getModeDecayRatio(self) -> Decimal:
        return self._modesFactory.getModeDecayRatio()

    def setModeDecayRatio(self, value: Decimal) -> None:
        self._modesFactory.setModeDecayRatio(value)

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

        return Decimal.from_float(float(power[mode]))

    def update(self, observable: Observable) -> None:
        if observable is self._modesFactory:
            self.reinitialize()
        elif observable is self._initializer:
            self.reinitialize()


ProbeRepository = ItemRepository[ProbeRepositoryItem]
