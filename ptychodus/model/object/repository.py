from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import logging

import numpy

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType
from ...api.observer import Observable, Observer
from ..itemRepository import ItemRepository
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectInitializer(ABC, Observable):
    '''ABC for plugins that can initialize objects'''

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
    def syncFromSettings(self, settings: ObjectSettings) -> None:
        '''synchronizes initializer state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ObjectSettings) -> None:
        '''synchronizes initializer state to settings'''
        pass

    @abstractmethod
    def __call__(self) -> ObjectArrayType:
        pass


class ObjectRepositoryItem(Observable, Observer):
    '''container for items that can be stored in a object repository'''

    def __init__(self, nameHint: str, array: Optional[ObjectArrayType] = None) -> None:
        super().__init__()
        self._nameHint = nameHint
        self._array = numpy.zeros((0, 0), dtype=complex)
        self._initializer: Optional[ObjectInitializer] = None

        if array is not None:
            self._setArray(array)

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
        return self._nameHint

    def getArray(self) -> ObjectArrayType:
        '''returns the array data'''
        return self._array

    def _setArray(self, array: ObjectArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Object must be a complex-valued ndarray')

        if array.ndim == 2:
            self._array = array
        else:
            raise ValueError('Object must be 2-dimensional ndarray.')

        self.notifyObservers()

    def setArray(self, array: ObjectArrayType) -> None:
        self._initializer = None
        self._setArray(array)

    def reinitialize(self) -> None:
        '''reinitializes the object array'''
        if self._initializer is None:
            logger.error('Missing object initializer!')
            return

        try:
            self._array = self._initializer()
        except Exception:
            logger.exception('Failed to reinitialize object!')
        else:
            self.notifyObservers()

    def getInitializerSimpleName(self) -> str:
        return 'FromMemory' if self._initializer is None else self._initializer.simpleName

    def getInitializerDisplayName(self) -> str:
        return 'From Memory' if self._initializer is None else self._initializer.displayName

    def getInitializer(self) -> Optional[ObjectInitializer]:
        '''returns the initializer'''
        return self._initializer

    def setInitializer(self, initializer: ObjectInitializer) -> None:
        '''sets the initializer'''
        if self._initializer is not None:
            self._initializer.removeObserver(self)

        self._initializer = initializer
        initializer.addObserver(self)
        self.reinitialize()

    def getDataType(self) -> str:
        '''returns the array data type'''
        return str(self._array.dtype)

    def getExtentInPixels(self) -> ImageExtent:
        '''returns the array width and height'''
        return ImageExtent(width=self._array.shape[-1], height=self._array.shape[-2])

    def getSizeInBytes(self) -> int:
        '''returns the array size in bytes'''
        return self._array.nbytes

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self.reinitialize()


ObjectRepository = ItemRepository[ObjectRepositoryItem]
