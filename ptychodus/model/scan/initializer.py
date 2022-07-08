from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from decimal import Decimal

import numpy

from ...api.observer import Observable, Observer
from ...api.scan import ScanPoint, ScanPointSequence, ScanPointTransform
from .settings import ScanSettings


@dataclass
class ScanInitializerParameters:
    rng: numpy.random.Generator
    transform: ScanPointTransform = ScanPointTransform.PXPY
    jitterRadiusInMeters: Decimal = Decimal()

    @classmethod
    def createFromSettings(cls, rng: numpy.random.Generator,
                           settings: ScanSettings) -> ScanInitializerParameters:
        transform = ScanPointTransform.fromSimpleName(settings.transform.value)
        jitterRadiusInMeters = settings.jitterRadiusInMeters.value
        return cls(rng, transform, jitterRadiusInMeters)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.transform.value = self.transform.simpleName
        settings.jitterRadiusInMeters.value = self.jitterRadiusInMeters


class ScanInitializer(ScanPointSequence, Observable):
    '''ABC for plugins that can initialize scan sequences'''

    @classmethod
    @abstractproperty
    def category(self) -> str:
        '''returns a unique category for organizing scan positions'''
        pass

    @classmethod
    @abstractproperty
    def name(self) -> str:
        '''returns a unique name'''
        pass

    @abstractmethod
    def _getPoint(self, index: int) -> ScanPoint:
        '''returns the scan point'''
        pass

    def __init__(self, parameters: ScanInitializerParameters) -> None:
        super().__init__()
        self._parameters = parameters

    def syncToSettings(self, settings: ScanSettings) -> None:
        '''synchronizes parameters to settings'''
        self._parameters.syncToSettings(settings)

    def getTransform(self) -> ScanPointTransform:
        '''gets the scan point transform'''
        return self._parameters.transform

    def setTransform(self, transform: ScanPointTransform) -> None:
        '''sets the scan point transform'''
        if self._parameters.transform != transform:
            self._parameters.transform = transform
            self.notifyObservers()

    def getJitterRadiusInMeters(self) -> Decimal:
        '''gets the jitter radius'''
        return self._parameters.jitterRadiusInMeters

    def setJitterRadiusInMeters(self, jitterRadiusInMeters: Decimal) -> None:
        '''sets the jitter radius'''
        if self._parameters.jitterRadiusInMeters != jitterRadiusInMeters:
            self._parameters.jitterRadiusInMeters = jitterRadiusInMeters
            self.notifyObservers()

    def __getitem__(self, index: int) -> ScanPoint:
        '''returns the jittered and transformed scan point'''
        point = self._getPoint(index)

        if self._parameters.jitterRadiusInMeters > Decimal():
            rad = Decimal(self._parameters.rng.uniform())
            dirX = Decimal(self._parameters.rng.normal())
            dirY = Decimal(self._parameters.rng.normal())

            scalar = self._parameters.jitterRadiusInMeters * (rad / (dirX**2 + dirY**2)).sqrt()
            point = ScanPoint(point.x + scalar * dirX, point.y + scalar * dirY)

        return self._parameters.transform(point)
