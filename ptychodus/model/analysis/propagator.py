from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable
from ptychodus.api.probe import Probe, WavefieldArrayType

from ..product import ProductRepository, ProductRepositoryItem
from ..propagator import FresnelPropagator
from .settings import ProbePropagationSettings

logger = logging.getLogger(__name__)


class ProbePropagator(Observable):

    def __init__(self, settings: ProbePropagationSettings, repository: ProductRepository) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository

        self._productIndex = 0
        self._propagatedWavefield: WavefieldArrayType | None = None

    def setProduct(self, productIndex: int) -> None:
        if self._productIndex != productIndex:
            self._productIndex = productIndex
            self._propagatedWavefield = None
            self.notifyObservers()

    def getProductName(self) -> str:
        item = self._repository[self._productIndex]
        return item.getName()

    def getBeginCoordinateInMeters(self) -> Decimal:
        return self._settings.beginCoordinateInMeters.value

    def getEndCoordinateInMeters(self) -> Decimal:
        return self._settings.endCoordinateInMeters.value

    def propagate(self, *, beginCoordinateInMeters: Decimal, endCoordinateInMeters: Decimal,
                  numberOfSteps: int) -> None:
        # FIXME verify multimodal probes
        item = self._repository[self._productIndex]
        probe = item.getProbe().getProbe()
        wavelengthInMeters = item.getGeometry().probeWavelengthInMeters
        propagatedWavefield = numpy.zeros(
            (numberOfSteps, probe.array.shape[-2], probe.array.shape[-1]),
            dtype=probe.array.dtype,
        )
        distanceInMeters = numpy.linspace(float(beginCoordinateInMeters),
                                          float(endCoordinateInMeters), numberOfSteps)

        for idx, zInMeters in enumerate(distanceInMeters):
            propagator = FresnelPropagator(probe.array.shape[-2:], probe.getPixelGeometry(),
                                           zInMeters, wavelengthInMeters)
            wf = propagator.propagate(probe.array[-2:])
            propagatedWavefield[idx, :, :] = wf

        self._settings.beginCoordinateInMeters.value = beginCoordinateInMeters
        self._settings.endCoordinateInMeters.value = endCoordinateInMeters
        self._propagatedWavefield = propagatedWavefield
        self.notifyObservers()

    def _getProbe(self) -> Probe:
        item = self._repository[self._productIndex]
        return item.getProbe().getProbe()

    def getPixelGeometry(self) -> PixelGeometry:
        probe = self._getProbe()
        return probe.getPixelGeometry()

    def getNumberOfSteps(self) -> int:
        if self._propagatedWavefield is None:
            return 1

        return self._propagatedWavefield.shape[-3]

    def getXYProjection(self, step: int) -> WavefieldArrayType:
        if self._propagatedWavefield is None:
            probe = self._getProbe()
            return probe.array[0]

        return self._propagatedWavefield[step]

    def getZXProjection(self) -> WavefieldArrayType:
        if self._propagatedWavefield is None:
            raise ValueError('No propagated wavefield!')

        sz = self._propagatedWavefield.shape[-2]
        arrayL = self._propagatedWavefield[:, (sz - 1) // 2, :]
        arrayR = self._propagatedWavefield[:, sz // 2, :]
        return numpy.transpose(arrayL + arrayR) / 2  # type: ignore

    def getZYProjection(self) -> WavefieldArrayType:
        if self._propagatedWavefield is None:
            raise ValueError('No propagated wavefield!')

        sz = self._propagatedWavefield.shape[-1]
        arrayL = self._propagatedWavefield[:, :, (sz - 1) // 2]
        arrayR = self._propagatedWavefield[:, :, sz // 2]
        return numpy.transpose(arrayL + arrayR) / 2  # type: ignore

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def savePropagatedProbe(self, filePath: Path) -> None:
        if self._propagatedWavefield is None:
            raise ValueError('No propagated wavefield!')

        pixelGeometry = self.getPixelGeometry()
        numpy.savez(
            filePath,
            'begin_coordinate_m',
            float(self.getBeginCoordinateInMeters()),
            'end_coordinate_m',
            float(self.getEndCoordinateInMeters()),
            'pixel_height_m',
            pixelGeometry.heightInMeters,
            'pixel_width_m',
            pixelGeometry.widthInMeters,
            'wavefield',
            self._propagatedWavefield,
        )
