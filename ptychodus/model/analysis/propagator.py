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

        self._productIndex: int | None = None
        self._propagatedWavefield: WavefieldArrayType | None = None

    def getBeginCoordinateInMeters(self) -> float:
        return float(self._settings.beginCoordinateInMeters.value)

    def getEndCoordinateInMeters(self) -> float:
        return float(self._settings.endCoordinateInMeters.value)

    def propagate(self,
                  productIndex: int,
                  *,
                  numberOfSteps: int,
                  beginCoordinateInMeters: float | None = None,
                  endCoordinateInMeters: float | None = None) -> str:
        # FIXME verify multimodal probes
        item = self._repository[productIndex]

        if beginCoordinateInMeters is not None:
            self._settings.beginCoordinateInMeters.value = Decimal(repr(beginCoordinateInMeters))

        if endCoordinateInMeters is not None:
            self._settings.endCoordinateInMeters.value = Decimal(repr(endCoordinateInMeters))

        distanceInMeters = numpy.linspace(
            float(self._settings.beginCoordinateInMeters.value),
            float(self._settings.endCoordinateInMeters.value),
            numberOfSteps,
        )
        probe = item.getProbe().getProbe()
        probeArray = probe.array
        wavelengthInMeters = item.getGeometry().probeWavelengthInMeters
        propagatedWavefield = numpy.zeros(
            (numberOfSteps, probeArray.shape[-2], probeArray.shape[-1]), dtype=probeArray.dtype)

        for idx, zInMeters in enumerate(distanceInMeters):
            propagator = FresnelPropagator(probeArray.shape[-2:], probe.getPixelGeometry(),
                                           zInMeters, wavelengthInMeters)
            wf = propagator.propagate(probeArray[-2:])
            propagatedWavefield[idx, :, :] = wf

        self._productIndex = productIndex
        self._propagatedWavefield = propagatedWavefield

        return item.getName()

    def _getProductItem(self) -> ProductRepositoryItem:
        if self._productIndex is None:
            raise ValueError('No product index!')

        return self._repository[self._productIndex]

    def _getProbe(self) -> Probe:
        item = self._getProductItem()
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
            return probe.array

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
            self.getBeginCoordinateInMeters(),
            'end_coordinate_m',
            self.getEndCoordinateInMeters(),
            'pixel_height_m',
            pixelGeometry.heightInMeters,
            'pixel_width_m',
            pixelGeometry.widthInMeters,
            'wavefield',
            self._propagatedWavefield,
        )
