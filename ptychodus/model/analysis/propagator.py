from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import WavefieldArrayType

from ..product import ProductRepository
from ..propagator import fresnel_propagate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PropagatedProbe:
    itemIndex: int
    itemName: str
    beginCoordinateInMeters: float
    endCoordinateInMeters: float
    pixelGeometry: PixelGeometry
    wavefield: WavefieldArrayType

    def getNumberOfSteps(self) -> int:
        return self.wavefield.shape[0]

    def getXYProjection(self, step: int) -> WavefieldArrayType:
        return self.wavefield[step]

    def getZXProjection(self) -> WavefieldArrayType:
        sz = self.wavefield.shape[-2]
        arrayL = self.wavefield[:, (sz - 1) // 2, :]
        arrayR = self.wavefield[:, sz // 2, :]
        return numpy.transpose(arrayL + arrayR) / 2

    def getZYProjection(self) -> WavefieldArrayType:
        sz = self.wavefield.shape[-1]
        arrayL = self.wavefield[:, :, (sz - 1) // 2]
        arrayR = self.wavefield[:, :, sz // 2]
        return numpy.transpose(arrayL + arrayR) / 2


class ProbePropagator:

    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def propagate(self, itemIndex: int, beginCoordinateInMeters: float,
                  endCoordinateInMeters: float, numberOfSteps: int) -> PropagatedProbe:
        item = self._repository[itemIndex]
        distanceInMeters = numpy.linspace(beginCoordinateInMeters, endCoordinateInMeters,
                                          numberOfSteps)
        probe = item.getProbe().getProbe()
        probeArray = probe.array
        pixelGeometry = probe.getPixelGeometry()
        wavelengthInMeters = item.getGeometry().probeWavelengthInMeters
        wavefield = numpy.zeros((numberOfSteps, probeArray.shape[-2], probeArray.shape[-1]),
                                dtype=probeArray.dtype)

        for idx, zInMeters in enumerate(distanceInMeters):
            wf = fresnel_propagate(probeArray[0], pixelGeometry, zInMeters, wavelengthInMeters)
            wavefield[idx, :, :] = wf

        return PropagatedProbe(
            itemIndex=itemIndex,
            itemName=item.getName(),
            beginCoordinateInMeters=beginCoordinateInMeters,
            endCoordinateInMeters=endCoordinateInMeters,
            pixelGeometry=probe.getPixelGeometry(),
            wavefield=wavefield,
        )

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def savePropagatedProbe(self, filePath: Path, result: PropagatedProbe) -> None:
        numpy.savez(
            filePath,
            'begin_coordinate_m',
            result.beginCoordinateInMeters,
            'end_coordinate_m',
            result.endCoordinateInMeters,
            'pixel_height_m',
            result.pixelGeometry.heightInMeters,
            'pixel_width_m',
            result.pixelGeometry.widthInMeters,
            'wavefield',
            result.wavefield,
        )
