from __future__ import annotations
from dataclasses import dataclass
import logging

import numpy

from ptychodus.api.probe import ProbeGeometry, WavefieldArrayType

from ..product import ProductRepository
from ..propagator import fresnel_propagate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PropagatedProbe:
    itemIndex: int
    itemName: str
    geometry: ProbeGeometry
    wavefield: WavefieldArrayType

    def getXYProjection(self, step: int) -> WavefieldArrayType:
        return self.wavefield[step]

    def getZXProjection(self) -> WavefieldArrayType:
        sz = self.wavefield.shape[-2]
        lhs = (sz - 1) // 2
        rhs = sz // 2
        return (self.wavefield[:, lhs, :] + self.wavefield[:, rhs, :]) / 2

    def getZYProjection(self) -> WavefieldArrayType:
        sz = self.wavefield.shape[-1]
        lhs = (sz - 1) // 2
        rhs = sz // 2
        return (self.wavefield[:, :, lhs] + self.wavefield[:, :, rhs]) / 2


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
        probeGeometry = probe.getGeometry()

        # TODO non-square pixels are unsupported
        pixelSizeInMeters = probeGeometry.pixelWidthInMeters

        wavelengthInMeters = item.getGeometry().probeWavelengthInMeters
        wavefield = numpy.zeros((numberOfSteps, probeArray.shape[-2], probeArray.shape[-1]),
                                dtype=probeArray.dtype)

        for idx, zInMeters in enumerate(distanceInMeters):
            wf = fresnel_propagate(probeArray, pixelSizeInMeters, zInMeters, wavelengthInMeters)
            wavefield[idx, :, :] = wf

        return PropagatedProbe(
            itemIndex=itemIndex,
            itemName=item.getName(),
            geometry=probeGeometry,
            wavefield=wavefield,
        )
