from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import sys

from .observer import Observable


class Experiment(Observable):

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._probeEnergyInElectronVolts = 10000.
        self._detectorObjectDistanceInMeters = 1.
        # FIXME validate data/scan/probe/object consistency for recon
        # FIXME sync to/from settings; perhaps from reconstructor
        # FIXME validate values (filter <0, inf, nan, etc.)

    def getName(self) -> str:
        return self._name

    def setName(self, name: str) -> None:
        if self._name != name:
            self._name = name
            self.notifyObservers()

    def getProbeEnergyInElectronVolts(self) -> float:
        return self._probeEnergyInElectronVolts

    def setProbeEnergyInElectronVolts(self, energyInElectronVolts: float) -> None:
        if self._probeEnergyInElectronVolts != energyInElectronVolts:
            self._probeEnergyInElectronVolts = energyInElectronVolts
            self.notifyObservers()

    def getDetectorObjectDistanceInMeters(self) -> float:
        return self._detectorObjectDistanceInMeters

    def setDetectorObjectDistanceInMeters(self, distanceInMeters: float) -> None:
        if self._detectorObjectDistanceInMeters != distanceInMeters:
            self._detectorObjectDistanceInMeters = distanceInMeters
            self.notifyObservers()

    def getSizeInBytes(self) -> int:
        sizeInBytes = 0
        sizeInBytes += sys.getsizeof(self._name)
        sizeInBytes += sys.getsizeof(self._probeEnergyInElectronVolts)
        sizeInBytes += sys.getsizeof(self._detectorObjectDistanceInMeters)
        return sizeInBytes


class ExperimentFileReader(ABC):

    @abstractmethod
    def read(self, filePath: Path) -> Experiment:
        '''reads an experiment from file'''
        pass


class ExperimentFileWriter(ABC):

    @abstractmethod
    def write(self, filePath: Path, experiment_: Experiment) -> None:
        '''writes an experiment to file'''
        pass
