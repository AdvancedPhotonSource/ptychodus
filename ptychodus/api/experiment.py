from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .object import Object
from .probe import Probe
from .scan import Scan


@dataclass(frozen=True)
class ExperimentMetadata:
    name: str
    comments: str
    probeEnergyInElectronVolts: float
    detectorObjectDistanceInMeters: float


@dataclass(frozen=True)
class Experiment:
    metadata: ExperimentMetadata
    scan: Scan
    probe: Probe
    object_: Object


class ExperimentFileReader(ABC):

    @abstractmethod
    def read(self, filePath: Path) -> Experiment:
        '''reads an experiment from file'''
        pass


class ExperimentFileWriter(ABC):

    @abstractmethod
    def write(self, filePath: Path, experiment: Experiment) -> None:
        '''writes an experiment to file'''
        pass
