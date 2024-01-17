from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .experiment import Experiment
from .patterns import DiffractionPatternArrayType
from .visualize import Plot2D


@dataclass(frozen=True)
class ReconstructInput:
    patterns: DiffractionPatternArrayType
    experiment: Experiment


@dataclass(frozen=True)
class ReconstructOutput:
    experiment: Experiment
    objective: Sequence[Sequence[float]]
    plot2D: Plot2D
    result: int

    @classmethod
    def createNull(cls, experiment: Experiment) -> ReconstructOutput:
        return cls(experiment, [[]], Plot2D.createNull(), 0)


class Reconstructor(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        pass


class TrainableReconstructor(Reconstructor):

    @abstractmethod
    def ingest(self, parameters: ReconstructInput) -> None:
        pass

    @abstractmethod
    def train(self) -> Plot2D:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def saveTrainingData(self, filePath: Path) -> None:
        pass


class NullReconstructor(TrainableReconstructor):

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        return ReconstructOutput(
            experiment=parameters.experiment,
            objective=[[]],
            plot2D=Plot2D.createNull(),
            result=0,
        )

    def ingest(self, parameters: ReconstructInput) -> None:
        pass

    def train(self) -> Plot2D:
        return Plot2D.createNull()

    def reset(self) -> None:
        pass

    def saveTrainingData(self, filePath: Path) -> None:
        pass


class ReconstructorLibrary(Iterable[Reconstructor]):

    @property
    @abstractmethod
    def name(self) -> str:
        pass
