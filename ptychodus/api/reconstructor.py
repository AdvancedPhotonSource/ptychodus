from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .product import Product
from .patterns import DiffractionPatternArrayType


@dataclass(frozen=True)
class ReconstructInput:
    patterns: DiffractionPatternArrayType
    product: Product


@dataclass(frozen=True)
class ReconstructOutput:
    product: Product
    result: int


class Reconstructor(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        pass


@dataclass(frozen=True)
class TrainOutput:
    trainingLoss: Sequence[float]
    validationLoss: Sequence[float]
    result: int


class TrainableReconstructor(Reconstructor):

    @abstractmethod
    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        pass

    @abstractmethod
    def getSaveFileFilterList(self) -> Sequence[str]:
        pass

    @abstractmethod
    def getSaveFileFilter(self) -> str:
        pass

    @abstractmethod
    def saveTrainingData(self, filePath: Path) -> None:
        pass

    @abstractmethod
    def train(self) -> TrainOutput:
        pass

    @abstractmethod
    def clearTrainingData(self) -> None:
        pass

    def saveModel(self, filePath: Path) -> None:
        raise NotImplementedError(f'Save trained model to \"{filePath}\"')  # FIXME


class NullReconstructor(TrainableReconstructor):

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        return ReconstructOutput(parameters.product, 0)

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        pass

    def getSaveFileFilterList(self) -> Sequence[str]:
        return list()

    def getSaveFileFilter(self) -> str:
        return str()

    def saveTrainingData(self, filePath: Path) -> None:
        pass

    def train(self) -> TrainOutput:
        return TrainOutput([], [], 0)

    def clearTrainingData(self) -> None:
        pass


class ReconstructorLibrary(Iterable[Reconstructor]):

    @property
    @abstractmethod
    def name(self) -> str:
        pass
