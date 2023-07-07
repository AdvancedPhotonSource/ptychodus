from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .data import DiffractionPatternArrayType
from .object import ObjectArrayType, ObjectInterpolator
from .probe import ProbeArrayType
from .scan import Scan


@dataclass(frozen=True)
class ReconstructInput:
    diffractionPatternArray: DiffractionPatternArrayType
    scan: Scan
    probeArray: ProbeArrayType
    objectInterpolator: ObjectInterpolator

    @property
    def objectArray(self) -> ObjectArrayType:
        return self.objectInterpolator.getArray()


@dataclass(frozen=True)
class ReconstructOutput:
    scan: Scan | None
    probeArray: ProbeArrayType | None
    objectArray: ObjectArrayType | None
    objective: Sequence[Sequence[float]]
    result: int

    @classmethod
    def createNull(cls) -> ReconstructOutput:
        return cls(None, None, None, [[]], 0)


class Reconstructor(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def execute(self, parameters: ReconstructInput) -> ReconstructOutput:
        pass


class TrainableReconstructor(Reconstructor):

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class NullReconstructor(TrainableReconstructor):

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def execute(self, parameters: ReconstructInput) -> ReconstructOutput:
        return ReconstructOutput(
            scan=parameters.scan,
            probeArray=parameters.probeArray,
            objectArray=parameters.objectInterpolator.getArray(),
            objective=[[]],
            result=0,
        )

    def train(self) -> None:
        pass

    def reset(self) -> None:
        pass


class ReconstructorLibrary(Iterable[Reconstructor]):

    @property
    @abstractmethod
    def name(self) -> str:
        pass
