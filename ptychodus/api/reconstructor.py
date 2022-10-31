from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ReconstructResult:
    result: int
    objective: list[list[float]]


class Reconstructor(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def reconstruct(self) -> ReconstructResult:
        pass


class NullReconstructor(Reconstructor):

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self) -> ReconstructResult:
        return ReconstructResult(0, [[]])


class ReconstructorLibrary(Iterable[Reconstructor]):

    @abstractproperty
    def name(self) -> str:
        pass
