from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterable, Sequence
from typing import Iterator


class Reconstructor(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def reconstruct(self) -> int:
        pass


class NullReconstructor(Reconstructor):

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self) -> int:
        return 0


class ReconstructorLibrary(Iterable[Reconstructor]):

    @abstractproperty
    def name(self) -> str:
        pass
