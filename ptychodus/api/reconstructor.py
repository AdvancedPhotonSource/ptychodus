from abc import ABC, abstractmethod, abstractproperty


class Reconstructor(ABC):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def backendName(self) -> str:
        pass

    @abstractmethod
    def reconstruct(self) -> int:
        pass
