from abc import ABC, abstractmethod
from collections.abc import Iterator


class PtychoPackDevice(ABC):
    @abstractmethod
    def get_available_devices(self) -> Iterator[str]:
        pass

    @abstractmethod
    def get_device(self) -> str:
        pass

    @abstractmethod
    def set_device(self, name: str) -> None:
        pass


class NullPtychoPackDevice(PtychoPackDevice):
    def get_available_devices(self) -> Iterator[str]:
        return iter([])

    def get_device(self) -> str:
        return ""

    def set_device(self, name: str) -> None:
        pass
