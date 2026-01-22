from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GlobusJob:
    flow_input: Mapping[str, Any]
    label: str
    tags: Sequence[str]


class GlobusClient(ABC):
    @property
    @abstractmethod
    def is_supported(self) -> bool:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def request_status_update(self) -> None:
        pass

    @abstractmethod
    def run_flow(self, job: GlobusJob) -> None:
        pass


class FakeGlobusClient(GlobusClient):
    @property
    def is_supported(self) -> bool:
        return False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def request_status_update(self) -> None:
        pass

    def run_flow(self, job: GlobusJob) -> None:
        pass
