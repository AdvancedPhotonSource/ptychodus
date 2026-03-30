from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class GlobusJob:
    flow_input: Mapping[str, Any]
    label: str
    tags: Sequence[str]


@dataclass(frozen=True)
class GlobusStatus:
    label: str
    start_time: datetime
    completion_time: datetime | None
    status: str
    action: str
    run_id: str

    @property
    def run_url(self) -> str:
        return f'https://app.globus.org/runs/{self.run_id}/logs'


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
    def run_flow(self, job: GlobusJob) -> None:
        pass

    @abstractmethod
    def refresh_status(self) -> None:
        pass


class FakeGlobusClient(GlobusClient):
    @property
    def is_supported(self) -> bool:
        return False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def run_flow(self, job: GlobusJob) -> None:
        pass

    def refresh_status(self) -> None:
        pass
