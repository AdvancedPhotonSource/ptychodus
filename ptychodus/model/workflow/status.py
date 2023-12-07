from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Union, overload
import threading


@dataclass(frozen=True)
class WorkflowStatus:
    label: str
    startTime: datetime
    completionTime: datetime | None
    status: str
    action: str
    runID: str
    runURL: str


class WorkflowStatusRepository(Sequence[WorkflowStatus]):

    def __init__(self) -> None:
        super().__init__()
        self._statusLock = threading.Lock()
        self._statusList: list[WorkflowStatus] = list()
        self._statusDateTime = datetime.min
        self.refreshStatusEvent = threading.Event()

    @overload
    def __getitem__(self, index: int) -> WorkflowStatus:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[WorkflowStatus]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> \
            Union[WorkflowStatus, Sequence[WorkflowStatus]]:
        with self._statusLock:
            return self._statusList[index]

    def __len__(self) -> int:
        with self._statusLock:
            return len(self._statusList)

    def getStatusDateTime(self) -> datetime:
        with self._statusLock:
            return self._statusDateTime

    def refreshStatus(self) -> None:
        self.refreshStatusEvent.set()

    def update(self, statusSequence: Sequence[WorkflowStatus]) -> None:
        with self._statusLock:
            self._statusDateTime = datetime.utcnow()
            self._statusList = list(statusSequence)
            self._statusList.sort(key=lambda x: x.startTime)
