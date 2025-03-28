from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import overload
import threading


@dataclass(frozen=True)
class WorkflowStatus:
    label: str
    start_time: datetime
    completion_time: datetime | None
    status: str
    action: str
    run_id: str
    run_url: str


class WorkflowStatusRepository(Sequence[WorkflowStatus]):
    def __init__(self) -> None:
        super().__init__()
        self._status_lock = threading.Lock()
        self._status_list: list[WorkflowStatus] = list()
        self._status_date_time = datetime.min
        self.refresh_status_event = threading.Event()

    @overload
    def __getitem__(self, index: int) -> WorkflowStatus: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[WorkflowStatus]: ...

    def __getitem__(self, index: int | slice) -> WorkflowStatus | Sequence[WorkflowStatus]:
        with self._status_lock:
            return self._status_list[index]

    def __len__(self) -> int:
        with self._status_lock:
            return len(self._status_list)

    def get_status_date_time(self) -> datetime:
        with self._status_lock:
            return self._status_date_time

    def refresh_status(self) -> None:
        self.refresh_status_event.set()

    def update(self, status_sequence: Sequence[WorkflowStatus]) -> None:
        with self._status_lock:
            self._status_date_time = datetime.now(timezone.utc)
            self._status_list = list(status_sequence)
            self._status_list.sort(key=lambda x: x.start_time)
