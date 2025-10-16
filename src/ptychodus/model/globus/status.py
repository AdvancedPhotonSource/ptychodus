from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import overload
import threading


@dataclass(frozen=True)
class GlobusStatus:
    label: str
    start_time: datetime
    completion_time: datetime | None
    status: str
    action: str
    run_id: str
    run_url: str


class GlobusStatusRepository(Sequence[GlobusStatus]):
    def __init__(self) -> None:
        super().__init__()
        self._status_lock = threading.Lock()
        self._status_list: list[GlobusStatus] = list()
        self._status_date_time = datetime.min
        self.refresh_status_event = threading.Event()

    @overload
    def __getitem__(self, index: int) -> GlobusStatus: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[GlobusStatus]: ...

    def __getitem__(self, index: int | slice) -> GlobusStatus | Sequence[GlobusStatus]:
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

    def update(self, status_sequence: Sequence[GlobusStatus]) -> None:
        with self._status_lock:
            self._status_date_time = datetime.now(timezone.utc)
            self._status_list = list(status_sequence)
            self._status_list.sort(key=lambda x: x.start_time)
