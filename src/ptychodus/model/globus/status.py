from bisect import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import overload

from ptychodus.api.observer import ObservableSequence


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


class GlobusStatusRepository(ObservableSequence[GlobusStatus]):
    def __init__(self) -> None:
        super().__init__()
        self._status_list: list[GlobusStatus] = list()

    @overload
    def __getitem__(self, index: int) -> GlobusStatus: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[GlobusStatus]: ...

    def __getitem__(self, index: int | slice) -> GlobusStatus | Sequence[GlobusStatus]:
        return self._status_list[index]

    def __len__(self) -> int:
        return len(self._status_list)

    def _insert_sorted(self, status: GlobusStatus) -> None:
        index = bisect(self._status_list, status.start_time, key=lambda x: x.start_time)
        self._status_list.insert(index, status)
        self.notify_observers_item_inserted(index, status)

    def _pop(self, index: int) -> GlobusStatus:
        status = self._status_list.pop(index)
        self.notify_observers_item_removed(index, status)
        return status

    def _update_if_changed(self, index: int, status: GlobusStatus) -> None:
        old_status = self._status_list[index]

        if old_status != status:
            self._status_list[index] = status
            self.notify_observers_item_changed(index, status)


class UpdateGlobusStatusRepository:
    def __init__(
        self, status_repository: GlobusStatusRepository, status_sequence: Sequence[GlobusStatus]
    ) -> None:
        self._status_repository = status_repository
        self._new_status_list = sorted(status_sequence, key=lambda x: x.start_time)

    def __call__(self) -> None:
        run_id_to_new_status_map = {status.run_id: status for status in self._new_status_list}

        for index in reversed(range(len(self._status_repository))):
            old_status = self._status_repository[index]

            try:
                new_status = run_id_to_new_status_map.pop(old_status.run_id)
            except KeyError:
                self._status_repository._pop(index)
            else:
                self._status_repository._update_if_changed(index, new_status)

        for new_status in run_id_to_new_status_map.values():
            self._status_repository._insert_sorted(new_status)
