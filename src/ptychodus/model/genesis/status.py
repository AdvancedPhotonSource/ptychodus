from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import overload
import logging
import queue

from ptychodus.api.observer import ObservableSequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenesisStatus:
    facility: str
    label: str
    action: str
    status: str
    start_time: datetime
    completion_time: datetime | None


class GenesisStatusRepository(ObservableSequence[GenesisStatus]):
    def __init__(self, status_q: queue.Queue[GenesisStatus]) -> None:
        super().__init__()
        self._status_q = status_q
        self._status_list: list[GenesisStatus] = list()

    @overload
    def __getitem__(self, index: int) -> GenesisStatus: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[GenesisStatus]: ...

    def __getitem__(self, index: int | slice) -> GenesisStatus | Sequence[GenesisStatus]:
        return self._status_list[index]

    def __len__(self) -> int:
        return len(self._status_list)

    def run_foreground_tasks(self) -> None:
        while True:
            try:
                new_status = self._status_q.get(block=False)
            except queue.Empty:
                break
            else:
                logger.info(f'{new_status.label} [{new_status.action}]: {new_status.status}')
                self._status_q.task_done()

                try:
                    old_status = self._status_list[-1]
                except IndexError:
                    self._status_list.append(new_status)
                    self.notify_observers_item_inserted(0, new_status)
                else:
                    if (
                        old_status.facility == new_status.facility
                        and old_status.label == new_status.label
                        and old_status.action == new_status.action
                    ):
                        self._status_list[-1] = new_status
                        self.notify_observers_item_changed(len(self._status_list) - 1, new_status)
                    else:
                        self._status_list.append(new_status)
                        self.notify_observers_item_inserted(len(self._status_list) - 1, new_status)
