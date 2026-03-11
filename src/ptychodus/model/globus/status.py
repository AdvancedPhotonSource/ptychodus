from bisect import bisect
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import overload
import queue

from ptychodus.api.observer import ObservableSequence

from .settings import GlobusSettings
from .client import GlobusClient, GlobusStatus


class GlobusStatusRepository(ObservableSequence[GlobusStatus]):
    def __init__(
        self, settings: GlobusSettings, client: GlobusClient, status_q: queue.Queue[GlobusStatus]
    ) -> None:
        super().__init__()
        self._settings = settings
        self._client = client
        self._status_q = status_q
        self._status_list: list[GlobusStatus] = list()
        self._run_id_to_index_map: dict[str, int] = dict()
        self._status_date_time = datetime.min

    def refresh_status(self) -> None:
        self._status_date_time = datetime.now(timezone.utc)
        self._client.refresh_status()

    @overload
    def __getitem__(self, index: int) -> GlobusStatus: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[GlobusStatus]: ...

    def __getitem__(self, index: int | slice) -> GlobusStatus | Sequence[GlobusStatus]:
        return self._status_list[index]

    def __len__(self) -> int:
        return len(self._status_list)

    def _update_run_id_to_index_map(self) -> None:
        self._run_id_to_index_map = {
            status.run_id: index for index, status in enumerate(self._status_list)
        }

    def run_foreground_tasks(self) -> None:
        if self._settings.status_auto_refresh.get_value():
            status_time_delta = datetime.now(timezone.utc) - self._status_date_time
            status_age_s = status_time_delta.total_seconds()

            if status_age_s >= self._settings.status_refresh_interval_s.get_value():
                self.refresh_status()

        while True:
            try:
                new_status = self._status_q.get(block=False)
            except queue.Empty:
                break
            else:
                try:
                    index = self._run_id_to_index_map[new_status.run_id]
                except KeyError:
                    index = bisect(
                        self._status_list, new_status.start_time, key=lambda x: x.start_time
                    )
                    self._status_list.insert(index, new_status)
                    self._update_run_id_to_index_map()
                    self.notify_observers_item_inserted(index, new_status)
                else:
                    old_status = self._status_list[index]

                    if old_status != new_status:
                        self._status_list[index] = new_status
                        self.notify_observers_item_changed(index, new_status)

                self._status_q.task_done()
