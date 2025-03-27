from collections import OrderedDict
from pathlib import Path
from time import monotonic as time
import logging
import threading

from .processor import AutomationDatasetProcessor
from .repository import AutomationDatasetRepository, AutomationDatasetState
from .settings import AutomationSettings

logger = logging.getLogger(__name__)


class AutomationDatasetBuffer:
    def __init__(
        self,
        settings: AutomationSettings,
        repository: AutomationDatasetRepository,
        processor: AutomationDatasetProcessor,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._processor = processor
        self._event_times: OrderedDict[Path, float] = OrderedDict()
        self._event_times_lock = threading.Lock()
        self._stop_work_event = threading.Event()
        self._worker = threading.Thread()

    def put(self, file_path: Path) -> None:
        with self._event_times_lock:
            self._event_times[file_path] = time()
            self._event_times.move_to_end(file_path)

        self._repository.put(file_path, AutomationDatasetState.EXISTS)

    def _process(self) -> None:
        while not self._stop_work_event.is_set():
            is_file_ready_for_processing = False

            with self._event_times_lock:
                try:
                    file_path, event_time = next(iter(self._event_times.items()))
                except StopIteration:
                    pass
                else:
                    delay_time = self._settings.watchdog_delay_s.get_value()
                    is_file_ready_for_processing = event_time + delay_time < time()

                    if is_file_ready_for_processing:
                        self._event_times.popitem(last=False)

            if is_file_ready_for_processing:
                self._processor.put(file_path)
            else:
                self._stop_work_event.wait(timeout=5.0)  # TODO make configurable

    def start(self) -> None:
        if self._worker.is_alive():
            self.stop()

        logger.info('Starting automation thread...')
        self._stop_work_event.clear()
        self._worker = threading.Thread(target=self._process)
        self._worker.start()
        logger.info('Automation thread started.')

    def stop(self) -> None:
        logger.info('Stopping automation thread...')
        self._stop_work_event.set()
        self._worker.join()
        logger.info('Automation thread stopped.')
