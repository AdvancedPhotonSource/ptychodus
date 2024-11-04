from collections.abc import Iterator
import queue
import logging


class ReconstructorLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self._log: queue.Queue[str] = queue.Queue()

    def messages(self) -> Iterator[str]:
        while True:
            try:
                yield self._log.get(block=False)
                self._log.task_done()
            except queue.Empty:
                break

    def emit(self, record: logging.LogRecord) -> None:
        text = self.format(record)
        self._log.put(text)
