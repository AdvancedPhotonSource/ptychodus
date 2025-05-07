import logging
import threading

logger = logging.getLogger(__name__)


class WorkflowAuthorizer:
    def __init__(self) -> None:
        super().__init__()
        self._authorize_lock = threading.Lock()
        self._authorize_code = str()
        self._authorize_url = 'https://aps.anl.gov'
        self.is_authorized_event = threading.Event()
        self.is_authorized_event.set()
        self.shutdown_event = threading.Event()

    @property
    def is_authorized(self) -> bool:
        return self.is_authorized_event.is_set()

    def get_authorize_url(self) -> str:
        with self._authorize_lock:
            return self._authorize_url

    def set_code_from_authorize_url(self, code: str) -> None:
        with self._authorize_lock:
            self._authorize_code = code
            self.is_authorized_event.set()

    def get_code_from_authorize_url(self) -> str:
        with self._authorize_lock:
            return self._authorize_code

    def authenticate(self, authorize_url: str) -> None:
        logger.info(f'Authenticate at {authorize_url}')

        with self._authorize_lock:
            self._authorize_url = authorize_url
            self.is_authorized_event.clear()

        while not self.shutdown_event.is_set():
            if self.is_authorized_event.wait(timeout=1.0):
                break
