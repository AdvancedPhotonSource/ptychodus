import logging
import threading

logger = logging.getLogger(__name__)


class WorkflowAuthorizer:
    def __init__(self) -> None:
        super().__init__()
        self._authorizeLock = threading.Lock()
        self._authorizeCode = str()
        self._authorizeURL = "https://aps.anl.gov"
        self.isAuthorizedEvent = threading.Event()
        self.isAuthorizedEvent.set()
        self.shutdownEvent = threading.Event()

    @property
    def isAuthorized(self) -> bool:
        return self.isAuthorizedEvent.is_set()

    def getAuthorizeURL(self) -> str:
        with self._authorizeLock:
            return self._authorizeURL

    def setCodeFromAuthorizeURL(self, code: str) -> None:
        with self._authorizeLock:
            self._authorizeCode = code
            self.isAuthorizedEvent.set()

    def getCodeFromAuthorizeURL(self) -> str:
        with self._authorizeLock:
            return self._authorizeCode

    def authenticate(self, authorizeURL: str) -> None:
        logger.info(f"Authenticate at {authorizeURL}")

        with self._authorizeLock:
            self._authorizeURL = authorizeURL
            self.isAuthorizedEvent.clear()

        while not self.shutdownEvent.is_set():
            if self.isAuthorizedEvent.wait(timeout=1.0):
                break
