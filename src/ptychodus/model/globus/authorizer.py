import logging
import queue

from ptychodus.api.observer import Observable

logger = logging.getLogger(__name__)


class GlobusAuthorizer(Observable):
    def __init__(self) -> None:
        super().__init__()
        self._auth_url_q: queue.Queue[str] = queue.Queue()
        self._auth_code_q: queue.Queue[str] = queue.Queue()
        self._auth_url = ''

    @property
    def needs_authorize_code(self) -> bool:
        return len(self._auth_url) > 0

    def get_authorize_url(self) -> str:
        return self._auth_url

    def set_code_from_authorize_url(self, code: str) -> None:
        logger.info('Received authorization code.')
        self._auth_code_q.put(code)
        self._auth_url = ''
        self.notify_observers()

    def cancel_authorization(self) -> None:
        logger.info('Canceling authorization.')
        self._auth_code_q.put('')
        self._auth_url = ''
        self.notify_observers()

    def run_foreground_tasks(self) -> None:
        try:
            url = self._auth_url_q.get(block=False)
        except queue.Empty:
            pass
        else:
            logger.info(f'Authorize at {url}')
            self._auth_url = url
            self._authorize_code = ''
            self.notify_observers()
