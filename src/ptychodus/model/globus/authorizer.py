import logging

from ptychodus.api.observer import Observable

logger = logging.getLogger(__name__)


class GlobusAuthorizer(Observable):
    def __init__(self) -> None:
        super().__init__()
        self._authorize_url = ''
        self._authorize_code = ''

    @property
    def _has_authorize_url(self) -> bool:
        auth_url_empty = not self._authorize_url
        return not auth_url_empty

    @property
    def has_authorize_code(self) -> bool:
        auth_code_empty = not self._authorize_code
        return not auth_code_empty

    @property
    def needs_authorize_code(self) -> bool:
        return self._has_authorize_url and not self.has_authorize_code

    def _authorize(self, url: str) -> None:
        logger.info(f'Authorize at {url}')
        self._authorize_url = url
        self._authorize_code = ''
        self.notify_observers()

    def get_authorize_url(self) -> str:
        return self._authorize_url

    def set_code_from_authorize_url(self, code: str) -> None:
        logger.info('Received authorization code.')
        self._authorize_code = code
        self.notify_observers()

    def get_authorize_code(self) -> str:
        return self._authorize_code

    def cancel_authorization(self) -> None:
        logger.info('Canceled authorization')
        self._authorize_url = ''
        self._authorize_code = ''
        self.notify_observers()


class AuthorizeWithGlobus:
    def __init__(self, authorizer: GlobusAuthorizer, authorize_url: str) -> None:
        self._authorizer = authorizer
        self._authorize_url = authorize_url

    def __call__(self) -> None:
        self._authorizer._authorize(self._authorize_url)
