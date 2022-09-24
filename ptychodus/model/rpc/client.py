import logging
import socket

logger = logging.getLogger(__name__)


class RPCMessageClient:

    def __init__(self, portNumber: int) -> None:
        self._serverAddress = ('127.0.0.1', portNumber)

    def send(self, message: str) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self._serverAddress)

            sock.sendall((message + '\n').encode('utf-8'))
            logger.debug(f'SEND: \"{message}\"')

            with sock.makefile(mode='r', encoding='utf-8') as fp:
                response = fp.readline()

            logger.debug(f'RECV: \"{response}\"')

            return response
