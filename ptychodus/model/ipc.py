from __future__ import annotations
import logging
import socket
import socketserver
import sys
import threading

logger = logging.getLogger(__name__)


class InterProcessCommunicationHandler(socketserver.StreamRequestHandler):

    def handle(self):
        message = self.rfile.readline().decode('utf-8').strip()
        logger.debug(f'{self.client_address[0]} wrote: \"{message}\"')
        self.wfile.write(message.upper().encode('utf-8'))


class InterProcessCommunicationServer:

    def __init__(self, portNumber: int) -> None:
        self._serverAddress = ('127.0.0.1', portNumber)
        self._ipcServer = socketserver.TCPServer(self._serverAddress,
                                                 InterProcessCommunicationHandler)
        self._thread = threading.Thread(target=self._ipcServer.serve_forever)

    def start(self) -> None:
        if self._thread.is_alive():
            logger.debug('Server thread is already alive!')
        else:
            self._thread.start()
            logger.debug(f'Server thread is communicating on {self._ipcServer.server_address}')

    def stop(self) -> None:
        self._ipcServer.shutdown()
        logger.debug('Server stopped.')


class InterProcessCommunicationClient:

    def __init__(self, portNumber: int) -> None:
        self._serverAddress = ('127.0.0.1', portNumber)

    def send(self, message: str) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self._serverAddress)

            sock.sendall((message + '\n').encode('utf-8'))
            logger.debug(f'SEND: \"{message}\"')

            response = sock.recv(1024).decode('utf-8')
            logger.debug(f'RECV: \"{response}\"')


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    portNumber = int(sys.argv[1])
    client = InterProcessCommunicationClient(portNumber)
    client.send(' '.join(sys.argv[2:]))
