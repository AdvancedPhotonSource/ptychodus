from __future__ import annotations
from enum import IntEnum, auto
from pathlib import Path
import argparse
import json
import logging
import queue
import socket
import socketserver
import sys
import threading

from ..api.rpc import RPCMessage

logger = logging.getLogger(__name__)


class LoadResultsMessage(RPCMessage):

    def __init__(self, filePath: Path) -> None:
        self._filePath = filePath

    @classmethod
    @property
    def messageType(self) -> int:
        return 100

    @property
    def filePath(self) -> Path:
        return self._filePath

    @classmethod
    def fromDict(self, values: dict[str, typing.Any]) -> LoadResultsMessage:
        filePath = Path(values['filePath'])
        return LoadResultsMessage(filePath)

    def toDict(self) -> dict[str, typing.Any]:
        result = super().toDict()
        result['filePath'] = str(self._filePath)
        return result


class RPC_StreamRequestHandler(socketserver.StreamRequestHandler):

    def handle(self):
        message = self.rfile.readline().decode('utf-8').strip()
        logger.debug(f'RECV: \"{message}\" from {self.client_address[0]}')
        response = self.server.processMessage(message)
        logger.debug(f'SEND: \"{response}\" to {self.client_address[0]}')
        self.wfile.write(response.encode('utf-8'))


class RPC_TCPServer(socketserver.TCPServer):

    def __init__(self, portNumber: int, messageQueue: queue.Queue) -> None:
        super().__init__(('127.0.0.1', portNumber), RPC_StreamRequestHandler)
        self._messageQueue = messageQueue

    def processMessage(self, message: str) -> str:
        response = 'UNKNOWN'

        try:
            messageDict = json.loads(message)
            logger.debug(messageDict)
        except json.JSONDecodeError:
            logger.debug(f'Failed to decode \"{message}\"!')
            response = 'FAILURE'
        else:
            # FIXME generalize and handle conversion errors
            messageObject = LoadResultsMessage.fromDict(messageDict)
            logger.debug(messageObject)
            self._messageQueue.put(messageObject)
            response = 'SUCCESS'

        return response


class RemoteProcessCommunicationServer:

    def __init__(self, portNumber: int) -> None:
        self._messageQueue = queue.Queue()
        self._rpcServer = RPC_TCPServer(portNumber, self._messageQueue)
        self._thread = threading.Thread(target=self._rpcServer.serve_forever)

    def start(self) -> None:
        if self._thread.is_alive():
            logger.debug('Server thread is already alive!')
        else:
            self._thread.start()
            logger.debug(f'Server thread is communicating on {self._rpcServer.server_address}')

    def stop(self) -> None:
        self._rpcServer.shutdown()
        self._rpcServer.server_close()
        # FIXME self._messageQueue.join()
        logger.debug('Server stopped.')


class RemoteProcessCommunicationClient:

    def __init__(self, portNumber: int) -> None:
        self._serverAddress = ('127.0.0.1', portNumber)

    def send(self, message: str) -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect(self._serverAddress)

            sock.sendall((message + '\n').encode('utf-8'))
            logger.debug(f'SEND: \"{message}\"')

            response = sock.recv(1024).decode('utf-8')  # TODO buffer size
            logger.debug(f'RECV: \"{response}\"')

            return response


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='ptychodus-rpc',
        description='ptychodus-rpc communicates with an active ptychodus process')
    parser.add_argument('-m', '--message', action='store', required=True, \
            help='message to send')
    parser.add_argument('-p', '--port', action='store', type=int, default=9999, \
            help='remote process communication port number')
    args = parser.parse_args()

    client = RemoteProcessCommunicationClient(args.port)
    response = client.send(args.message)

    print(response)

    return 0
