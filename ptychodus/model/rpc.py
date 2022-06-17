from __future__ import annotations
from pathlib import Path
import argparse
import json
import logging
import queue
import socket
import socketserver
import threading
import typing

from ..api.rpc import RPCMessage, RPCMessageHandler

logger = logging.getLogger(__name__)


class LoadResultsMessage(RPCMessage):

    def __init__(self, filePath: Path) -> None:
        self._filePath = filePath

    @classmethod
    @property
    def messageType(cls) -> int:
        return 100

    @classmethod
    def fromDict(cls, values: dict[str, typing.Any]) -> LoadResultsMessage:
        filePath = Path(values['filePath'])
        return cls(filePath)

    def toDict(self) -> dict[str, typing.Any]:
        result = super().toDict()
        result['filePath'] = str(self._filePath)
        return result

    @property
    def filePath(self) -> Path:
        return self._filePath


class RPCStreamRequestHandler(socketserver.StreamRequestHandler):

    def handle(self):
        message = self.rfile.readline().decode('utf-8').strip()
        logger.debug(f'RECV: \"{message}\" from {self.client_address[0]}')
        response = self.server.processMessage(message)
        logger.debug(f'SEND: \"{response}\" to {self.client_address[0]}')
        self.wfile.write(response.encode('utf-8'))


class RPCSocketServer(socketserver.TCPServer):

    def __init__(self, portNumber: int, messageQueue: queue.Queue) -> None:
        super().__init__(('127.0.0.1', portNumber), RPCStreamRequestHandler)
        self._messageQueue = messageQueue

    def processMessage(self, message: str) -> str:
        try:
            messageDict = json.loads(message)
        except json.JSONDecodeError:
            logger.debug(f'Failed to decode JSON: \"{message}\"!')
            return 'FAILURE'

        try:
            messageType = messageDict['messageType']
        except KeyError:
            logger.debug(f'Missing messageType information: \"{message}\"!')
            return 'FAILURE'

        try:
            # TODO generalize to support other message types
            messageObject = LoadResultsMessage.fromDict(messageDict)
        except:
            logger.debug(f'Exception while creating message object: \"{message}\"!')
            return 'FAILURE'

        self._messageQueue.put(messageObject)

        return 'SUCCESS'


class RPCMessageService:

    def __init__(self, portNumber: int) -> None:
        self._messageQueue: queue.Queue[RPCMessage] = queue.Queue()
        self._tcpServer = RPCSocketServer(portNumber, self._messageQueue)
        self._producerThread = threading.Thread(target=self._tcpServer.serve_forever)
        self._consumerThread = threading.Thread(target=self._processMessages)
        self._consumerStopEvent = threading.Event()
        self._messageHandlers: dict[int, RPCMessageHandler] = dict()

    def registerMessageHandler(self, messageType: int, handler: RPCMessageHandler) -> None:
        self._messageHandlers[messageType] = handler

    def start(self) -> None:
        logger.debug('Starting message service...')

        if self._producerThread.is_alive():
            logger.debug('Producer thread is already alive!')
        else:
            self._producerThread.start()
            logger.debug('Producer thread has started.')

        if self._consumerThread.is_alive():
            logger.debug('Consumer thread is already alive!')
        else:
            self._consumerThread.start()
            logger.debug('Consumer thread has started.')

        logger.debug(f'Message service is started on {self._tcpServer.server_address}.')

    def _processMessages(self) -> None:
        while not self._consumerStopEvent.is_set():
            try:
                message = self._messageQueue.get(block=True, timeout=1)
            except queue.Empty:
                pass
            else:
                handler = self._messageHandlers.get(message.messageType)

                if handler is None:
                    logger.debug(f'No handler for message type {message.messageType}')
                else:
                    logger.debug(f'Processing message type {message.messageType}')
                    handler.handleMessage(message)

                self._messageQueue.task_done()

    def stop(self) -> None:
        logger.debug('Stopping message service...')
        self._tcpServer.shutdown()
        self._tcpServer.server_close()
        self._consumerStopEvent.set()
        self._consumerThread.join()
        logger.debug('Message service stopped.')


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


def main() -> int:
    parser = argparse.ArgumentParser(
        prog='ptychodus-rpc',
        description='ptychodus-rpc communicates with an active ptychodus process')
    parser.add_argument('-m', '--message', action='store', required=True, \
            help='message to send')
    parser.add_argument('-p', '--port', action='store', type=int, default=9999, \
            help='remote process communication port number')
    args = parser.parse_args()

    client = RPCMessageClient(args.port)
    response = client.send(args.message)

    print(response)

    return 0
