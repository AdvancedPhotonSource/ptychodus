from __future__ import annotations
from pathlib import Path
import argparse
import json
import logging
import queue
import socket
import socketserver
import threading

from ..api.rpc import RPCMessage, RPCExecutor

logger = logging.getLogger(__name__)


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
        self._messageClasses: dict[str, type[RPCMessage]] = dict()

    def registerMessageClass(self, messageClass: type[RPCMessage]) -> None:
        self._messageClasses[messageClass.procedure] = messageClass

    def processMessage(self, message: str) -> str:
        try:
            messageDict = json.loads(message)
        except json.JSONDecodeError:
            logger.debug(f'Failed to decode JSON: \"{message}\"!')
            return 'FAILURE'

        try:
            procedure = messageDict['procedure']
        except KeyError:
            logger.debug(f'Missing procedure information: \"{message}\"!')
            return 'FAILURE'

        try:
            MessageClass = self._messageClasses[procedure]
        except KeyError:
            logger.debug(f'Missing message class for \"{procedure}\"')
            return 'FAILURE'

        try:
            messageObject = MessageClass.fromDict(messageDict)
        except:
            logger.debug(f'Exception while creating message object: \"{message}\"!')
            return 'FAILURE'

        self._messageQueue.put(messageObject)

        return 'SUCCESS'


class RPCMessageService:

    def __init__(self, portNumber: int) -> None:
        self._messageQueue: queue.Queue[RPCMessage] = queue.Queue()
        self._socketServer = RPCSocketServer(portNumber, self._messageQueue)
        self._producerThread = threading.Thread(target=self._socketServer.serve_forever)
        self._consumerThread = threading.Thread(target=self._processMessages)
        self._consumerStopEvent = threading.Event()
        self._executors: dict[str, RPCExecutor] = dict()

    def registerMessageClass(self, messageClass: type[RPCMessage]) -> None:
        self._socketServer.registerMessageClass(messageClass)

    def registerExecutor(self, procedure: str, executor: RPCExecutor) -> None:
        self._executors[procedure] = executor

    def start(self) -> None:
        logger.info('Starting message service...')
        self._producerThread.start()
        self._consumerThread.start()
        logger.info(f'Message service is started on {self._socketServer.server_address}.')

    def _processMessages(self) -> None:
        while not self._consumerStopEvent.is_set():
            try:
                message = self._messageQueue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            try:
                executor = self._executors[message.procedure]
            except KeyError:
                logger.debug(f'No executor for \"{message.procedure}\" procedure')
            else:
                logger.debug(f'Executing \"{message.procedure}\" procedure')
                executor.submit(message)
            finally:
                self._messageQueue.task_done()

    def stop(self) -> None:
        logger.info('Stopping message service...')
        self._socketServer.shutdown()
        self._socketServer.server_close()
        self._producerThread.join()
        self._consumerStopEvent.set()
        self._consumerThread.join()
        logger.info('Message service stopped.')


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
