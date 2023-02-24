import json
import logging
import queue
import socketserver

from ...api.rpc import RPCMessage

logger = logging.getLogger(__name__)


class RPCStreamRequestHandler(socketserver.StreamRequestHandler):

    def handle(self) -> None:
        message = self.rfile.readline().decode('utf-8').strip()
        logger.debug(f'RECV: \"{message}\" from {self.client_address[0]}')

        if isinstance(self.server, RPCSocketServer):
            response = self.server.processMessage(message)
        else:
            logger.error('This should be impossible.')

        logger.debug(f'SEND: \"{response}\" to {self.client_address[0]}')
        self.wfile.write(response.encode('utf-8'))


class RPCSocketServer(socketserver.TCPServer):

    def __init__(self, rpcPort: int, messageQueue: queue.Queue[RPCMessage]) -> None:
        super().__init__(('127.0.0.1', rpcPort), RPCStreamRequestHandler)
        self._messageQueue = messageQueue
        self._messageClasses: dict[str, type[RPCMessage]] = dict()

    def registerMessageClass(self, messageClass: type[RPCMessage]) -> None:
        self._messageClasses[messageClass.getProcedure()] = messageClass

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
