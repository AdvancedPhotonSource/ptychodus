from __future__ import annotations
import logging
import queue
import threading

from ...api.rpc import RPCMessage, RPCExecutor
from .server import RPCSocketServer

logger = logging.getLogger(__name__)


class RPCMessageService:

    def __init__(self, rpcPort: int, autoExecuteRPCs: bool) -> None:
        self._messageQueue: queue.Queue[RPCMessage] = queue.Queue()
        self._socketServer = RPCSocketServer(rpcPort, self._messageQueue)
        self._autoExecuteRPCs = autoExecuteRPCs
        self._producerThread = threading.Thread(target=self._socketServer.serve_forever)
        self._consumerThread = threading.Thread(target=self._processMessagesForever)
        self._consumerStopEvent = threading.Event()
        self._executors: dict[str, RPCExecutor] = dict()

    def getPortNumber(self) -> int:
        return self._socketServer.server_address[1]

    def registerProcedure(self, messageClass: type[RPCMessage], executor: RPCExecutor) -> None:
        self._socketServer.registerMessageClass(messageClass)
        self._executors[messageClass.getProcedure()] = executor

    @property
    def isActive(self) -> bool:
        return self._producerThread.is_alive()

    def start(self) -> None:
        logger.info('Starting message service...')
        self._producerThread.start()

        if self._autoExecuteRPCs:
            self._consumerThread.start()

        logger.info(f'Message service is started on {self._socketServer.server_address}.')

    def processMessages(self) -> None:
        while not self._consumerStopEvent.is_set():
            try:
                message = self._messageQueue.get(block=False)
            except queue.Empty:
                break

            self._runProcess(message)

    def _processMessagesForever(self) -> None:
        while not self._consumerStopEvent.is_set():
            try:
                message = self._messageQueue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            self._runProcess(message)

    def _runProcess(self, message: RPCMessage) -> None:
        try:
            executor = self._executors[message.getProcedure()]
        except KeyError:
            logger.debug(f'No executor for \"{message.getProcedure()}\" procedure')
        else:
            logger.debug(f'Executing \"{message.getProcedure()}\" procedure')
            executor.submit(message)
        finally:
            self._messageQueue.task_done()

    def stop(self) -> None:
        logger.info('Stopping message service...')
        self._socketServer.shutdown()
        self._socketServer.server_close()
        self._producerThread.join()

        if self._autoExecuteRPCs:
            self._consumerStopEvent.set()
            self._consumerThread.join()

        logger.info('Message service stopped.')


class RPCCore:

    def __init__(self, rpcPort: int, autoExecuteRPCs: bool) -> None:
        self.messageService = RPCMessageService(rpcPort, autoExecuteRPCs)

    def start(self) -> None:
        if self.messageService.getPortNumber() >= 0:
            self.messageService.start()

    def stop(self) -> None:
        self.messageService.stop()
