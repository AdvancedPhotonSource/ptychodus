from __future__ import annotations
from pathlib import Path
from typing import Any
import logging

from ..api.rpc import RPCMessage, RPCExecutor
from ..api.state import StateDataRegistry

logger = logging.getLogger(__name__)


class LoadResultsMessage(RPCMessage):

    def __init__(self, filePath: Path) -> None:
        self._filePath = filePath

    @classmethod
    def getProcedure(cls) -> str:
        return 'LoadResults'

    @classmethod
    def fromDict(cls, values: dict[str, Any]) -> LoadResultsMessage:
        filePath = Path(values['filePath'])
        return cls(filePath)

    def toDict(self) -> dict[str, Any]:
        result = super().toDict()
        result['filePath'] = str(self._filePath)
        return result

    @property
    def filePath(self) -> Path:
        return self._filePath


class LoadResultsExecutor(RPCExecutor):

    def __init__(self, registry: StateDataRegistry) -> None:
        self._registry = registry

    def submit(self, message: RPCMessage) -> None:
        if isinstance(message, LoadResultsMessage):
            if message.filePath.is_file():
                self._registry.openStateData(message.filePath)
            else:
                logger.debug(f'{message.filePath} is not a file.')
