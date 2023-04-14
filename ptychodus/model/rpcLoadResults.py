from __future__ import annotations
from pathlib import Path
from typing import Any
import logging

import numpy

from ..api.rpc import RPCMessage, RPCExecutor
from .object import ObjectAPI
from .probe import Probe

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

    def __init__(self, probe: Probe, objectAPI: ObjectAPI) -> None:
        self._probe = probe
        self._objectAPI = objectAPI

    def submit(self, message: RPCMessage) -> None:
        if isinstance(message, LoadResultsMessage):
            if message.filePath.is_file():
                logger.debug(f'Loading results from {message.filePath}')
                results = numpy.load(message.filePath)
                self._probe.setArray(results['probe'])
                objectItem = self._objectAPI.insertItemIntoRepositoryFromArray(
                    message.filePath.stem,
                    results['object'],
                    filePath=message.filePath,
                    simpleFileType='NPZ')

                if objectItem is None:
                    logger.error('Failed to load object result!')
                else:
                    self._objectAPI.selectItem(objectItem)
            else:
                logger.debug(f'{message.filePath} is not a file.')
