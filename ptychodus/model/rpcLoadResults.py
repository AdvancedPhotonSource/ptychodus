from __future__ import annotations
from pathlib import Path
from typing import Any
import logging

import numpy

from ..api.rpc import RPCMessage, RPCExecutor
from .object import Object
from .probe import Probe

logger = logging.getLogger(__name__)


class LoadResultsMessage(RPCMessage):

    def __init__(self, filePath: Path) -> None:
        self._filePath = filePath

    @classmethod
    @property
    def procedure(cls) -> str:
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

    def __init__(self, probe: Probe, object_: Object) -> None:
        self._probe = probe
        self._object = object_

    def submit(self, message: RPCMessage) -> None:
        if isinstance(message, LoadResultsMessage):
            if message.filePath.is_file():
                logger.debug(f'Loading results from {message.filePath}')
                results = numpy.load(message.filePath)
                self._probe.setArray(results['probe'])
                self._object.setArray(results['object'])
            else:
                logger.debug(f'{message.filePath} is not a file.')
