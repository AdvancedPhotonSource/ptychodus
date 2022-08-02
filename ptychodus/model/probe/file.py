from __future__ import annotations
from pathlib import Path
import logging

import numpy

from ...api.plugins import PluginChooser
from ...api.probe import ProbeArrayType, ProbeFileReader
from .settings import ProbeSettings
from .sizer import ProbeSizer

logger = logging.getLogger(__name__)


class FileProbeInitializer:

    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer,
                 fileReaderChooser: PluginChooser[ProbeFileReader]) -> None:
        self._settings = settings
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser

    def __call__(self) -> ProbeArrayType:
        array = numpy.zeros(self._sizer.getProbeExtent().shape, dtype=complex)

        self._fileReaderChooser.setFromSimpleName(self._settings.inputFileType.value)
        fileReader = self._fileReaderChooser.getCurrentStrategy()
        fileType = self._fileReaderChooser.getCurrentSimpleName()
        filePath = self._settings.inputFilePath.value

        if filePath is None:
            logger.error(f'Path is None!')
        elif filePath.is_file():
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            array = fileReader.read(filePath)
        else:
            logger.error(f'Path \"{filePath}\" is not a file!')

        return array
