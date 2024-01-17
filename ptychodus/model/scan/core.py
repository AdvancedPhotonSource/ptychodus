from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanFileReader, ScanFileWriter, ScanPoint
from ...api.settings import SettingsRegistry
from ..patterns import ActiveDiffractionDataset
from .factory import ScanBuilderFactory
from .settings import ScanSettings
from .streaming import StreamingScanBuilder

logger = logging.getLogger(__name__)


class ScanCore:

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 dataset: ActiveDiffractionDataset,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._builder = StreamingScanBuilder()  # FIXME
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self._builderFactory = ScanBuilderFactory(self._settings, fileReaderChooser, fileWriterChooser)
