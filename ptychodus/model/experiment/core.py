from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging
import sys
import threading

from ...api.experiment import ExperimentFileReader, ExperimentFileWriter, ExperimentMetadata
from ...api.geometry import Box2D
from ...api.observer import Observable, ObservableSequence, Observer
from ...api.parametric import ParametricBase
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry
from ..object import ObjectRepositoryItem
from ..patterns import PatternSizer
from ..probe import ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .metadata import MetadataRepository
from .repository import ExperimentRepository, ExperimentRepositoryObserver
from .settings import ExperimentSettings


class ExperimentCore:

    def __init__(self, settingsRegistry: SettingsRegistry, patternSizer: PatternSizer,
                 fileReaderChooser: PluginChooser[ExperimentFileReader],
                 fileWriterChooser: PluginChooser[ExperimentFileWriter]) -> None:
        self.settings = ExperimentSettings.createInstance(settingsRegistry)
        self.repository = ExperimentRepository(patternSizer, fileReaderChooser, fileWriterChooser)
        self.metadataRepository = MetadataRepository(self.repository)
