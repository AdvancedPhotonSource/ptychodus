from __future__ import annotations
from types import TracebackType
from typing import overload
import logging

import numpy

from ..api.plugins import PluginRegistry
from ..api.settings import SettingsRegistry
from .data import *
from .detector import *
from .object import *
from .probe import *
from .ptychonn import PtychoNNBackend
from .ptychopy import PtychoPyBackend
from .reconstructor import *
from .scan import *
from .tike import TikeBackend
from .velociprobe import *

import ptychodus.plugins

logger = logging.getLogger(__name__)


class ModelCore:
    def __init__(self, isDeveloperModeEnabled: bool = False) -> None:
        self.rng = numpy.random.default_rng()

        self._pluginRegistry = PluginRegistry.loadPlugins()

        self.settingsRegistry = SettingsRegistry()
        self._dataSettings = Datasettings.createInstance(self.settingsRegistry)
        self._detectorSettings = DetectorSettings.createInstance(self.settingsRegistry)
        self._cropSettings = CropSettings.createInstance(self.settingsRegistry)
        self._scanSettings = ScanSettings.createInstance(self.settingsRegistry)
        self._probeSettings = ProbeSettings.createInstance(self.settingsRegistry)
        self._objectSettings = ObjectSettings.createInstance(self.settingsRegistry)
        self._reconstructorSettings = ReconstructorSettings.createInstance(self.settingsRegistry)

        self._dataDirectoryWatcher = DataDirectoryWatcher()
        self._detector = Detector.createInstance(self._detectorSettings)
        self._cropSizer = CropSizer.createInstance(self._cropSettings, self._detector)

        self._scan = Scan.createInstance(self._scanSettings)
        self._fileScanInitializer = FileScanInitializer.createInstance(
            self._scanSettings, self._pluginRegistry.buildScanFileReaderChooser())
        self._scanInitializer = ScanInitializer.createInstance(self.rng, self._scanSettings,
                                                               self._scan,
                                                               self._fileScanInitializer,
                                                               self.settingsRegistry)

        self._probeSizer = ProbeSizer.createInstance(self._probeSettings, self._cropSizer)
        self._probe = Probe(self._probeSettings, self._probeSizer)
        self._fileProbeInitializer = FileProbeInitializer.createInstance(
            self._probeSettings, self._probeSizer,
            self._pluginRegistry.buildProbeFileReaderChooser())
        self._probeInitializer = ProbeInitializer.createInstance(self._detector,
                                                                 self._probeSettings,
                                                                 self._probeSizer, self._probe,
                                                                 self._fileProbeInitializer,
                                                                 self.settingsRegistry)

        self._objectSizer = ObjectSizer.createInstance(self._detector, self._cropSizer, self._scan,
                                                       self._probeSizer)
        self._object = Object(self._objectSettings, self._objectSizer)
        self._fileObjectInitializer = FileObjectInitializer.createInstance(
            self._objectSettings, self._objectSizer,
            self._pluginRegistry.buildObjectFileReaderChooser())
        self._objectInitializer = ObjectInitializer.createInstance(self.rng, self._objectSettings,
                                                                   self._objectSizer, self._object,
                                                                   self._fileObjectInitializer,
                                                                   self.settingsRegistry)

        self._velociprobeReader = next(entry.strategy
                                       for entry in self._pluginRegistry.dataFileReaders
                                       if type(entry.strategy).__name__ ==
                                       'VelociprobeDataFileReader')  # TODO remove when able

        self._activeDataFile = ActiveDataFile()
        self._activeDiffractionDataset = ActiveDiffractionDataset.createInstance(
            self._activeDataFile, self._cropSizer)

        self.reconstructorPlotPresenter = ReconstructorPlotPresenter()

        self.ptychopyBackend = PtychoPyBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self.tikeBackend = TikeBackend.createInstance(self.settingsRegistry, self._cropSizer,
                                                      self._activeDataFile, self._scan,
                                                      self._probeSizer, self._probe,
                                                      self._objectSizer, self._object,
                                                      self.reconstructorPlotPresenter,
                                                      isDeveloperModeEnabled)
        self.ptychonnBackend = PtychoNNBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self._selectableReconstructor = SelectableReconstructor.createInstance(
            self._reconstructorSettings, self.ptychopyBackend.reconstructorList +
            self.tikeBackend.reconstructorList + self.ptychonnBackend.reconstructorList)

        self.dataFilePresenter = DataFilePresenter.createInstance(
            self._dataSettings, self._activeDataFile,
            self._pluginRegistry.buildDataFileReaderChooser())
        self.detectorPresenter = DetectorPresenter.createInstance(self._detectorSettings)
        self.cropPresenter = CropPresenter.createInstance(self._cropSettings, self._cropSizer)
        self.detectorImagePresenter = DetectorImagePresenter.createInstance(
            self._activeDiffractionDataset)
        self.velociprobePresenter = VelociprobePresenter.createInstance(
            self._velociprobeReader, self._detectorSettings, self._cropSettings,
            self._probeSettings)
        self.probePresenter = ProbePresenter.createInstance(self._probeSettings, self._probeSizer,
                                                            self._probe, self._probeInitializer)
        self.scanPresenter = ScanPresenter.createInstance(self._scanSettings, self._scan,
                                                          self._scanInitializer)
        self.objectPresenter = ObjectPresenter.createInstance(self._objectSettings,
                                                              self._objectSizer, self._object,
                                                              self._objectInitializer)
        self.reconstructorPresenter = ReconstructorPresenter.createInstance(
            self._reconstructorSettings, self._selectableReconstructor)

    def __enter__(self) -> ModelCore:
        self._dataDirectoryWatcher.start()
        return self

    @overload
    def __exit__(self, exception_type: None, exception_value: None, traceback: None) -> None:
        ...

    @overload
    def __exit__(self, exception_type: type[BaseException], exception_value: BaseException,
                 traceback: TracebackType) -> None:
        ...

    def __exit__(self, exception_type: type[BaseException] | None,
                 exception_value: BaseException | None, traceback: TracebackType | None) -> None:
        self._dataDirectoryWatcher.stop()
        self._dataDirectoryWatcher.join()
