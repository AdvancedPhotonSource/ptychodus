from __future__ import annotations
from types import ModuleType, TracebackType
from typing import overload
import importlib
import logging
import pkgutil

import numpy

from ..api.observer import *
from ..api.settings import *
from .crop import *
from .data import *
from .detector import *
from .image import *
from .object import *
from .probe import *
from .reconstructor import *
from .scan import *
from .velociprobe import *

import ptychodus.plugins.data_readers
import ptychodus.plugins.scan_readers
import ptychodus.plugins.probe_readers
import ptychodus.plugins.object_readers

logger = logging.getLogger(__name__)


class ModelCore:
    @staticmethod
    def loadPlugins(ns_pkg: ModuleType) -> list:  # TODO typing
        plugin_list = list()

        # Specifying the second argument (prefix) to iter_modules makes the
        # returned name an absolute name instead of a relative one. This allows
        # import_module to work without having to do additional modification to
        # the name.
        for moduleInfo in pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + '.'):
            logger.info(f'Importing plugin {moduleInfo.name}')
            module = importlib.import_module(moduleInfo.name)
            plugin_list.extend(module.registrable_plugins())

        return plugin_list

    def __init__(self, isDeveloperModeEnabled: bool = False) -> None:
        self.rng = numpy.random.default_rng()

        self.settingsRegistry = SettingsRegistry()
        self._dataSettings = DataSettings.createInstance(self.settingsRegistry)
        self._detectorSettings = DetectorSettings.createInstance(self.settingsRegistry)
        self._cropSettings = CropSettings.createInstance(self.settingsRegistry)
        self._scanSettings = ScanSettings.createInstance(self.settingsRegistry)
        self._probeSettings = ProbeSettings.createInstance(self.settingsRegistry)
        self._objectSettings = ObjectSettings.createInstance(self.settingsRegistry)
        self._reconstructorSettings = ReconstructorSettings.createInstance(self.settingsRegistry)

        self._detector = Detector.createInstance(self._detectorSettings)
        self._cropSizer = CropSizer.createInstance(self._cropSettings, self._detector)

        self._scan = Scan.createInstance(self._scanSettings)
        scanFileReaderList = ModelCore.loadPlugins(ptychodus.plugins.scan_readers)
        self._fileScanInitializer = FileScanInitializer.createInstance(
            self._scanSettings, scanFileReaderList)
        self._scanInitializer = ScanInitializer.createInstance(self.rng, self._scanSettings,
                                                               self._scan,
                                                               self._fileScanInitializer,
                                                               self.settingsRegistry)

        self._probeSizer = ProbeSizer.createInstance(self._probeSettings, self._cropSizer)
        self._probe = Probe(self._probeSettings, self._probeSizer)
        probeFileReaderList = ModelCore.loadPlugins(ptychodus.plugins.probe_readers)
        self._fileProbeInitializer = FileProbeInitializer.createInstance(
            self._probeSettings, self._probeSizer, probeFileReaderList)
        self._probeInitializer = ProbeInitializer.createInstance(self._detectorSettings,
                                                                 self._probeSettings,
                                                                 self._probeSizer, self._probe,
                                                                 self._fileProbeInitializer,
                                                                 self.settingsRegistry)

        self._objectSizer = ObjectSizer.createInstance(self._detector, self._cropSizer, self._scan,
                                                       self._probeSizer)
        self._object = Object(self._objectSettings, self._objectSizer)
        objectFileReaderList = ModelCore.loadPlugins(ptychodus.plugins.object_readers)
        self._fileObjectInitializer = FileObjectInitializer.createInstance(
            self._objectSettings, self._objectSizer, objectFileReaderList)
        self._objectInitializer = ObjectInitializer.createInstance(self.rng, self._objectSettings,
                                                                   self._objectSizer, self._object,
                                                                   self._fileObjectInitializer,
                                                                   self.settingsRegistry)

        self._velociprobeReader = None  # FIXME
        self._velociprobeImageSequence = VelociprobeImageSequence.createInstance(
            self._velociprobeReader)
        self._croppedImageSequence = CroppedImageSequence.createInstance(
            self._cropSizer, self._velociprobeImageSequence)
        self._dataDirectoryWatcher = DataDirectoryWatcher()
        self.reconstructorPlotPresenter = ReconstructorPlotPresenter()

        self.ptychopyBackend = PtychoPyBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self.tikeBackend = TikeBackend.createInstance(self.settingsRegistry, self._cropSizer,
                                                      self._velociprobeReader, self._scan,
                                                      self._probeSizer, self._probe,
                                                      self._objectSizer, self._object,
                                                      self.reconstructorPlotPresenter,
                                                      isDeveloperModeEnabled)
        self.ptychonnBackend = PtychoNNBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self._selectableReconstructor = SelectableReconstructor.createInstance(
            self._reconstructorSettings, self.ptychopyBackend.reconstructorList +
            self.tikeBackend.reconstructorList + self.ptychonnBackend.reconstructorList)

        dataFileReaderList = ModelCore.loadPlugins(ptychodus.plugins.data_readers)
        self.dataFilePresenter = DataFilePresenter.createInstance(self._dataSettings,
                                                                  dataFileReaderList)
        self.detectorPresenter = DetectorPresenter.createInstance(self._detectorSettings)
        self.cropPresenter = CropPresenter.createInstance(self._cropSettings, self._cropSizer)
        self.detectorImagePresenter = DetectorImagePresenter.createInstance(
            self._croppedImageSequence)
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
