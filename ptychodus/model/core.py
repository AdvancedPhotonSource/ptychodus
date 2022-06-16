from __future__ import annotations
from types import TracebackType
from typing import overload
import logging

import numpy

from ..api.plugins import PluginRegistry
from ..api.settings import SettingsRegistry
from .data import *
from .detector import *
from .image import *
from .ipc import InterProcessCommunicationServer
from .object import *
from .probe import *
from .ptychonn import PtychoNNBackend
from .ptychopy import PtychoPyBackend
from .reconstructor import *
from .scan import *
from .tike import TikeBackend
from .velociprobe import *
from .watcher import DataDirectoryWatcher

import ptychodus.plugins

logger = logging.getLogger(__name__)


class ModelCore:

    def __init__(self, ipcPort: int, isDeveloperModeEnabled: bool = False) -> None:
        self.rng = numpy.random.default_rng()
        self._pluginRegistry = PluginRegistry.loadPlugins()
        self._colormapChooserFactory = ColormapChooserFactory()

        self.detectorImagePresenter = ImagePresenter.createInstance(
            self._colormapChooserFactory, self._pluginRegistry.buildScalarTransformationChooser(),
            self._pluginRegistry.buildComplexToRealStrategyChooser())
        self.probeImagePresenter = ImagePresenter.createInstance(
            self._colormapChooserFactory, self._pluginRegistry.buildScalarTransformationChooser(),
            self._pluginRegistry.buildComplexToRealStrategyChooser())
        self.objectImagePresenter = ImagePresenter.createInstance(
            self._colormapChooserFactory, self._pluginRegistry.buildScalarTransformationChooser(),
            self._pluginRegistry.buildComplexToRealStrategyChooser())

        self.settingsRegistry = SettingsRegistry()
        self._dataSettings = DataSettings.createInstance(self.settingsRegistry)
        self._detectorSettings = DetectorSettings.createInstance(self.settingsRegistry)
        self._cropSettings = CropSettings.createInstance(self.settingsRegistry)
        self._scanSettings = ScanSettings.createInstance(self.settingsRegistry)
        self._probeSettings = ProbeSettings.createInstance(self.settingsRegistry)
        self._objectSettings = ObjectSettings.createInstance(self.settingsRegistry)
        self._reconstructorSettings = ReconstructorSettings.createInstance(self.settingsRegistry)

        self._ipcServer = InterProcessCommunicationServer(ipcPort)
        self._dataDirectoryWatcher = DataDirectoryWatcher.createInstance(self._dataSettings)

        self._detector = Detector.createInstance(self._detectorSettings)
        self._cropSizer = CropSizer.createInstance(self._cropSettings, self._detector)

        self._scan = Scan.createInstance(self._scanSettings)
        self._fileScanInitializer = FileScanInitializer.createInstance(
            self._scanSettings, self._pluginRegistry.buildScanFileReaderChooser())
        self._scanInitializer = ScanInitializer.createInstance(
            self.rng, self._scanSettings, self._scan, self._fileScanInitializer,
            self._pluginRegistry.buildScanFileWriterChooser(), self.settingsRegistry)

        self._probeSizer = ProbeSizer.createInstance(self._probeSettings, self._cropSizer)
        self._probe = Probe(self._probeSettings, self._probeSizer)
        self._fileProbeInitializer = FileProbeInitializer.createInstance(
            self._probeSettings, self._probeSizer,
            self._pluginRegistry.buildProbeFileReaderChooser())
        self._probeInitializer = ProbeInitializer.createInstance(
            self._detector, self._probeSettings, self._probeSizer,
            self._probe, self._fileProbeInitializer,
            self._pluginRegistry.buildProbeFileWriterChooser(), self.settingsRegistry)

        self._objectSizer = ObjectSizer.createInstance(self._detector, self._cropSizer, self._scan,
                                                       self._probeSizer)
        self._object = Object(self._objectSettings, self._objectSizer)
        self._fileObjectInitializer = FileObjectInitializer.createInstance(
            self._objectSettings, self._objectSizer,
            self._pluginRegistry.buildObjectFileReaderChooser())
        self._objectInitializer = ObjectInitializer.createInstance(
            self.rng, self._objectSettings, self._objectSizer,
            self._object, self._fileObjectInitializer,
            self._pluginRegistry.buildObjectFileWriterChooser(), self.settingsRegistry)

        self._activeDataFile = ActiveDataFile(self._dataSettings, self._cropSizer)

        self.reconstructorPlotPresenter = ReconstructorPlotPresenter()

        self.ptychopyBackend = PtychoPyBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self.tikeBackend = TikeBackend.createInstance(self.settingsRegistry, self._activeDataFile,
                                                      self._scan, self._probeSizer, self._probe,
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
            self._pluginRegistry.buildDataFileReaderChooser(),
            self._pluginRegistry.buildDataFileWriterChooser())
        self.detectorPresenter = DetectorPresenter.createInstance(self._detectorSettings)
        self.cropPresenter = CropPresenter.createInstance(self._cropSettings, self._cropSizer)
        self.diffractionDatasetPresenter = DiffractionDatasetPresenter.createInstance(
            self._activeDataFile)
        self._velociprobeReader = next(entry.strategy
                                       for entry in self._pluginRegistry.dataFileReaders
                                       if type(entry.strategy).__name__ ==
                                       'VelociprobeDataFileReader')  # TODO remove when able
        self.velociprobePresenter = VelociprobePresenter.createInstance(
            self._velociprobeReader, self._detectorSettings, self._cropSettings,
            self._probeSettings, self._activeDataFile, self._scanInitializer)
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
        self._ipcServer.start()
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
        self._ipcServer.stop()

    def batchModeReconstruct(self) -> int:
        outputFilePath = self._reconstructorSettings.outputFilePath.value

        if outputFilePath.exists():
            logger.error('Output file path already exists!')
            return -1

        result = self.reconstructorPresenter.reconstruct()

        pixelSizeXInMeters = float(self._objectSizer.getPixelSizeXInMeters())
        pixelSizeYInMeters = float(self._objectSizer.getPixelSizeYInMeters())

        scanXInMeters = [float(point.x) for point in self._scan]
        scanYInMeters = [float(point.y) for point in self._scan]

        dataDump = dict()
        dataDump['pixelSizeInMeters'] = numpy.array([pixelSizeYInMeters, pixelSizeXInMeters])
        dataDump['scanInMeters'] = numpy.column_stack((scanYInMeters, scanXInMeters))
        dataDump['probe'] = self._probe.getArray()
        dataDump['object'] = self._object.getArray()
        numpy.savez(outputFilePath, **dataDump)

        return result
