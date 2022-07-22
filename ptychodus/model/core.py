from __future__ import annotations
from dataclasses import dataclass
from types import TracebackType
from typing import overload
import logging

import numpy

from ..api.plugins import PluginRegistry
from ..api.settings import SettingsRegistry
from .data import *
from .detector import *
from .image import *
from .object import *
from .probe import *
from .ptychonn import PtychoNNBackend
from .ptychopy import PtychoPyBackend
from .reconstructor import *
from .rpc import RPCMessageService
from .rpcLoadResults import LoadResultsExecutor, LoadResultsMessage
from .scan import ScanCore, ScanPresenter
from .tike import TikeBackend
from .velociprobe import *
from .watcher import DataDirectoryWatcher
from .workflow import WorkflowPresenter, WorkflowSettings

import ptychodus.plugins

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArgs:
    rpcPort: int
    autoExecuteRPCs: bool
    replacementPathPrefix: Optional[str]
    isDeveloperModeEnabled: bool = False


class ModelCore:

    def __init__(self, modelArgs: ModelArgs) -> None:
        self.rng = numpy.random.default_rng()
        self._pluginRegistry = PluginRegistry.loadPlugins()

        self._detectorImageCore = ImageCore(
            self._pluginRegistry.buildScalarTransformationChooser())
        self._probeImageCore = ImageCore(self._pluginRegistry.buildScalarTransformationChooser())
        self._objectImageCore = ImageCore(self._pluginRegistry.buildScalarTransformationChooser())

        self.settingsRegistry = SettingsRegistry(modelArgs.replacementPathPrefix)
        self._dataSettings = DataSettings.createInstance(self.settingsRegistry)
        self._detectorSettings = DetectorSettings.createInstance(self.settingsRegistry)
        self._cropSettings = CropSettings.createInstance(self.settingsRegistry)
        self._probeSettings = ProbeSettings.createInstance(self.settingsRegistry)
        self._objectSettings = ObjectSettings.createInstance(self.settingsRegistry)
        self._reconstructorSettings = ReconstructorSettings.createInstance(self.settingsRegistry)
        self._workflowSettings = WorkflowSettings.createInstance(self.settingsRegistry)

        # TODO DataDirectoryWatcher should be optional
        self._dataDirectoryWatcher = DataDirectoryWatcher.createInstance(self._dataSettings)

        self._detector = Detector.createInstance(self._detectorSettings)
        self._cropSizer = CropSizer.createInstance(self._cropSettings, self._detector)

        self._probeSizer = ProbeSizer.createInstance(self._probeSettings, self._cropSizer)
        self._probe = Probe(self._probeSettings, self._probeSizer)
        self._fileProbeInitializer = FileProbeInitializer.createInstance(
            self._probeSettings, self._probeSizer,
            self._pluginRegistry.buildProbeFileReaderChooser())
        self._probeInitializer = ProbeInitializer.createInstance(
            self._detector, self._probeSettings, self._probeSizer,
            self._probe, self._fileProbeInitializer,
            self._pluginRegistry.buildProbeFileWriterChooser(), self.settingsRegistry)

        self._scanCore = ScanCore(self.rng, self.settingsRegistry,
                                  self._pluginRegistry.buildScanFileReaderChooser(),
                                  self._pluginRegistry.buildScanFileWriterChooser())

        self._objectSizer = ObjectSizer.createInstance(self._detector, self._cropSizer,
                                                       self._scanCore.scan, self._probeSizer)
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
                                                              modelArgs.isDeveloperModeEnabled)
        self.tikeBackend = TikeBackend.createInstance(self.settingsRegistry, self._activeDataFile,
                                                      self._scanCore.scan, self._probeSizer,
                                                      self._probe, self._objectSizer, self._object,
                                                      self.reconstructorPlotPresenter,
                                                      modelArgs.isDeveloperModeEnabled)
        self.ptychonnBackend = PtychoNNBackend.createInstance(self.settingsRegistry,
                                                              modelArgs.isDeveloperModeEnabled)
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
            self._probeSettings, self._activeDataFile, self._scanCore.scan)
        self.probePresenter = ProbePresenter.createInstance(self._probeSettings, self._probeSizer,
                                                            self._probe, self._probeInitializer)
        self.objectPresenter = ObjectPresenter.createInstance(self._objectSettings,
                                                              self._objectSizer, self._object,
                                                              self._objectInitializer)
        self.reconstructorPresenter = ReconstructorPresenter.createInstance(
            self._reconstructorSettings, self._selectableReconstructor)
        self.workflowPresenter = WorkflowPresenter.createInstance(self._workflowSettings)

        if modelArgs.rpcPort >= 0:
            self._loadResultsExecutor = LoadResultsExecutor(self._probe, self._object)
            self.rpcMessageService = RPCMessageService(modelArgs.rpcPort,
                                                       modelArgs.autoExecuteRPCs)
            self.rpcMessageService.registerMessageClass(LoadResultsMessage)
            self.rpcMessageService.registerExecutor(LoadResultsMessage.procedure,
                                                    self._loadResultsExecutor)
        else:
            self.rpcMessageService = None

    def __enter__(self) -> ModelCore:
        if self.rpcMessageService:
            self.rpcMessageService.start()

        if self._dataDirectoryWatcher:
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
        if self._dataDirectoryWatcher:
            self._dataDirectoryWatcher.stop()

        if self.rpcMessageService:
            self.rpcMessageService.stop()

    def batchModeReconstruct(self) -> int:
        result = self.reconstructorPresenter.reconstruct()

        pixelSizeXInMeters = float(self._objectSizer.getPixelSizeXInMeters())
        pixelSizeYInMeters = float(self._objectSizer.getPixelSizeYInMeters())

        scanXInMeters = [float(point.x) for point in self._scanCore.scan]
        scanYInMeters = [float(point.y) for point in self._scanCore.scan]

        # TODO document output file format; include cost function values
        dataDump = dict()
        dataDump['pixelSizeInMeters'] = numpy.array([pixelSizeYInMeters, pixelSizeXInMeters])
        dataDump['scanInMeters'] = numpy.column_stack((scanYInMeters, scanXInMeters))
        dataDump['probe'] = self._probe.getArray()
        dataDump['object'] = self._object.getArray()
        numpy.savez(self._reconstructorSettings.outputFilePath.value, **dataDump)

        return result

    @property
    def detectorImagePresenter(self) -> ImagePresenter:
        return self._detectorImageCore.presenter

    @property
    def probeImagePresenter(self) -> ImagePresenter:
        return self._probeImageCore.presenter

    @property
    def objectImagePresenter(self) -> ImagePresenter:
        return self._objectImageCore.presenter

    @property
    def scanPresenter(self) -> ScanPresenter:
        return self._scanCore.scanPresenter
