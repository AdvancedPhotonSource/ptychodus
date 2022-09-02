from __future__ import annotations
from dataclasses import dataclass
from importlib.metadata import version
from types import TracebackType
from typing import overload, Optional
import logging
import sys

import h5py
import matplotlib
import numpy

from ..api.plugins import PluginRegistry
from ..api.settings import SettingsRegistry
from .data import *
from .detector import *
from .image import *
from .object import ObjectCore, ObjectPresenter
from .probe import ProbeCore, ProbePresenter
from .ptychonn import PtychoNNBackend
from .ptychopy import PtychoPyBackend
from .reconstructor import *
from .rpc import RPCMessageService
from .rpcLoadResults import LoadResultsExecutor, LoadResultsMessage
from .scan import ScanCore, ScanPresenter
from .tike import TikeBackend
from .velociprobe import *
from .watcher import DataDirectoryWatcher
from .workflow import WorkflowCore, WorkflowPresenter

logger = logging.getLogger(__name__)


def configureLogger() -> None:
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                        stream=sys.stdout,
                        encoding='utf-8',
                        level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.info(f'Ptychodus {version("ptychodus")}')
    logger.info(f'NumPy {version("numpy")}')
    logger.info(f'Matplotlib {version("matplotlib")}')
    logger.info(f'H5Py {version("h5py")}')
    logger.info(f'HDF5 {h5py.version.hdf5_version}')


@dataclass(frozen=True)
class ModelArgs:
    settingsFilePath: Optional[Path]
    replacementPathPrefix: Optional[str]
    rpcPort: int = -1
    autoExecuteRPCs: bool = False
    isDeveloperModeEnabled: bool = False


class ModelCore:

    def __init__(self, modelArgs: ModelArgs) -> None:
        configureLogger()
        self._modelArgs = modelArgs
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
        self._reconstructorSettings = ReconstructorSettings.createInstance(self.settingsRegistry)

        # TODO DataDirectoryWatcher should be optional
        self._dataDirectoryWatcher = DataDirectoryWatcher.createInstance(self._dataSettings)

        self._detector = Detector.createInstance(self._detectorSettings)
        self._cropSizer = CropSizer.createInstance(self._cropSettings, self._detector)
        self._activeDataFile = ActiveDataFile(self._dataSettings, self._cropSizer)

        self._scanCore = ScanCore(self.rng, self.settingsRegistry,
                                  self._pluginRegistry.buildScanFileReaderChooser(),
                                  self._pluginRegistry.buildScanFileWriterChooser())
        self._probeCore = ProbeCore(self.rng, self.settingsRegistry, self._detector,
                                    self._cropSizer,
                                    self._pluginRegistry.buildProbeFileReaderChooser(),
                                    self._pluginRegistry.buildProbeFileWriterChooser())
        self._objectCore = ObjectCore(self.rng, self.settingsRegistry, self._probeCore.apparatus,
                                      self._scanCore.scan, self._probeCore.sizer,
                                      self._pluginRegistry.buildObjectFileReaderChooser(),
                                      self._pluginRegistry.buildObjectFileWriterChooser())

        self.reconstructorPlotPresenter = ReconstructorPlotPresenter()

        self.ptychopyBackend = PtychoPyBackend.createInstance(self.settingsRegistry,
                                                              modelArgs.isDeveloperModeEnabled)
        self.tikeBackend = TikeBackend.createInstance(self.settingsRegistry, self._activeDataFile,
                                                      self._scanCore.scan, self._probeCore.probe,
                                                      self._probeCore.apparatus,
                                                      self._objectCore.object,
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

        # TODO remove velociprobePresenter lookup when able
        self._velociprobeReader = next(
            entry.strategy for entry in self._pluginRegistry.dataFileReaders
            if type(entry.strategy).__name__ == 'VelociprobeDataFileReader')
        self.velociprobePresenter = VelociprobePresenter.createInstance(
            self._velociprobeReader, self._detectorSettings, self._cropSettings,
            self._probeCore.settings, self._activeDataFile, self._scanCore)
        self.reconstructorPresenter = ReconstructorPresenter.createInstance(
            self._reconstructorSettings, self._selectableReconstructor)

        self._workflowCore = WorkflowCore(self.settingsRegistry)

        if modelArgs.rpcPort >= 0:
            self._loadResultsExecutor = LoadResultsExecutor(self._probeCore.probe,
                                                            self._objectCore.object)
            self.rpcMessageService: Optional[RPCMessageService] = RPCMessageService(
                modelArgs.rpcPort, modelArgs.autoExecuteRPCs)
            self.rpcMessageService.registerMessageClass(LoadResultsMessage)
            self.rpcMessageService.registerExecutor(LoadResultsMessage.procedure,
                                                    self._loadResultsExecutor)
        else:
            self.rpcMessageService = None

    def __enter__(self) -> ModelCore:
        if self._modelArgs.settingsFilePath:
            self.settingsRegistry.openSettings(self._modelArgs.settingsFilePath)

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

        pixelSizeXInMeters = float(self._probeCore.apparatus.getObjectPlanePixelSizeXInMeters())
        pixelSizeYInMeters = float(self._probeCore.apparatus.getObjectPlanePixelSizeYInMeters())

        scanXInMeters = [float(point.x) for point in self._scanCore.scan]
        scanYInMeters = [float(point.y) for point in self._scanCore.scan]

        # TODO document output file format; include cost function values
        dataDump = dict()
        dataDump['pixelSizeInMeters'] = numpy.array([pixelSizeYInMeters, pixelSizeXInMeters])
        dataDump['scanInMeters'] = numpy.column_stack((scanYInMeters, scanXInMeters))
        dataDump['probe'] = self._probeCore.probe.getArray()
        dataDump['object'] = self._objectCore.object.getArray()
        numpy.savez(self._reconstructorSettings.outputFilePath.value, **dataDump)

        return result

    @property
    def detectorImagePresenter(self) -> ImagePresenter:
        return self._detectorImageCore.presenter

    @property
    def scanPresenter(self) -> ScanPresenter:
        return self._scanCore.presenter

    @property
    def probePresenter(self) -> ProbePresenter:
        return self._probeCore.presenter

    @property
    def probeImagePresenter(self) -> ImagePresenter:
        return self._probeImageCore.presenter

    @property
    def objectPresenter(self) -> ObjectPresenter:
        return self._objectCore.presenter

    @property
    def objectImagePresenter(self) -> ImagePresenter:
        return self._objectImageCore.presenter

    @property
    def workflowPresenter(self) -> WorkflowPresenter:
        return self._workflowCore.presenter
