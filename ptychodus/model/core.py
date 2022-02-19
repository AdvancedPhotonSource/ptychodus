import logging

import numpy

from .data_file import *
from .detector import *
from .h5tree import *
from .metadata import *
from .object import *
from .observer import *
from .probe import *
from .ptychopy import PtychoPyBackend
from .ptychonn import PtychoNNBackend
from .reconstructor import *
from .scan import *
from .settings import *
from .tike import TikeBackend
from .velociprobe import *

logger = logging.getLogger(__name__)


class ModelCore:
    def __init__(self, isDeveloperModeEnabled: bool) -> None:
        self.rng = numpy.random.default_rng()

        self.settingsRegistry = SettingsRegistry()
        self._detectorSettings = DetectorSettings.createInstance(self.settingsRegistry)
        self._cropSettings = CropSettings.createInstance(self.settingsRegistry)
        self._scanSettings = ScanSettings.createInstance(self.settingsRegistry)
        self._probeSettings = ProbeSettings.createInstance(self.settingsRegistry)
        self._objectSettings = ObjectSettings.createInstance(self.settingsRegistry)
        self._reconstructorSettings = ReconstructorSettings.createInstance(self.settingsRegistry)

        self._selectableScanSequence = SelectableScanSequence.createInstance(self._scanSettings)
        self._transformedScanSequence = TransformedScanSequence.createInstance(
            self._scanSettings, self._selectableScanSequence)
        self._detector = Detector.createInstance(self._detectorSettings, self._cropSettings)
        self._probe = Probe.createInstance(self._probeSettings)
        self._objectSizer = ObjectSizer.createInstance(self._transformedScanSequence,
                                                       self._detector, self._probe)

        self.h5FileTreeReader = H5FileTreeReader()
        self._velociprobeReader = VelociprobeReader()
        self._velociprobeImageSequence = VelociprobeImageSequence.createInstance(
            self._velociprobeReader)
        self._croppedImageSequence = CroppedImageSequence.createInstance(
            self._cropSettings, self._velociprobeImageSequence)
        self._dataDirectoryWatcher = DataDirectoryWatcher()

        self.ptychopyBackend = PtychoPyBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self.tikeBackend = TikeBackend.createInstance(self.settingsRegistry,
                                                      isDeveloperModeEnabled)
        self.ptychonnBackend = PtychoNNBackend.createInstance(self.settingsRegistry,
                                                              isDeveloperModeEnabled)
        self._selectableReconstructor = SelectableReconstructor.createInstance(
            self._reconstructorSettings, self.ptychopyBackend.reconstructorList +
            self.tikeBackend.reconstructorList + self.ptychonnBackend.reconstructorList)

        self.dataFilePresenter = DataFilePresenter()
        self.settingsPresenter = SettingsPresenter.createInstance(self.settingsRegistry)
        self.detectorParametersPresenter = DetectorParametersPresenter.createInstance(
            self._detectorSettings, self._cropSettings)
        self.detectorDatasetPresenter = DetectorDatasetPresenter.createInstance(
            self._velociprobeReader)
        self.detectorImagePresenter = DetectorImagePresenter.createInstance(
            self._croppedImageSequence)
        self.importSettingsPresenter = ImportSettingsPresenter.createInstance(
            self._velociprobeReader, self._detectorSettings, self._cropSettings,
            self._probeSettings)
        self.probePresenter = ProbePresenter.createInstance(self._detectorSettings,
                                                            self._probeSettings, self._probe)
        self.scanPresenter = ScanPresenter.createInstance(self._scanSettings,
                                                          self._selectableScanSequence,
                                                          self._transformedScanSequence)
        self.objectPresenter = ObjectPresenter.createInstance(self.rng, self._objectSettings,
                                                              self._objectSizer)
        self.reconstructorPresenter = ReconstructorPresenter.createInstance(
            self._reconstructorSettings, self._selectableReconstructor)

    @classmethod
    def createInstance(cls, isDeveloperModeEnabled: bool = False):
        model = cls(isDeveloperModeEnabled)
        model.dataFilePresenter.addReader(model.h5FileTreeReader)
        model.dataFilePresenter.addReader(model._velociprobeReader)
        return model

    def start(self) -> None:
        self._dataDirectoryWatcher.start()

    def stop(self) -> None:
        self._dataDirectoryWatcher.stop()
        self._dataDirectoryWatcher.join()
