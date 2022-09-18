from __future__ import annotations
from decimal import Decimal
from typing import Final
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from ..data import DiffractionDataAssembler
from ..object import Object
from ..probe import Apparatus, Probe, ProbeSizer
from ..reconstructor import Reconstructor, NullReconstructor, ReconstructorPlotPresenter
from ..scan import Scan, ScanInitializerFactory, ScanRepository
from .arrayConverter import TikeArrayConverter
from .objectCorrection import TikeObjectCorrectionPresenter, TikeObjectCorrectionSettings
from .positionCorrection import TikePositionCorrectionPresenter, TikePositionCorrectionSettings
from .probeCorrection import TikeProbeCorrectionPresenter, TikeProbeCorrectionSettings
from .settings import TikeSettings

logger = logging.getLogger(__name__)


class TikePresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: TikeSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: TikeSettings) -> TikePresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def isMpiEnabled(self) -> bool:
        return self._settings.useMpi.value

    def setMpiEnabled(self, enabled: bool) -> None:
        self._settings.useMpi.value = enabled

    def getNumGpus(self) -> str:
        return self._settings.numGpus.value

    def setNumGpus(self, value: str) -> None:
        self._settings.numGpus.value = value

    def getNoiseModelList(self) -> list[str]:
        return ['poisson', 'gaussian']

    def getNoiseModel(self) -> str:
        return self._settings.noiseModel.value

    def setNoiseModel(self, name: str) -> None:
        self._settings.noiseModel.value = name

    def getMinNumBatch(self) -> int:
        return 1

    def getMaxNumBatch(self) -> int:
        return self.MAX_INT

    def getNumBatch(self) -> int:
        return self._clamp(self._settings.numBatch.value, self.getMinNumBatch(),
                           self.getMaxNumBatch())

    def setNumBatch(self, value: int) -> None:
        self._settings.numBatch.value = value

    def getMinNumIter(self) -> int:
        return 1

    def getMaxNumIter(self) -> int:
        return self.MAX_INT

    def getNumIter(self) -> int:
        return self._clamp(self._settings.numIter.value, self.getMinNumIter(),
                           self.getMaxNumIter())

    def setNumIter(self, value: int) -> None:
        self._settings.numIter.value = value

    def getMinCgIter(self) -> int:
        return 1

    def getMaxCgIter(self) -> int:
        return 64

    def getCgIter(self) -> int:
        return self._clamp(self._settings.cgIter.value, self.getMinCgIter(), self.getMaxCgIter())

    def setCgIter(self, value: int) -> None:
        self._settings.cgIter.value = value

    def getMinAlpha(self) -> Decimal:
        return Decimal(0)

    def getMaxAlpha(self) -> Decimal:
        return Decimal(1)

    def getAlpha(self) -> Decimal:
        return self._clamp(self._settings.alpha.value, self.getMinAlpha(), self.getMaxAlpha())

    def setAlpha(self, value: Decimal) -> None:
        self._settings.alpha.value = value

    def getMinStepLength(self) -> Decimal:
        return Decimal(0)

    def getMaxStepLength(self) -> Decimal:
        return Decimal(1)

    def getStepLength(self) -> Decimal:
        return self._clamp(self._settings.stepLength.value, self.getMinStepLength(),
                           self.getMaxStepLength())

    def setStepLength(self, value: Decimal) -> None:
        self._settings.stepLength.value = value

    @staticmethod
    def _clamp(x, xmin, xmax):
        assert xmin <= xmax
        return max(xmin, min(x, xmax))

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class TikeBackend:

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self._settings = TikeSettings.createInstance(settingsRegistry)
        self._positionCorrectionSettings = TikePositionCorrectionSettings.createInstance(
            settingsRegistry)
        self._probeCorrectionSettings = TikeProbeCorrectionSettings.createInstance(
            settingsRegistry)
        self._objectCorrectionSettings = TikeObjectCorrectionSettings.createInstance(
            settingsRegistry)

        self.presenter = TikePresenter.createInstance(self._settings)
        self.positionCorrectionPresenter = TikePositionCorrectionPresenter.createInstance(
            self._positionCorrectionSettings)
        self.probeCorrectionPresenter = TikeProbeCorrectionPresenter.createInstance(
            self._probeCorrectionSettings)
        self.objectCorrectionPresenter = TikeObjectCorrectionPresenter.createInstance(
            self._objectCorrectionSettings)

        self.reconstructorList: list[Reconstructor] = list()

    @classmethod
    def createInstance(cls,
                       settingsRegistry: SettingsRegistry,
                       diffractionDataAssembler: DiffractionDataAssembler,
                       scan: Scan,
                       probe: Probe,
                       apparatus: Apparatus,
                       object_: Object,
                       scanInitializerFactory: ScanInitializerFactory,
                       scanRepository: ScanRepository,
                       reconstructorPlotPresenter: ReconstructorPlotPresenter,
                       isDeveloperModeEnabled: bool = False) -> TikeBackend:
        core = cls(settingsRegistry)

        try:
            from .reconstructor import AdaptiveMomentGradientDescentReconstructor
            from .reconstructor import ConjugateGradientReconstructor
            from .reconstructor import DifferenceMapReconstructor
            from .reconstructor import IterativeLeastSquaresReconstructor
            from .reconstructor import RegularizedPIEReconstructor
            from .reconstructor import TikeReconstructor
        except ModuleNotFoundError:
            logger.info('tike not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('rpie', 'Tike'))
                core.reconstructorList.append(NullReconstructor('adam_grad', 'Tike'))
                core.reconstructorList.append(NullReconstructor('cgrad', 'Tike'))
                core.reconstructorList.append(NullReconstructor('lstsq_grad', 'Tike'))
                core.reconstructorList.append(NullReconstructor('dm', 'Tike'))
        else:
            arrayConverter = TikeArrayConverter(apparatus, scan, probe, object_,
                                                diffractionDataAssembler, scanInitializerFactory,
                                                scanRepository)
            tikeReconstructor = TikeReconstructor(core._settings, core._objectCorrectionSettings,
                                                  core._positionCorrectionSettings,
                                                  core._probeCorrectionSettings, arrayConverter,
                                                  reconstructorPlotPresenter)
            core.reconstructorList.append(RegularizedPIEReconstructor(tikeReconstructor))
            core.reconstructorList.append(
                AdaptiveMomentGradientDescentReconstructor(tikeReconstructor))
            core.reconstructorList.append(ConjugateGradientReconstructor(tikeReconstructor))
            core.reconstructorList.append(IterativeLeastSquaresReconstructor(tikeReconstructor))
            core.reconstructorList.append(DifferenceMapReconstructor(tikeReconstructor))

        return core
