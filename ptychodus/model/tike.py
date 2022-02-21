from __future__ import annotations
from decimal import Decimal
from typing import Any
import logging

import numpy

try:
    import tike
except ImportError:
    tike = None

from .observer import Observable, Observer
from .reconstructor import Reconstructor, NullReconstructor
from .scan import ScanSequence
from .settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class TikeAdaptiveMomentSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.useAdaptiveMoment = settingsGroup.createBooleanEntry('UseAdaptiveMoment', False)
        self.mdecay = settingsGroup.createRealEntry('MDecay', '0.9')
        self.vdecay = settingsGroup.createRealEntry('VDecay', '0.999')
        settingsGroup.addObserver(self)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikeAdaptiveMomentPresenter(Observable, Observer):
    def __init__(self, settings: TikeAdaptiveMomentSettings) -> None:
        super().__init__()
        self._settings = settings
        settings.addObserver(self)

    def isAdaptiveMomentEnabled(self) -> bool:
        return self._settings.useAdaptiveMoment.value

    def setAdaptiveMomentEnabled(self, enabled: bool) -> None:
        self._settings.useAdaptiveMoment.value = enabled

    def getMinMDecay(self) -> Decimal:
        return Decimal(0)

    def getMaxMDecay(self) -> Decimal:
        return Decimal(1)

    def getMDecay(self) -> Decimal:
        return self._clamp(self._settings.mdecay.value, self.getMinMDecay(), self.getMaxMDecay())

    def setMDecay(self, value: Decimal) -> None:
        self._settings.mdecay.value = value

    def getMinVDecay(self) -> Decimal:
        return Decimal(0)

    def getMaxVDecay(self) -> Decimal:
        return Decimal(1)

    def getVDecay(self) -> Decimal:
        return self._clamp(self._settings.vdecay.value, self.getMinVDecay(), self.getMaxVDecay())

    def setVDecay(self, value: Decimal) -> None:
        self._settings.vdecay.value = value

    @staticmethod
    def _clamp(x, xmin, xmax):
        assert xmin <= xmax
        return max(xmin, min(x, xmax))

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class TikeProbeCorrectionSettings(TikeAdaptiveMomentSettings):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useProbeCorrection = settingsGroup.createBooleanEntry('UseProbeCorrection', False)
        self.sparsityConstraint = settingsGroup.createRealEntry('SparsityConstraint', '1')
        self.orthogonalityConstraint = settingsGroup.createBooleanEntry(
            'OrthogonalityConstraint', True)
        self.centeredIntensityConstraint = settingsGroup.createBooleanEntry(
            'CenteredIntensityConstraint', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeProbeCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikeProbeCorrection'))


class TikeProbeCorrectionPresenter(TikeAdaptiveMomentPresenter):
    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(cls, settings: TikeProbeCorrectionSettings) -> TikeProbeCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isProbeCorrectionEnabled(self) -> bool:
        return self._settings.useProbeCorrection.value

    def setProbeCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.useProbeCorrection.value = enabled

    def getMinSparsityConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxSparsityConstraint(self) -> Decimal:
        return Decimal(1)

    def getSparsityConstraint(self) -> Decimal:
        return self._clamp(self._settings.sparsityConstraint.value,
                           self.getMinSparsityConstraint(), self.getMaxSparsityConstraint())

    def setSparsityConstraint(self, value: Decimal) -> None:
        self._settings.sparsityConstraint.value = value

    def isOrthogonalityConstraintEnabled(self) -> bool:
        return self._settings.orthogonalityConstraint.value

    def setOrthogonalityConstraintEnabled(self, enabled: bool) -> None:
        self._settings.orthogonalityConstraint.value = enabled

    def isCenteredIntensityConstraintEnabled(self) -> bool:
        return self._settings.centeredIntensityConstraint.value

    def setCenteredIntensityConstraintEnabled(self, enabled: bool) -> None:
        self._settings.centeredIntensityConstraint.value = enabled


class TikeObjectCorrectionSettings(TikeAdaptiveMomentSettings):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useObjectCorrection = settingsGroup.createBooleanEntry('UseObjectCorrection', False)
        self.positivityConstraint = settingsGroup.createRealEntry('PositivityConstraint', 0)
        self.smoothnessConstraint = settingsGroup.createRealEntry('SmoothnessConstraint', 0)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeObjectCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikeObjectCorrection'))


class TikeObjectCorrectionPresenter(TikeAdaptiveMomentPresenter):
    def __init__(self, settings: TikeObjectCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(cls,
                       settings: TikeObjectCorrectionSettings) -> TikeObjectCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isObjectCorrectionEnabled(self) -> bool:
        return self._settings.useObjectCorrection.value

    def setObjectCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.useObjectCorrection.value = enabled

    def getMinPositivityConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxPositivityConstraint(self) -> Decimal:
        return Decimal(1)

    def getPositivityConstraint(self) -> Decimal:
        return self._clamp(self._settings.positivityConstraint.value,
                           self.getMinPositivityConstraint(), self.getMaxPositivityConstraint())

    def setPositivityConstraint(self, value: Decimal) -> None:
        self._settings.positivityConstraint.value = value

    def getMinSmoothnessConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxSmoothnessConstraint(self) -> Decimal:
        return Decimal('0.125')

    def getSmoothnessConstraint(self) -> Decimal:
        return self._clamp(self._settings.smoothnessConstraint.value,
                           self.getMinSmoothnessConstraint(), self.getMaxSmoothnessConstraint())

    def setSmoothnessConstraint(self, value: Decimal) -> None:
        self._settings.smoothnessConstraint.value = value


class TikePositionCorrectionSettings(TikeAdaptiveMomentSettings):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.usePositionCorrection = settingsGroup.createBooleanEntry(
            'UsePositionCorrection', False)
        self.usePositionRegularization = settingsGroup.createBooleanEntry(
            'UsePositionRegularization', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikePositionCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikePositionCorrection'))


class TikePositionCorrectionPresenter(TikeAdaptiveMomentPresenter):
    def __init__(self, settings: TikePositionCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(
            cls, settings: TikePositionCorrectionSettings) -> TikePositionCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isPositionCorrectionEnabled(self) -> bool:
        return self._settings.usePositionCorrection.value

    def setPositionCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.usePositionCorrection.value = enabled

    def isPositionRegularizationEnabled(self) -> bool:
        return self._settings.usePositionRegularization.value

    def setPositionRegularizationEnabled(self, enabled: bool) -> None:
        self._settings.usePositionRegularization.value = enabled


class TikeSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.useMpi = settingsGroup.createBooleanEntry('UseMpi', False)
        self.numGpus = settingsGroup.createIntegerEntry('NumGpus', 1)
        self.noiseModel = settingsGroup.createStringEntry('NoiseModel', 'gaussian')
        self.numBatch = settingsGroup.createIntegerEntry('NumBatch', 1)
        self.numIter = settingsGroup.createIntegerEntry('NumIter', 1)
        self.cgIter = settingsGroup.createIntegerEntry('CgIter', 2)
        self.alpha = settingsGroup.createRealEntry('Alpha', '0.05')
        self.stepLength = settingsGroup.createRealEntry('StepLength', 1)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeSettings:
        settings = cls(settingsRegistry.createGroup('Tike'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikePresenter(Observable, Observer):
    MAX_INT = 0x7FFFFFFF

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

    def getMinNumGpus(self) -> int:
        return 1

    def getMaxNumGpus(self) -> int:
        return self.MAX_INT

    def getNumGpus(self) -> int:
        return self._clamp(self._settings.numGpus.value, self.getMinNumGpus(),
                           self.getMaxNumGpus())

    def setNumGpus(self, value: int) -> None:
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


class TikeReconstructor:
    def __init__(self, settings: TikeSettings,
                 objectCorrectionSettings: TikeObjectCorrectionSettings,
                 positionCorrectionSettings: TikePositionCorrectionSettings,
                 probeCorrectionSettings: TikeProbeCorrectionSettings,
                 scanSequence: ScanSequence) -> None:
        self._settings = settings
        self._objectCorrectionSettings = objectCorrectionSettings
        self._positionCorrectionSettings = positionCorrectionSettings
        self._probeCorrectionSettings = probeCorrectionSettings
        self._scanSequence = scanSequence

    @property
    def backendName(self) -> str:
        return 'Tike'

    def getScan(self) -> numpy.ndarray:
        x = [point.x for point in iter(self._scanSequence)]
        y = [point.y for point in iter(self._scanSequence)]
        return numpy.column_stack((x, y))

    def getObjectOptions(self) -> tike.ptycho.ObjectOptions:
        settings = self._objectCorrectionSettings
        options = None

        if settings.useObjectCorrection.value:
            options = tike.ptycho.ObjectOptions(
                positivity_constraint=settings.positivityConstraint.value,
                smoothness_constraint=settings.smoothnessConstraint.value,
                use_adaptive_moment=settings.useAdaptiveMoment.value,
                vdecay=settings.vdecay.value,
                mdecay=settings.mdecay.value,
            )

        return options

    def getPositionOptions(self) -> tike.ptycho.PositionOptions:
        settings = self._positionCorrectionSettings
        options = None

        if settings.usePositionCorrection.value:
            options = tike.ptycho.PositionOptions(
                num_positions=len(self._scanSequence),
                initial_scan=self.getScan(),  # TODO CHECKME
                use_adaptive_moment=settings.useAdaptiveMoment.value,
                vdecay=settings.vdecay.value,
                mdecay=settings.mdecay.value,
                use_position_regularization=settings.usePositionRegularization.value,
            )

        return options

    def getProbeOptions(self) -> tike.ptycho.ProbeOptions:
        settings = self._probeCorrectionSettings
        options = None

        if settings.useProbeCorrection.value:
            options = tike.ptycho.ProbeOptions(
                orthogonality_constraint=settings.orthogonalityConstraint.value,
                centered_intensity_constraint=settings.centeredIntensityConstraint.value,
                sparsity_constraint=settings.sparsityConstraint.value,
                use_adaptive_moment=settings.useAdaptiveMoment.value,
                vdecay=settings.vdecay.value,
                mdecay=settings.mdecay.value,
            )

        return options

    def __call__(self, algorithmOptions: tike.ptycho.solvers.IterativeOptions) -> int:
        """
        data : (FRAME, WIDE, HIGH) float32
            The intensity (square of the absolute value) of the propagated
            wavefront; i.e. what the detector records. FFT-shifted so the
            diffraction peak is at the corners.
        """
        data = None  # FIXME
        """
        probe : (1, 1, SHARED, WIDE, HIGH) complex64
            The shared complex illumination function amongst all positions.
        """
        probe = None  # FIXME
        """
        psi : (WIDE, HIGH) complex64
            The wavefront modulation coefficients of the object.
        """
        initialObject = None  # FIXME

        result = tike.ptycho.reconstruct(
            data=data,
            probe=probe,
            scan=self.getScan(),
            algorithm_options=algorithmOptions,
            model=self._settings.noiseModel.value,
            num_gpu=self._settings.numGpus.value,
            object_options=self.getObjectOptions(),
            position_options=self.getPositionOptions(),
            probe_options=self.getProbeOptions(),
            psi=initialObject,
            use_mpi=self._settings.useMpi.value,
        )

        # TODO do stuff with result

        return 0  # TODO return non-zero if problems


class RegularizedPIEReconstructor(Reconstructor):
    def __init__(self, tikeReconstructor: TikeReconstructor) -> None:
        super().__init__()
        self._algorithmOptions = tike.ptycho.solvers.RpieOptions()
        self._tikeReconstructor = tikeReconstructor

    @property
    def name(self) -> str:
        return self._algorithmOptions.name

    @property
    def backendName(self) -> str:
        return self._tikeReconstructor.backendName

    def reconstruct(self) -> int:
        # TODO update self._algorithmOptions
        return self._tikeReconstructor(self._algorithmOptions)


class AdaptiveMomentGradientDescentReconstructor(Reconstructor):
    def __init__(self, tikeReconstructor: TikeReconstructor) -> None:
        super().__init__()
        self._algorithmOptions = tike.ptycho.solvers.AdamOptions()
        self._tikeReconstructor = tikeReconstructor

    @property
    def name(self) -> str:
        return self._algorithmOptions.name

    @property
    def backendName(self) -> str:
        return self._tikeReconstructor.backendName

    def reconstruct(self) -> int:
        # TODO update self._algorithmOptions
        return self._tikeReconstructor(self._algorithmOptions)


class ConjugateGradientReconstructor(Reconstructor):
    def __init__(self, tikeReconstructor: TikeReconstructor) -> None:
        super().__init__()
        self._algorithmOptions = tike.ptycho.solvers.CgradOptions()
        self._tikeReconstructor = tikeReconstructor

    @property
    def name(self) -> str:
        return self._algorithmOptions.name

    @property
    def backendName(self) -> str:
        return self._tikeReconstructor.backendName

    def reconstruct(self) -> int:
        # TODO update self._algorithmOptions
        return self._tikeReconstructor(self._algorithmOptions)


class IterativeLeastSquaresReconstructor(Reconstructor):
    def __init__(self, tikeReconstructor: TikeReconstructor) -> None:
        super().__init__()
        self._algorithmOptions = tike.ptycho.solvers.LstsqOptions()
        self._tikeReconstructor = tikeReconstructor

    @property
    def name(self) -> str:
        return self._algorithmOptions.name

    @property
    def backendName(self) -> str:
        return self._tikeReconstructor.backendName

    def reconstruct(self) -> int:
        # TODO update self._algorithmOptions
        return self._tikeReconstructor(self._algorithmOptions)


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
                       scanSequence: ScanSequence,
                       isDeveloperModeEnabled: bool = False) -> TikeBackend:
        core = cls(settingsRegistry)

        if tike:
            logger.info(f'{tike.__name__} ({tike.__version__})')

            tikeReconstructor = TikeReconstructor(core._settings, core._objectCorrectionSettings,
                                                  core._positionCorrectionSettings,
                                                  core._probeCorrectionSettings, scanSequence)
            core.reconstructorList.append(RegularizedPIEReconstructor(tikeReconstructor))
            core.reconstructorList.append(
                AdaptiveMomentGradientDescentReconstructor(tikeReconstructor))
            core.reconstructorList.append(ConjugateGradientReconstructor(tikeReconstructor))
            core.reconstructorList.append(IterativeLeastSquaresReconstructor(tikeReconstructor))
        else:
            logger.info('tike not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('rpie', 'Tike'))
                core.reconstructorList.append(NullReconstructor('adam_grad', 'Tike'))
                core.reconstructorList.append(NullReconstructor('cgrad', 'Tike'))
                core.reconstructorList.append(NullReconstructor('lstsq_grad', 'Tike'))

        return core
