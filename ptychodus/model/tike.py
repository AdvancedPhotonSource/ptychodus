from __future__ import annotations
from decimal import Decimal
from typing import Any, Union
import logging

import numpy

try:
    import tike.ptycho
except ModuleNotFoundError:

    class tike:
        ptycho = None


from ..api.data import DataArrayType, DataFile, DiffractionDataset
from ..api.observer import Observable, Observer
from ..api.settings import SettingsRegistry, SettingsGroup
from .image import ImageExtent
from .object import Object, ObjectSizer
from .probe import Probe, ProbeSizer
from .reconstructor import Reconstructor, NullReconstructor, ReconstructorPlotPresenter
from .scan import Scan

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
        self.useProbeCorrection = settingsGroup.createBooleanEntry('UseProbeCorrection', True)
        self.orthogonalityConstraint = settingsGroup.createBooleanEntry(
            'OrthogonalityConstraint', True)
        self.centeredIntensityConstraint = settingsGroup.createBooleanEntry(
            'CenteredIntensityConstraint', False)
        self.sparsityConstraint = settingsGroup.createRealEntry('SparsityConstraint', '1')
        self.useFiniteProbeSupport = settingsGroup.createBooleanEntry(
            'UseFiniteProbeSupport', True)
        self.probeSupportWeight = settingsGroup.createRealEntry('ProbeSupportWeight', '10')
        self.probeSupportRadius = settingsGroup.createRealEntry('ProbeSupportRadius', '0.3')
        self.probeSupportDegree = settingsGroup.createRealEntry('ProbeSupportDegree', '5')

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

    def isOrthogonalityConstraintEnabled(self) -> bool:
        return self._settings.orthogonalityConstraint.value

    def setOrthogonalityConstraintEnabled(self, enabled: bool) -> None:
        self._settings.orthogonalityConstraint.value = enabled

    def isCenteredIntensityConstraintEnabled(self) -> bool:
        return self._settings.centeredIntensityConstraint.value

    def setCenteredIntensityConstraintEnabled(self, enabled: bool) -> None:
        self._settings.centeredIntensityConstraint.value = enabled

    def getMinSparsityConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxSparsityConstraint(self) -> Decimal:
        return Decimal(1)

    def getSparsityConstraint(self) -> Decimal:
        return self._clamp(self._settings.sparsityConstraint.value,
                           self.getMinSparsityConstraint(), self.getMaxSparsityConstraint())

    def setSparsityConstraint(self, value: Decimal) -> None:
        self._settings.sparsityConstraint.value = value

    def isFiniteProbeSupportEnabled(self) -> bool:
        return self._settings.useFiniteProbeSupport.value

    def setFiniteProbeSupportEnabled(self, enabled: bool) -> None:
        self._settings.useFiniteProbeSupport.value = enabled

    def getMinProbeSupportWeight(self) -> Decimal:
        return Decimal()

    def getProbeSupportWeight(self) -> Decimal:
        weight = self._settings.probeSupportWeight.value
        weightMin = self.getMinProbeSupportWeight()
        return weight if weight >= weightMin else weightMin

    def setProbeSupportWeight(self, value: Decimal) -> None:
        self._settings.probeSupportWeight.value = value

    def getMinProbeSupportRadius(self) -> Decimal:
        return Decimal()

    def getMaxProbeSupportRadius(self) -> Decimal:
        return Decimal('0.5')

    def getProbeSupportRadius(self) -> Decimal:
        return self._clamp(self._settings.probeSupportRadius.value,
                           self.getMinProbeSupportRadius(), self.getMaxProbeSupportRadius())

    def setProbeSupportRadius(self, value: Decimal) -> None:
        self._settings.probeSupportRadius.value = value

    def getMinProbeSupportDegree(self) -> Decimal:
        return Decimal()

    def getProbeSupportDegree(self) -> Decimal:
        return self._settings.probeSupportDegree.value

    def setProbeSupportDegree(self, value: Decimal) -> None:
        self._settings.probeSupportDegree.value = value


class TikeObjectCorrectionSettings(TikeAdaptiveMomentSettings):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useObjectCorrection = settingsGroup.createBooleanEntry('UseObjectCorrection', True)
        self.positivityConstraint = settingsGroup.createRealEntry('PositivityConstraint', '0')
        self.smoothnessConstraint = settingsGroup.createRealEntry('SmoothnessConstraint', '0')

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
        self.numGpus = settingsGroup.createStringEntry('NumGpus', '1')
        self.noiseModel = settingsGroup.createStringEntry('NoiseModel', 'gaussian')
        self.numProbeModes = settingsGroup.createIntegerEntry('NumProbeModes', 1)
        self.numBatch = settingsGroup.createIntegerEntry('NumBatch', 10)
        self.numIter = settingsGroup.createIntegerEntry('NumIter', 1)
        self.cgIter = settingsGroup.createIntegerEntry('CgIter', 2)
        self.alpha = settingsGroup.createRealEntry('Alpha', '0.05')
        self.stepLength = settingsGroup.createRealEntry('StepLength', '1')

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

    def getMinNumProbeModes(self) -> int:
        return 1

    def getMaxNumProbeModes(self) -> int:
        return self.MAX_INT

    def getNumProbeModes(self) -> int:
        return self._clamp(self._settings.numProbeModes.value, self.getMinNumProbeModes(),
                           self.getMaxNumProbeModes())

    def setNumProbeModes(self, value: int) -> None:
        self._settings.numProbeModes.value = value

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
                 probeCorrectionSettings: TikeProbeCorrectionSettings, dataFile: DataFile,
                 scan: Scan, probeSizer: ProbeSizer, probe: Probe, objectSizer: ObjectSizer,
                 object_: Object, reconstructorPlotPresenter: ReconstructorPlotPresenter) -> None:
        self._settings = settings
        self._objectCorrectionSettings = objectCorrectionSettings
        self._positionCorrectionSettings = positionCorrectionSettings
        self._probeCorrectionSettings = probeCorrectionSettings
        self._dataFile = dataFile
        self._scan = scan
        self._probeSizer = probeSizer
        self._probe = probe
        self._objectSizer = objectSizer
        self._object = object_
        self._reconstructorPlotPresenter = reconstructorPlotPresenter

    @property
    def backendName(self) -> str:
        return 'Tike'

    def getDiffractionData(self) -> DataArrayType:
        data = self._dataFile.getDiffractionData()
        return numpy.fft.ifftshift(data, axes=(-2, -1)).astype('float32')

    def getProbe(self) -> numpy.ndarray:
        probe = self._probe.getArray()
        probe = probe[numpy.newaxis, numpy.newaxis, :, :].astype('complex64')

        if self._settings.numProbeModes.value > 0:
            probe = tike.ptycho.probe.add_modes_random_phase(probe,
                                                             self._settings.numProbeModes.value)

        return probe

    def getInitialObject(self) -> numpy.ndarray:
        delta = self._probeSizer.getProbeExtent()
        before = delta // 2
        after = delta - before

        widthPad = (before.width, after.width)
        heightPad = (before.height, after.height)

        return numpy.pad(self._object.getArray(), (heightPad, widthPad)).astype('complex64')

    def getScan(self) -> numpy.ndarray:
        xvalues = list()
        yvalues = list()

        px_m = self._objectSizer.getPixelSizeXInMeters()
        py_m = self._objectSizer.getPixelSizeYInMeters()

        logger.debug(f'object pixel size x = {px_m} m')
        logger.debug(f'object pixel size y = {py_m} m')

        for point in self._scan:
            xvalues.append(point.x / px_m)
            yvalues.append(point.y / py_m)

        pad = self._probeSizer.getProbeExtent() // 2
        ux = pad.width - min(xvalues)
        uy = pad.height - min(yvalues)

        xvalues = [x + ux for x in xvalues]
        yvalues = [y + uy for y in yvalues]

        return numpy.column_stack((yvalues, xvalues)).astype('float32')

    def getObjectOptions(self) -> tike.ptycho.ObjectOptions:
        settings = self._objectCorrectionSettings
        options = None

        if settings.useObjectCorrection.value:
            options = tike.ptycho.ObjectOptions(
                positivity_constraint=float(settings.positivityConstraint.value),
                smoothness_constraint=float(settings.smoothnessConstraint.value),
                use_adaptive_moment=settings.useAdaptiveMoment.value,
                vdecay=float(settings.vdecay.value),
                mdecay=float(settings.mdecay.value),
            )

        return options

    def getPositionOptions(self) -> tike.ptycho.PositionOptions:
        settings = self._positionCorrectionSettings
        options = None

        if settings.usePositionCorrection.value:
            options = tike.ptycho.PositionOptions(
                initial_scan=self.getScan(),
                use_adaptive_moment=settings.useAdaptiveMoment.value,
                vdecay=float(settings.vdecay.value),
                mdecay=float(settings.mdecay.value),
                use_position_regularization=settings.usePositionRegularization.value,
            )

        return options

    def getProbeOptions(self) -> tike.ptycho.ProbeOptions:
        settings = self._probeCorrectionSettings
        options = None

        if settings.useProbeCorrection.value:
            probeSupport = float(settings.probeSupportWeight.value) \
                    if settings.useFiniteProbeSupport.value else 0.

            options = tike.ptycho.ProbeOptions(
                orthogonality_constraint=settings.orthogonalityConstraint.value,
                centered_intensity_constraint=settings.centeredIntensityConstraint.value,
                sparsity_constraint=float(settings.sparsityConstraint.value),
                use_adaptive_moment=settings.useAdaptiveMoment.value,
                vdecay=float(settings.vdecay.value),
                mdecay=float(settings.mdecay.value),
                probe_support=probeSupport,
                probe_support_radius=float(settings.probeSupportRadius.value),
                probe_support_degree=float(settings.probeSupportDegree.value),
            )

        return options

    def getNumGpus(self) -> Union[int, tuple[int, ...]]:
        numGpus = self._settings.numGpus.value
        onlyDigitsAndCommas = all(c.isdigit() or c == ',' for c in numGpus)
        hasDigit = any(c.isdigit() for c in numGpus)

        if onlyDigitsAndCommas and hasDigit:
            if ',' in numGpus:
                return tuple(int(n) for n in numGpus.split(',') if n)
            else:
                return int(numGpus)

        return 1

    def __call__(self, algorithmOptions: tike.ptycho.solvers.IterativeOptions) -> int:
        data = self.getDiffractionData()
        scan = self.getScan()
        probe = self.getProbe()
        psi = self.getInitialObject()
        numGpus = self.getNumGpus()

        if len(data) != len(scan):
            numFrame = min(len(data), len(scan))
            scan = scan[:numFrame, ...]
            data = data[:numFrame, ...]

        # FIXME figure out how to remove the next line (get_padded_object)
        psi, scan = tike.ptycho.object.get_padded_object(scan, probe)

        logger.debug(f'data shape={data.shape}')
        logger.debug(f'scan shape={scan.shape}')
        logger.debug(f'probe shape={probe.shape}')
        logger.debug(f'object shape={psi.shape}')
        logger.debug(f'num_gpu={numGpus}')

        parameters = tike.ptycho.solvers.PtychoParameters(
            probe=probe,
            psi=psi,
            scan=scan,
            algorithm_options=algorithmOptions,
            probe_options=self.getProbeOptions(),
            object_options=self.getObjectOptions(),
            position_options=self.getPositionOptions())

        result = tike.ptycho.reconstruct(data=data,
                                         parameters=parameters,
                                         model=self._settings.noiseModel.value,
                                         num_gpu=numGpus,
                                         use_mpi=self._settings.useMpi.value)

        # TODO self._scan.setScanPoints(...)
        self._probe.setArray(result.probe[0, 0])
        self._object.setArray(result.psi)
        self._reconstructorPlotPresenter.setEnumeratedYValues(result.algorithm_options.costs)

        logger.debug(result)

        return 0


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

    @property
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> int:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.alpha = float(self._settings.alpha.value)
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

    @property
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> int:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.alpha = float(self._settings.alpha.value)
        self._algorithmOptions.step_length = float(self._settings.stepLength.value)
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

    @property
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> int:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.cg_iter = self._settings.cgIter.value
        self._algorithmOptions.step_length = float(self._settings.stepLength.value)
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

    @property
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> int:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        return self._tikeReconstructor(self._algorithmOptions)


class DifferenceMapReconstructor(Reconstructor):

    def __init__(self, tikeReconstructor: TikeReconstructor) -> None:
        super().__init__()
        self._algorithmOptions = tike.ptycho.solvers.DmOptions()
        self._tikeReconstructor = tikeReconstructor

    @property
    def name(self) -> str:
        return self._algorithmOptions.name

    @property
    def backendName(self) -> str:
        return self._tikeReconstructor.backendName

    @property
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> int:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
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
                       dataFile: DataFile,
                       scan: Scan,
                       probeSizer: ProbeSizer,
                       probe: Probe,
                       objectSizer: ObjectSizer,
                       object_: Object,
                       reconstructorPlotPresenter: ReconstructorPlotPresenter,
                       isDeveloperModeEnabled: bool = False) -> TikeBackend:
        core = cls(settingsRegistry)

        if tike.ptycho:
            logger.info(f'{tike.__name__} ({tike.__version__})')

            tikeReconstructor = TikeReconstructor(core._settings, core._objectCorrectionSettings,
                                                  core._positionCorrectionSettings,
                                                  core._probeCorrectionSettings, dataFile, scan,
                                                  probeSizer, probe, objectSizer, object_,
                                                  reconstructorPlotPresenter)
            core.reconstructorList.append(RegularizedPIEReconstructor(tikeReconstructor))
            core.reconstructorList.append(
                AdaptiveMomentGradientDescentReconstructor(tikeReconstructor))
            core.reconstructorList.append(ConjugateGradientReconstructor(tikeReconstructor))
            core.reconstructorList.append(IterativeLeastSquaresReconstructor(tikeReconstructor))
            core.reconstructorList.append(DifferenceMapReconstructor(tikeReconstructor))
        else:
            logger.info('tike not found.')

            if isDeveloperModeEnabled:
                core.reconstructorList.append(NullReconstructor('rpie', 'Tike'))
                core.reconstructorList.append(NullReconstructor('adam_grad', 'Tike'))
                core.reconstructorList.append(NullReconstructor('cgrad', 'Tike'))
                core.reconstructorList.append(NullReconstructor('lstsq_grad', 'Tike'))
                core.reconstructorList.append(NullReconstructor('dm', 'Tike'))

        return core
