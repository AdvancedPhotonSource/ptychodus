from typing import Union
import logging

import numpy
import tike.ptycho

from ...api.data import DataArrayType, DataFile
from ..object import Object
from ..probe import Apparatus, Probe, ProbeSizer
from ..reconstructor import Reconstructor, ReconstructorPlotPresenter
from ..scan import Scan
from .objectCorrection import TikeObjectCorrectionSettings
from .positionCorrection import TikePositionCorrectionSettings
from .probeCorrection import TikeProbeCorrectionSettings
from .settings import TikeSettings

logger = logging.getLogger(__name__)
logger.info(f'{tike.__name__} ({tike.__version__})')


class TikeReconstructor:

    def __init__(self, settings: TikeSettings,
                 objectCorrectionSettings: TikeObjectCorrectionSettings,
                 positionCorrectionSettings: TikePositionCorrectionSettings,
                 probeCorrectionSettings: TikeProbeCorrectionSettings, dataFile: DataFile,
                 scan: Scan, probeSizer: ProbeSizer, probe: Probe, apparatus: Apparatus,
                 object_: Object, reconstructorPlotPresenter: ReconstructorPlotPresenter) -> None:
        self._settings = settings
        self._objectCorrectionSettings = objectCorrectionSettings
        self._positionCorrectionSettings = positionCorrectionSettings
        self._probeCorrectionSettings = probeCorrectionSettings
        self._dataFile = dataFile
        self._scan = scan
        self._probeSizer = probeSizer
        self._probe = probe
        self._apparatus = apparatus
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

        px_m = self._apparatus.getObjectPlanePixelSizeXInMeters()
        py_m = self._apparatus.getObjectPlanePixelSizeYInMeters()

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

        # FIXME self._scan.setScanPoints(...)
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
