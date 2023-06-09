from importlib.metadata import version
from typing import Any, Union
import logging
import pprint

import numpy
import numpy.typing

import tike.ptycho

from ...api.reconstructor import ReconstructResult, Reconstructor
from .arrayConverter import TikeArrays, TikeArrayConverter
from .multigrid import TikeMultigridSettings
from .objectCorrection import TikeObjectCorrectionSettings
from .positionCorrection import TikePositionCorrectionSettings
from .probeCorrection import TikeProbeCorrectionSettings
from .settings import TikeSettings

logger = logging.getLogger(__name__)


class TikeReconstructor:

    def __init__(self, settings: TikeSettings, multigridSettings: TikeMultigridSettings,
                 positionCorrectionSettings: TikePositionCorrectionSettings,
                 probeCorrectionSettings: TikeProbeCorrectionSettings,
                 objectCorrectionSettings: TikeObjectCorrectionSettings,
                 arrayConverter: TikeArrayConverter) -> None:
        self._settings = settings
        self._multigridSettings = multigridSettings
        self._positionCorrectionSettings = positionCorrectionSettings
        self._probeCorrectionSettings = probeCorrectionSettings
        self._objectCorrectionSettings = objectCorrectionSettings
        self._arrayConverter = arrayConverter

        tikeVersion = version('tike')
        logger.info(f'\tTike {tikeVersion}')

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
                clip_magnitude=settings.useMagnitudeClipping.value,
            )

        return options

    def getPositionOptions(self,
                           initialScan: numpy.typing.NDArray[Any]) -> tike.ptycho.PositionOptions:
        settings = self._positionCorrectionSettings
        options = None

        if settings.usePositionCorrection.value:
            options = tike.ptycho.PositionOptions(
                initial_scan=initialScan,
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
                force_orthogonality=settings.forceOrthogonality.value,
                force_centered_intensity=settings.forceCenteredIntensity.value,
                force_sparsity=float(settings.forceSparsity.value),
                use_adaptive_moment=settings.useAdaptiveMoment.value,
                vdecay=float(settings.vdecay.value),
                mdecay=float(settings.mdecay.value),
                probe_support=probeSupport,
                probe_support_radius=float(settings.probeSupportRadius.value),
                probe_support_degree=float(settings.probeSupportDegree.value),
                additional_probe_penalty=float(settings.additionalProbePenalty.value),
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

    def __call__(self,
                 algorithmOptions: tike.ptycho.solvers.IterativeOptions) -> ReconstructResult:
        inputArrays = self._arrayConverter.exportToTike()

        data = self._arrayConverter.getDiffractionData()
        scan = inputArrays.scan
        probe = inputArrays.probe
        psi = inputArrays.object_
        numGpus = self.getNumGpus()

        if len(data) != len(scan):
            numFrame = min(len(data), len(scan))
            scan = scan[:numFrame, ...]
            data = data[:numFrame, ...]

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
            position_options=self.getPositionOptions(scan))

        if self._multigridSettings.useMultigrid.value:
            result = tike.ptycho.reconstruct_multigrid(
                data=data,
                parameters=parameters,
                model=self._settings.noiseModel.value,
                num_gpu=numGpus,
                use_mpi=False,
                num_levels=self._multigridSettings.numLevels.value,
                interp=None,  # TODO does this have other options?
            )
        else:
            # TODO support interactive reconstructions
            with tike.ptycho.Reconstruction(
                    data=data,
                    parameters=parameters,
                    model=self._settings.noiseModel.value,
                    num_gpu=numGpus,
                    use_mpi=False,
            ) as context:
                context.iterate(parameters.algorithm_options.num_iter)
            result = context.parameters

        logger.debug(f'Result: {pprint.pformat(result)}')

        outputArrays = TikeArrays(
            indexes=inputArrays.indexes,
            scan=result.scan,
            probe=result.probe,
            object_=result.psi,
        )
        self._arrayConverter.importFromTike(outputArrays)

        return ReconstructResult(0, result.algorithm_options.costs)


class RegularizedPIEReconstructor(Reconstructor):

    def __init__(self, tikeReconstructor: TikeReconstructor) -> None:
        super().__init__()
        self._algorithmOptions = tike.ptycho.solvers.RpieOptions()
        self._tikeReconstructor = tikeReconstructor

    @property
    def name(self) -> str:
        return self._algorithmOptions.name

    @property
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> ReconstructResult:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.batch_method = self._settings.batchMethod.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.value
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
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> ReconstructResult:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.batch_method = self._settings.batchMethod.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.value
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
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> ReconstructResult:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.batch_method = self._settings.batchMethod.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.value
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
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> ReconstructResult:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.batch_method = self._settings.batchMethod.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.value
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
    def _settings(self) -> TikeSettings:
        return self._tikeReconstructor._settings

    def reconstruct(self) -> ReconstructResult:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.batch_method = self._settings.batchMethod.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.value
        return self._tikeReconstructor(self._algorithmOptions)
