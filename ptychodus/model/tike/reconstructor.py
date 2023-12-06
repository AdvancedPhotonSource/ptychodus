from collections.abc import Sequence
from importlib.metadata import version
from typing import Any, Union
import logging
import pprint

import numpy
import numpy.typing

import tike.ptycho

from ...api.object import ObjectArrayType
from ...api.object import ObjectPoint
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.probe import ProbeArrayType
from ...api.reconstructor import Reconstructor, ReconstructInput, ReconstructOutput
from ...api.scan import Scan, ScanPoint, TabularScan
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
                 objectCorrectionSettings: TikeObjectCorrectionSettings) -> None:
        self._settings = settings
        self._multigridSettings = multigridSettings
        self._positionCorrectionSettings = positionCorrectionSettings
        self._probeCorrectionSettings = probeCorrectionSettings
        self._objectCorrectionSettings = objectCorrectionSettings

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

    def _plotCosts(self, costs: Sequence[Sequence[float]]) -> Plot2D:
        plot = Plot2D.createNull()
        numIterations = len(costs)

        if numIterations > 0:
            seriesX = PlotSeries(label='Iteration', values=[*range(numIterations)])
            minCost: list[float] = list()
            midCost: list[float] = list()
            maxCost: list[float] = list()

            seriesYList: list[PlotSeries] = list()

            for values in costs:
                minCost.append(min(values))
                midCost.append(float(numpy.median(values)))
                maxCost.append(max(values))

            seriesYList = [
                PlotSeries(label='Minimum', values=minCost),
                PlotSeries(label='Median', values=midCost),
                PlotSeries(label='Maximum', values=maxCost),
            ]

            plot = Plot2D(
                axisX=PlotAxis(label='Iteration', series=[seriesX]),
                axisY=PlotAxis(label='Cost', series=seriesYList),
            )

        return plot

    def __call__(self, parameters: ReconstructInput,
                 algorithmOptions: tike.ptycho.solvers.IterativeOptions) -> ReconstructOutput:
        objectGrid = parameters.objectInterpolator.getGrid()
        psi = parameters.objectInterpolator.getArray().astype('complex64')
        probe = parameters.probeArray[numpy.newaxis, numpy.newaxis, ...].astype('complex64')
        data = numpy.fft.ifftshift(parameters.diffractionPatternArray, axes=(-2, -1))
        coordinateList: list[float] = list()

        # Tike coordinate system origin is top-left corner; requires padding
        ux = -probe.shape[-1] / 2
        uy = -probe.shape[-2] / 2

        for scanPoint in parameters.scan.values():
            objectPoint = objectGrid.mapScanPointToObjectPoint(scanPoint)
            coordinateList.append(objectPoint.y + uy)
            coordinateList.append(objectPoint.x + ux)

        scan = numpy.array(coordinateList, dtype=numpy.float32).reshape(len(parameters.scan), 2)
        scanMin = scan.min(axis=0)
        scanMax = scan.max(axis=0)
        logger.debug(f'Scan range [px]: {scanMin} -> {scanMax}')
        numGpus = self.getNumGpus()

        logger.debug(f'data shape={data.shape}')
        logger.debug(f'scan shape={scan.shape}')
        logger.debug(f'probe shape={probe.shape}')
        logger.debug(f'object shape={psi.shape}')
        logger.debug(f'num_gpu={numGpus}')

        exitwave_options = tike.ptycho.ExitWaveOptions(
            # FIXME: Use a user supplied `measured_pixels` instead
            measured_pixels=numpy.ones(probe.shape[-2:], dtype=numpy.bool_),
            noise_model=self._settings.noiseModel.value,
        )

        ptychoParameters = tike.ptycho.solvers.PtychoParameters(
            probe=probe,
            psi=psi,
            scan=scan,
            algorithm_options=algorithmOptions,
            probe_options=self.getProbeOptions(),
            object_options=self.getObjectOptions(),
            position_options=self.getPositionOptions(scan),
            exitwave_options=exitwave_options,
        )

        if self._multigridSettings.useMultigrid.value:
            result = tike.ptycho.reconstruct_multigrid(
                data=data,
                parameters=ptychoParameters,
                num_gpu=numGpus,
                use_mpi=False,
                num_levels=self._multigridSettings.numLevels.value,
                interp=None,  # TODO does this have other options?
            )
        else:
            # TODO support interactive reconstructions
            with tike.ptycho.Reconstruction(
                    data=data,
                    parameters=ptychoParameters,
                    num_gpu=numGpus,
                    use_mpi=False,
            ) as context:
                context.iterate(ptychoParameters.algorithm_options.num_iter)
            result = context.parameters

        logger.debug(f'Result: {pprint.pformat(result)}')

        scanOutput: Scan | None = None
        probeOutputArray: ProbeArrayType | None = None
        objectOutputArray: ObjectArrayType | None = None

        if self._positionCorrectionSettings.usePositionCorrection.value:
            pointDict: dict[int, ScanPoint] = dict()

            for index, xy in zip(parameters.scan, result.scan):
                objectPoint = ObjectPoint(x=xy[1], y=xy[0])
                pointDict[index] = objectGrid.mapObjectPointToScanPoint(objectPoint)

            scanOutput = TabularScan(pointDict)

        if self._probeCorrectionSettings.useProbeCorrection.value:
            probeOutputArray = result.probe[0, 0]

        if self._objectCorrectionSettings.useObjectCorrection.value:
            objectOutputArray = result.psi

        return ReconstructOutput(
            scan=scanOutput,
            probeArray=probeOutputArray,
            objectArray=objectOutputArray,
            objective=result.algorithm_options.costs,
            plot2D=self._plotCosts(result.algorithm_options.costs),
            result=0,
        )


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

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.batch_method = self._settings.batchMethod.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.value
        self._algorithmOptions.alpha = float(self._settings.alpha.value)
        return self._tikeReconstructor(parameters, self._algorithmOptions)


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

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.batch_method = self._settings.batchMethod.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.value
        return self._tikeReconstructor(parameters, self._algorithmOptions)


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

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        self._algorithmOptions.num_batch = self._settings.numBatch.value
        self._algorithmOptions.batch_method = self._settings.batchMethod.value
        self._algorithmOptions.num_iter = self._settings.numIter.value
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.value
        return self._tikeReconstructor(parameters, self._algorithmOptions)
