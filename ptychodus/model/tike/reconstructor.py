from importlib.metadata import version
from typing import Any
import logging
import pprint

import numpy
import numpy.typing

import tike.ptycho

from ptychodus.api.object import Object, ObjectPoint
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import Reconstructor, ReconstructInput, ReconstructOutput
from ptychodus.api.scan import Scan, ScanPoint

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
                update_magnitude_limit=float(settings.updateMagnitudeLimit.value),
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

    def getNumGpus(self) -> int | tuple[int, ...]:
        numGpus = self._settings.numGpus.value
        onlyDigitsAndCommas = all(c.isdigit() or c == ',' for c in numGpus)
        hasDigit = any(c.isdigit() for c in numGpus)

        if onlyDigitsAndCommas and hasDigit:
            if ',' in numGpus:
                return tuple(int(n) for n in numGpus.split(',') if n)
            else:
                return int(numGpus)

        return 1

    def __call__(self, parameters: ReconstructInput,
                 algorithmOptions: tike.ptycho.solvers.IterativeOptions) -> ReconstructOutput:
        patternsArray = numpy.fft.ifftshift(parameters.patterns, axes=(-2, -1))

        objectInput = parameters.product.object_
        objectGeometry = objectInput.getGeometry()
        # TODO change array[0] -> array when multislice is available
        objectInputArray = objectInput.array[0].astype('complex64')

        probeInput = parameters.product.probe
        probeInputArray = probeInput.array[numpy.newaxis, numpy.newaxis, ...].astype('complex64')

        scanInput = parameters.product.scan
        scanInputCoords: list[float] = list()

        # Tike coordinate system origin is top-left corner; requires padding
        ux = -probeInputArray.shape[-1] / 2
        uy = -probeInputArray.shape[-2] / 2

        for scanPoint in scanInput:
            objectPoint = objectGeometry.mapScanPointToObjectPoint(scanPoint)
            scanInputCoords.append(objectPoint.positionYInPixels + uy)
            scanInputCoords.append(objectPoint.positionXInPixels + ux)

        scanInputArray = numpy.array(
            scanInputCoords,
            dtype=numpy.float32,
        ).reshape(len(scanInput), 2)
        scanMin = scanInputArray.min(axis=0)
        scanMax = scanInputArray.max(axis=0)
        logger.debug(f'Scan range [px]: {scanMin} -> {scanMax}')
        numGpus = self.getNumGpus()

        logger.debug(f'data shape={patternsArray.shape}')
        logger.debug(f'scan shape={scanInputArray.shape}')
        logger.debug(f'probe shape={probeInputArray.shape}')
        logger.debug(f'object shape={objectInputArray.shape}')
        logger.debug(f'num_gpu={numGpus}')

        exitwave_options = tike.ptycho.ExitWaveOptions(
            # TODO: Use a user supplied `measured_pixels` instead
            measured_pixels=numpy.ones(probeInputArray.shape[-2:], dtype=numpy.bool_),
            noise_model=self._settings.noiseModel.value,
        )

        ptychoParameters = tike.ptycho.solvers.PtychoParameters(
            probe=probeInputArray,
            psi=objectInputArray,
            scan=scanInputArray,
            algorithm_options=algorithmOptions,
            probe_options=self.getProbeOptions(),
            object_options=self.getObjectOptions(),
            position_options=self.getPositionOptions(scanInputArray),
            exitwave_options=exitwave_options,
        )

        if self._multigridSettings.useMultigrid.value:
            result = tike.ptycho.reconstruct_multigrid(
                data=patternsArray,
                parameters=ptychoParameters,
                num_gpu=numGpus,
                use_mpi=False,
                num_levels=self._multigridSettings.numLevels.value,
                interp=None,  # TODO does this have other options?
            )
        else:
            # TODO support interactive reconstructions
            with tike.ptycho.Reconstruction(
                    data=patternsArray,
                    parameters=ptychoParameters,
                    num_gpu=numGpus,
                    use_mpi=False,
            ) as context:
                context.iterate(ptychoParameters.algorithm_options.num_iter)
            result = context.parameters

        logger.debug(f'Result: {pprint.pformat(result)}')

        scanOutputPoints: list[ScanPoint] = list()

        for uncorrectedPoint, xy in zip(scanInput, result.scan):
            objectPoint = ObjectPoint(uncorrectedPoint.index, xy[1] - ux, xy[0] - uy)
            scanPoint = objectGeometry.mapObjectPointToScanPoint(objectPoint)
            scanOutputPoints.append(scanPoint)

        scanOutput = Scan(scanOutputPoints)

        if self._probeCorrectionSettings.useProbeCorrection.value:
            probeOutput = Probe(
                array=result.probe[0, 0],
                pixelWidthInMeters=probeInput.pixelWidthInMeters,
                pixelHeightInMeters=probeInput.pixelHeightInMeters,
            )
        else:
            probeOutput = probeInput.copy()

        if self._objectCorrectionSettings.useObjectCorrection.value:
            objectOutput = Object(
                array=result.psi,
                layerDistanceInMeters=objectInput.layerDistanceInMeters,
                pixelWidthInMeters=objectInput.pixelWidthInMeters,
                pixelHeightInMeters=objectInput.pixelHeightInMeters,
                centerXInMeters=objectInput.centerXInMeters,
                centerYInMeters=objectInput.centerYInMeters,
            )
        else:
            objectOutput = objectInput.copy()

        product = Product(
            metadata=parameters.product.metadata,
            scan=scanOutput,
            probe=probeOutput,
            object_=objectOutput,
            costs=[float(numpy.mean(values)) for values in result.algorithm_options.costs],
        )
        return ReconstructOutput(product, 0)


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
