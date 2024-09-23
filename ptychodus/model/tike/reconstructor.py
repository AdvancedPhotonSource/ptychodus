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

        if settings.useObjectCorrection.getValue():
            options = tike.ptycho.ObjectOptions(
                positivity_constraint=float(settings.positivityConstraint.getValue()),
                smoothness_constraint=float(settings.smoothnessConstraint.getValue()),
                use_adaptive_moment=settings.useAdaptiveMoment.getValue(),
                vdecay=float(settings.vdecay.getValue()),
                mdecay=float(settings.mdecay.getValue()),
                clip_magnitude=settings.useMagnitudeClipping.getValue(),
            )

        return options

    def getPositionOptions(self,
                           initialScan: numpy.typing.NDArray[Any]) -> tike.ptycho.PositionOptions:
        settings = self._positionCorrectionSettings
        options = None

        if settings.usePositionCorrection.getValue():
            options = tike.ptycho.PositionOptions(
                initial_scan=initialScan,
                use_adaptive_moment=settings.useAdaptiveMoment.getValue(),
                vdecay=float(settings.vdecay.getValue()),
                mdecay=float(settings.mdecay.getValue()),
                use_position_regularization=settings.usePositionRegularization.getValue(),
                update_magnitude_limit=float(settings.updateMagnitudeLimit.getValue()),
            )

        return options

    def getProbeOptions(self) -> tike.ptycho.ProbeOptions:
        settings = self._probeCorrectionSettings
        options = None

        if settings.useProbeCorrection.getValue():
            probeSupport = float(settings.probeSupportWeight.getValue()) \
                    if settings.useFiniteProbeSupport.getValue() else 0.

            options = tike.ptycho.ProbeOptions(
                force_orthogonality=settings.forceOrthogonality.getValue(),
                force_centered_intensity=settings.forceCenteredIntensity.getValue(),
                force_sparsity=float(settings.forceSparsity.getValue()),
                use_adaptive_moment=settings.useAdaptiveMoment.getValue(),
                vdecay=float(settings.vdecay.getValue()),
                mdecay=float(settings.mdecay.getValue()),
                probe_support=probeSupport,
                probe_support_radius=float(settings.probeSupportRadius.getValue()),
                probe_support_degree=float(settings.probeSupportDegree.getValue()),
                additional_probe_penalty=float(settings.additionalProbePenalty.getValue()),
            )

        return options

    def getNumGpus(self) -> int | tuple[int, ...]:
        numGpus = self._settings.numGpus.getValue()
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
            noise_model=self._settings.noiseModel.getValue(),
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

        if self._multigridSettings.useMultigrid.getValue():
            result = tike.ptycho.reconstruct_multigrid(
                data=patternsArray,
                parameters=ptychoParameters,
                num_gpu=numGpus,
                use_mpi=False,
                num_levels=self._multigridSettings.numLevels.getValue(),
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

        if self._probeCorrectionSettings.useProbeCorrection.getValue():
            probeOutput = Probe(
                array=result.probe[0, 0],
                pixelWidthInMeters=probeInput.pixelWidthInMeters,
                pixelHeightInMeters=probeInput.pixelHeightInMeters,
            )
        else:
            probeOutput = probeInput.copy()

        if self._objectCorrectionSettings.useObjectCorrection.getValue():
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
        self._algorithmOptions.num_batch = self._settings.numBatch.getValue()
        self._algorithmOptions.batch_method = self._settings.batchMethod.getValue()
        self._algorithmOptions.num_iter = self._settings.numIter.getValue()
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.getValue()
        self._algorithmOptions.alpha = float(self._settings.alpha.getValue())
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
        self._algorithmOptions.num_batch = self._settings.numBatch.getValue()
        self._algorithmOptions.batch_method = self._settings.batchMethod.getValue()
        self._algorithmOptions.num_iter = self._settings.numIter.getValue()
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.getValue()
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
        self._algorithmOptions.num_batch = self._settings.numBatch.getValue()
        self._algorithmOptions.batch_method = self._settings.batchMethod.getValue()
        self._algorithmOptions.num_iter = self._settings.numIter.getValue()
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.getValue()
        return self._tikeReconstructor(parameters, self._algorithmOptions)
