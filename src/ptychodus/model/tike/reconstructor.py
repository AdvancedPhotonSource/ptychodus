from typing import Any
import logging
import pprint

import numpy
import numpy.typing

import tike.ptycho

from ptychodus.api.object import Object, ObjectPoint
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    Reconstructor,
    ReconstructInput,
    ReconstructOutput,
)
from ptychodus.api.scan import PositionSequence, ScanPoint

from .settings import (
    TikeMultigridSettings,
    TikeObjectCorrectionSettings,
    TikePositionCorrectionSettings,
    TikeProbeCorrectionSettings,
    TikeSettings,
)

logger = logging.getLogger(__name__)


class TikeReconstructor:
    def __init__(
        self,
        settings: TikeSettings,
        multigridSettings: TikeMultigridSettings,
        positionCorrectionSettings: TikePositionCorrectionSettings,
        probeCorrectionSettings: TikeProbeCorrectionSettings,
        objectCorrectionSettings: TikeObjectCorrectionSettings,
    ) -> None:
        self._settings = settings
        self._multigridSettings = multigridSettings
        self._positionCorrectionSettings = positionCorrectionSettings
        self._probeCorrectionSettings = probeCorrectionSettings
        self._objectCorrectionSettings = objectCorrectionSettings

    def getObjectOptions(self) -> tike.ptycho.ObjectOptions:
        settings = self._objectCorrectionSettings
        options = None

        if settings.useObjectCorrection.get_value():
            options = tike.ptycho.ObjectOptions(
                positivity_constraint=float(settings.positivityConstraint.get_value()),
                smoothness_constraint=float(settings.smoothnessConstraint.get_value()),
                use_adaptive_moment=settings.useAdaptiveMoment.get_value(),
                vdecay=float(settings.vdecay.get_value()),
                mdecay=float(settings.mdecay.get_value()),
                clip_magnitude=settings.useMagnitudeClipping.get_value(),
            )

        return options

    def getPositionOptions(
        self, initialScan: numpy.typing.NDArray[Any]
    ) -> tike.ptycho.PositionOptions:
        settings = self._positionCorrectionSettings
        options = None

        if settings.usePositionCorrection.get_value():
            options = tike.ptycho.PositionOptions(
                initial_scan=initialScan,
                use_adaptive_moment=settings.useAdaptiveMoment.get_value(),
                vdecay=float(settings.vdecay.get_value()),
                mdecay=float(settings.mdecay.get_value()),
                use_position_regularization=settings.usePositionRegularization.get_value(),
                update_magnitude_limit=float(settings.updateMagnitudeLimit.get_value()),
            )

        return options

    def getProbeOptions(self) -> tike.ptycho.ProbeOptions:
        settings = self._probeCorrectionSettings
        options = None

        if settings.useProbeCorrection.get_value():
            probeSupport = (
                float(settings.probeSupportWeight.get_value())
                if settings.useFiniteProbeSupport.get_value()
                else 0.0
            )

            options = tike.ptycho.ProbeOptions(
                force_orthogonality=settings.forceOrthogonality.get_value(),
                force_centered_intensity=settings.forceCenteredIntensity.get_value(),
                force_sparsity=float(settings.forceSparsity.get_value()),
                use_adaptive_moment=settings.useAdaptiveMoment.get_value(),
                vdecay=float(settings.vdecay.get_value()),
                mdecay=float(settings.mdecay.get_value()),
                probe_support=probeSupport,
                probe_support_radius=float(settings.probeSupportRadius.get_value()),
                probe_support_degree=float(settings.probeSupportDegree.get_value()),
                additional_probe_penalty=float(settings.additionalProbePenalty.get_value()),
            )

        return options

    def getNumGpus(self) -> int | tuple[int, ...]:
        numGpus = self._settings.numGpus.get_value()
        onlyDigitsAndCommas = all(c.isdigit() or c == ',' for c in numGpus)
        hasDigit = any(c.isdigit() for c in numGpus)

        if onlyDigitsAndCommas and hasDigit:
            if ',' in numGpus:
                return tuple(int(n) for n in numGpus.split(',') if n)
            else:
                return int(numGpus)

        return 1

    def __call__(
        self,
        parameters: ReconstructInput,
        algorithmOptions: tike.ptycho.solvers.IterativeOptions,
    ) -> ReconstructOutput:
        patternsArray = numpy.fft.ifftshift(parameters.patterns, axes=(-2, -1))

        objectInput = parameters.product.object_
        objectGeometry = objectInput.get_geometry()
        objectInputArray = objectInput.get_array().astype('complex64')
        numberOfLayers = objectInput.num_layers

        if numberOfLayers == 1:
            objectInputArray = objectInputArray[0]
        else:
            raise ValueError(f'Tike does not support multislice (layers={numberOfLayers})!')

        probeInput = parameters.product.probe
        probeInputArray = probeInput.get_array().astype('complex64')

        scanInput = parameters.product.positions
        scanInputCoords: list[float] = list()

        # Tike coordinate system origin is top-left corner; requires padding
        ux = -probeInputArray.shape[-1] / 2
        uy = -probeInputArray.shape[-2] / 2

        for scanPoint in scanInput:
            objectPoint = objectGeometry.map_scan_point_to_object_point(scanPoint)
            scanInputCoords.append(objectPoint.position_y_px + uy)
            scanInputCoords.append(objectPoint.position_x_px + ux)

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
            measured_pixels=numpy.logical_not(parameters.bad_pixels),
            noise_model=self._settings.noiseModel.get_value(),
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

        if self._multigridSettings.useMultigrid.get_value():
            result = tike.ptycho.reconstruct_multigrid(
                data=patternsArray,
                parameters=ptychoParameters,
                num_gpu=numGpus,
                use_mpi=False,
                num_levels=self._multigridSettings.numLevels.get_value(),
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
            scanPoint = objectGeometry.map_object_point_to_scan_point(objectPoint)
            scanOutputPoints.append(scanPoint)

        scanOutput = PositionSequence(scanOutputPoints)

        if self._probeCorrectionSettings.useProbeCorrection.get_value():
            probeOutput = Probe(array=result.probe, pixel_geometry=probeInput.get_pixel_geometry())
        else:
            probeOutput = probeInput.copy()

        if self._objectCorrectionSettings.useObjectCorrection.get_value():
            objectOutput = Object(
                array=result.psi,
                layer_distance_m=objectInput.layer_distance_m,
                pixel_geometry=objectInput.get_pixel_geometry(),
                center=objectInput.get_center(),
            )
        else:
            objectOutput = objectInput.copy()

        product = Product(
            metadata=parameters.product.metadata,
            positions=scanOutput,
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
        self._algorithmOptions.num_batch = self._settings.numBatch.get_value()
        self._algorithmOptions.batch_method = self._settings.batchMethod.get_value()
        self._algorithmOptions.num_iter = self._settings.numIter.get_value()
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.get_value()
        self._algorithmOptions.alpha = float(self._settings.alpha.get_value())
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
        self._algorithmOptions.num_batch = self._settings.numBatch.get_value()
        self._algorithmOptions.batch_method = self._settings.batchMethod.get_value()
        self._algorithmOptions.num_iter = self._settings.numIter.get_value()
        self._algorithmOptions.convergence_window = self._settings.convergenceWindow.get_value()
        return self._tikeReconstructor(parameters, self._algorithmOptions)
