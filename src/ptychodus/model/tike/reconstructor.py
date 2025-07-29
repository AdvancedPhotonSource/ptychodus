from typing import Any
import logging
import pprint

import numpy
import numpy.typing

import tike.ptycho

from ptychodus.api.object import Object, ObjectPoint
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import LossValue, Product
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
        multigrid_settings: TikeMultigridSettings,
        position_correction_settings: TikePositionCorrectionSettings,
        probe_correction_settings: TikeProbeCorrectionSettings,
        object_correction_settings: TikeObjectCorrectionSettings,
    ) -> None:
        self._settings = settings
        self._multigrid_settings = multigrid_settings
        self._position_correction_settings = position_correction_settings
        self._probe_correction_settings = probe_correction_settings
        self._object_correction_settings = object_correction_settings

    def get_object_options(self) -> tike.ptycho.ObjectOptions:
        settings = self._object_correction_settings
        options = None

        if settings.use_object_correction.get_value():
            options = tike.ptycho.ObjectOptions(
                positivity_constraint=float(settings.positivity_constraint.get_value()),
                smoothness_constraint=float(settings.smoothness_constraint.get_value()),
                use_adaptive_moment=settings.use_adaptive_moment.get_value(),
                vdecay=float(settings.vdecay.get_value()),
                mdecay=float(settings.mdecay.get_value()),
                clip_magnitude=settings.use_magnitude_clipping.get_value(),
            )

        return options

    def get_position_options(
        self, initial_scan: numpy.typing.NDArray[Any]
    ) -> tike.ptycho.PositionOptions:
        settings = self._position_correction_settings
        options = None

        if settings.use_position_correction.get_value():
            options = tike.ptycho.PositionOptions(
                initial_scan=initial_scan,
                use_adaptive_moment=settings.use_adaptive_moment.get_value(),
                vdecay=float(settings.vdecay.get_value()),
                mdecay=float(settings.mdecay.get_value()),
                use_position_regularization=settings.use_position_regularization.get_value(),
                update_magnitude_limit=float(settings.update_magnitude_limit.get_value()),
            )

        return options

    def get_probe_options(self) -> tike.ptycho.ProbeOptions:
        settings = self._probe_correction_settings
        options = None

        if settings.use_probe_correction.get_value():
            probe_support = (
                float(settings.probe_support_weight.get_value())
                if settings.use_finite_probe_support.get_value()
                else 0.0
            )

            options = tike.ptycho.ProbeOptions(
                force_orthogonality=settings.force_orthogonality.get_value(),
                force_centered_intensity=settings.force_centered_intensity.get_value(),
                force_sparsity=float(settings.force_sparsity.get_value()),
                use_adaptive_moment=settings.use_adaptive_moment.get_value(),
                vdecay=float(settings.vdecay.get_value()),
                mdecay=float(settings.mdecay.get_value()),
                probe_support=probe_support,
                probe_support_radius=float(settings.probe_support_radius.get_value()),
                probe_support_degree=float(settings.probe_support_degree.get_value()),
                additional_probe_penalty=float(settings.additional_probe_penalty.get_value()),
            )

        return options

    def get_num_gpus(self) -> int | tuple[int, ...]:
        num_gpus = self._settings.num_gpus.get_value()
        only_digits_and_commas = all(c.isdigit() or c == ',' for c in num_gpus)
        has_digit = any(c.isdigit() for c in num_gpus)

        if only_digits_and_commas and has_digit:
            if ',' in num_gpus:
                return tuple(int(n) for n in num_gpus.split(',') if n)
            else:
                return int(num_gpus)

        return 1

    def __call__(
        self,
        parameters: ReconstructInput,
        algorithm_options: tike.ptycho.solvers.IterativeOptions,
    ) -> ReconstructOutput:
        patterns_array = numpy.fft.ifftshift(parameters.diffraction_patterns, axes=(-2, -1))

        object_input = parameters.product.object_
        object_geometry = object_input.get_geometry()
        object_input_array = object_input.get_array().astype('complex64')
        num_layers = object_input.num_layers

        if num_layers == 1:
            object_input_array = object_input_array[0]
        else:
            raise ValueError(f'Tike does not support multislice (layers={num_layers})!')

        probe_input = parameters.product.probes
        probe_input_array = probe_input.get_array().astype('complex64')

        scan_input = parameters.product.positions
        scan_input_coords: list[float] = list()

        # Tike coordinate system origin is top-left corner; requires padding
        ux = -probe_input_array.shape[-1] / 2
        uy = -probe_input_array.shape[-2] / 2

        for scan_point in scan_input:
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            scan_input_coords.append(object_point.position_y_px + uy)
            scan_input_coords.append(object_point.position_x_px + ux)

        scan_input_array = numpy.array(
            scan_input_coords,
            dtype=numpy.float32,
        ).reshape(len(scan_input), 2)
        scan_min = scan_input_array.min(axis=0)
        scan_max = scan_input_array.max(axis=0)
        logger.debug(f'Scan range [px]: {scan_min} -> {scan_max}')
        num_gpus = self.get_num_gpus()

        logger.debug(f'data shape={patterns_array.shape}')
        logger.debug(f'scan shape={scan_input_array.shape}')
        logger.debug(f'probe shape={probe_input_array.shape}')
        logger.debug(f'object shape={object_input_array.shape}')
        logger.debug(f'num_gpu={num_gpus}')

        exitwave_options = tike.ptycho.ExitWaveOptions(
            measured_pixels=numpy.logical_not(parameters.bad_pixels),
            noise_model=self._settings.noise_model.get_value(),
        )

        ptycho_parameters = tike.ptycho.solvers.PtychoParameters(
            probe=probe_input_array,
            psi=object_input_array,
            scan=scan_input_array,
            algorithm_options=algorithm_options,
            probe_options=self.get_probe_options(),
            object_options=self.get_object_options(),
            position_options=self.get_position_options(scan_input_array),
            exitwave_options=exitwave_options,
        )

        if self._multigrid_settings.use_multigrid.get_value():
            result = tike.ptycho.reconstruct_multigrid(
                data=patterns_array,
                parameters=ptycho_parameters,
                num_gpu=num_gpus,
                use_mpi=False,
                num_levels=self._multigrid_settings.num_levels.get_value(),
                interp=None,  # TODO does this have other options?
            )
        else:
            # TODO support interactive reconstructions
            with tike.ptycho.Reconstruction(
                data=patterns_array,
                parameters=ptycho_parameters,
                num_gpu=num_gpus,
                use_mpi=False,
            ) as context:
                context.iterate(ptycho_parameters.algorithm_options.num_iter)
            result = context.parameters

        logger.debug(f'Result: {pprint.pformat(result)}')

        scan_output_points: list[ScanPoint] = list()

        for uncorrected_point, xy in zip(scan_input, result.scan):
            object_point = ObjectPoint(uncorrected_point.index, xy[1] - ux, xy[0] - uy)
            scan_point = object_geometry.map_object_point_to_scan_point(object_point)
            scan_output_points.append(scan_point)

        scan_output = PositionSequence(scan_output_points)

        if self._probe_correction_settings.use_probe_correction.get_value():
            probe_output = ProbeSequence(
                array=result.probe,
                opr_weights=None,
                pixel_geometry=probe_input.get_pixel_geometry(),
            )
        else:
            probe_output = probe_input.copy()

        if self._object_correction_settings.use_object_correction.get_value():
            object_output = Object(
                array=result.psi,
                layer_spacing_m=object_input.layer_spacing_m,
                pixel_geometry=object_input.get_pixel_geometry(),
                center=object_input.get_center(),
            )
        else:
            object_output = object_input.copy()

        losses: list[LossValue] = list()

        for epoch, values in enumerate(result.algorithm_options.costs):
            loss = LossValue(epoch=epoch, value=float(numpy.mean(values)))
            losses.append(loss)

        product = Product(
            metadata=parameters.product.metadata,
            positions=scan_output,
            probes=probe_output,
            object_=object_output,
            losses=losses,
        )
        return ReconstructOutput(product, 0)


class RegularizedPIEReconstructor(Reconstructor):
    def __init__(self, tike_reconstructor: TikeReconstructor) -> None:
        super().__init__()
        self._algorithm_options = tike.ptycho.solvers.RpieOptions()
        self._tike_reconstructor = tike_reconstructor

    @property
    def name(self) -> str:
        return self._algorithm_options.name

    @property
    def _settings(self) -> TikeSettings:
        return self._tike_reconstructor._settings

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        self._algorithm_options.num_batch = self._settings.num_batch.get_value()
        self._algorithm_options.batch_method = self._settings.batch_method.get_value()
        self._algorithm_options.num_iter = self._settings.num_iter.get_value()
        self._algorithm_options.convergence_window = self._settings.convergence_window.get_value()
        self._algorithm_options.alpha = float(self._settings.alpha.get_value())
        return self._tike_reconstructor(parameters, self._algorithm_options)


class IterativeLeastSquaresReconstructor(Reconstructor):
    def __init__(self, tike_reconstructor: TikeReconstructor) -> None:
        super().__init__()
        self._algorithm_options = tike.ptycho.solvers.LstsqOptions()
        self._tike_reconstructor = tike_reconstructor

    @property
    def name(self) -> str:
        return self._algorithm_options.name

    @property
    def _settings(self) -> TikeSettings:
        return self._tike_reconstructor._settings

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        self._algorithm_options.num_batch = self._settings.num_batch.get_value()
        self._algorithm_options.batch_method = self._settings.batch_method.get_value()
        self._algorithm_options.num_iter = self._settings.num_iter.get_value()
        self._algorithm_options.convergence_window = self._settings.convergence_window.get_value()
        return self._tike_reconstructor(parameters, self._algorithm_options)
