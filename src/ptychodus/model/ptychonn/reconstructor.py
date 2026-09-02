"""Parent-side factory that builds :class:`SubprocessReconstructor`s for PtychoNN.

Zero ptychonn / torch / lightning imports. All GPU work runs inside a
spawned child; see :mod:`._subprocess` for the child entry points.

Training-data export runs parent-side because it is pure-numpy (barycentric
interpolation + numpy.savez); the ptychonn / lightning stack is not touched.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.interpolate import BarycentricArrayInterpolator
from ptychodus.api.reconstruct import ReconstructInput

from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import (
    PtychoNNReconstructConfig,
    PtychoNNTrainConfig,
    ReconstructPayload,
    TrainPayload,
)
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

__all__ = [
    'build_reconstructor',
]

logger = logging.getLogger(__name__)


_RECONSTRUCT_ENTRY = 'ptychodus.model.ptychonn._subprocess:run_reconstruct'
_TRAIN_ENTRY = 'ptychodus.model.ptychonn._subprocess:run_train'
_PATCHES_KEY = 'real'
_PATTERNS_KEY = 'reciprocal'


def _export_training_data(
    file_path: Path, parameters: ReconstructInput, *, num_channels: int
) -> None:
    object_geometry = parameters.product.object_.get_geometry()
    interpolator = BarycentricArrayInterpolator(parameters.product.object_.get_array())
    probe_extent = ImageExtent(
        width_px=parameters.product.probes.width_px,
        height_px=parameters.product.probes.height_px,
    )
    patches = numpy.zeros(
        (len(parameters.product.probe_positions), num_channels, *probe_extent.get_shape()),
        dtype=numpy.float32,
    )

    for index, scan_point in enumerate(parameters.product.probe_positions):
        object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
        patch = interpolator.get_patch(
            object_point.x_px,
            object_point.y_px,
            probe_extent.width_px,
            probe_extent.height_px,
        )
        patches[index, 0, :, :] = numpy.angle(patch)
        if num_channels > 1:
            patches[index, 1, :, :] = numpy.absolute(patch)

    logger.debug(f'Writing "{file_path}" as "NPZ"')
    contents = {
        _PATTERNS_KEY: parameters.diffraction_patterns.astype(numpy.float32),
        _PATCHES_KEY: patches,
    }
    numpy.savez_compressed(file_path, allow_pickle=False, **contents)


def _build_reconstruct_config(
    *,
    enable_amplitude: bool,
    model_settings: PtychoNNModelSettings,
    training_settings: PtychoNNTrainingSettings,
) -> PtychoNNReconstructConfig:
    return PtychoNNReconstructConfig(
        enable_amplitude=enable_amplitude,
        num_convolution_kernels=model_settings.num_convolution_kernels.get_value(),
        use_batch_normalization=model_settings.use_batch_normalization.get_value(),
        max_learning_rate=float(training_settings.max_learning_rate.get_value()),
        min_learning_rate=float(training_settings.min_learning_rate.get_value()),
    )


def _build_train_config(
    *,
    enable_amplitude: bool,
    model_settings: PtychoNNModelSettings,
    training_settings: PtychoNNTrainingSettings,
) -> PtychoNNTrainConfig:
    return PtychoNNTrainConfig(
        enable_amplitude=enable_amplitude,
        num_convolution_kernels=model_settings.num_convolution_kernels.get_value(),
        use_batch_normalization=model_settings.use_batch_normalization.get_value(),
        max_learning_rate=float(training_settings.max_learning_rate.get_value()),
        min_learning_rate=float(training_settings.min_learning_rate.get_value()),
        batch_size=model_settings.batch_size.get_value(),
        training_epochs=training_settings.training_epochs.get_value(),
        status_interval_in_epochs=training_settings.status_interval_in_epochs.get_value(),
        validation_set_fractional_size=float(
            training_settings.validation_set_fractional_size.get_value()
        ),
    )


def build_reconstructor(
    display_name: str,
    *,
    enable_amplitude: bool,
    model_settings: PtychoNNModelSettings,
    training_settings: PtychoNNTrainingSettings,
) -> SubprocessReconstructor:
    num_channels = 2 if enable_amplitude else 1

    def build_reconstruct_payload(
        parameters: ReconstructInput, loaded_model_path: Path | None
    ) -> ReconstructPayload:
        return ReconstructPayload(
            config=_build_reconstruct_config(
                enable_amplitude=enable_amplitude,
                model_settings=model_settings,
                training_settings=training_settings,
            ),
            model_path=loaded_model_path,
            reconstruct_input=parameters,
        )

    def build_train_payload(input_path: Path, output_path: Path) -> TrainPayload:
        return TrainPayload(
            config=_build_train_config(
                enable_amplitude=enable_amplitude,
                model_settings=model_settings,
                training_settings=training_settings,
            ),
            input_path=input_path,
            output_path=output_path,
        )

    def export_training_data(file_path: Path, parameters: ReconstructInput) -> None:
        _export_training_data(file_path, parameters, num_channels=num_channels)

    return SubprocessReconstructor(
        name=display_name,
        reconstruct_entry_point=_RECONSTRUCT_ENTRY,
        progress_goal_fn=lambda: 0,
        build_reconstruct_payload=build_reconstruct_payload,
        is_trainable=True,
        train_entry_point=_TRAIN_ENTRY,
        build_train_payload=build_train_payload,
        model_file_filter='PyTorch Lightning Checkpoint Files (*.ckpt)',
        model_file_extension='.ckpt',
        training_data_file_filter='NumPy Zipped Archive (*.npz)',
        export_training_data=export_training_data,
    )
