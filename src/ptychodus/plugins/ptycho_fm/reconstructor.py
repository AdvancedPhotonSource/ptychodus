"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoFM.

Zero torch imports. All GPU work runs inside a spawned child; see
:mod:`._subprocess` for the child entry points.

PtychoFM's own ``config.yaml`` is a nested dict of scalars, so ``_build_config``
produces a plain :class:`dict` from the ptychodus settings groups -- pickleable
without pulling any ptycho_fm module in. The child feeds that dict to
``resolve_inference_model`` and to the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy

from ptychodus.api.preprocess.diffraction import zero_bad_pixels
from pydantic import BaseModel

from ptychodus.api.reconstruct import ReconstructInput, SubprocessReconstructor

from .options import PtychoFMOptions, default_options

__all__ = [
    'build_reconstruct_config',
    'build_reconstructor',
    'build_train_config',
]


# Losses evaluated on the detector plane. The supervised objective applies the
# criterion to object amplitude and phase patches instead, where a probe
# envelope, a radial detector coordinate and a total photon flux are undefined.
_DIFFRACTION_SPACE_LOSSES = frozenset({'total_flux', 'probe_aware', 'q_dependent'})


def _build_config(
    name: str, options: PtychoFMOptions, *, sections: frozenset[str]
) -> dict[str, Any]:
    """Render the branches named by *sections* as the nested dict ``ptycho_fm`` reads.

    ``exclude_unset`` is what preserves upstream's absent-means-resolve-later
    semantics: pretrained encoder sizes, validation batch size and dataset
    normalization are all decided by a field having no value, so only the
    fields ptychodus surfaces may appear. The declaration sets exactly those.
    Narrowing composes with it rather than fighting it -- the two intersect --
    so a branch that survives is spelled exactly as it would be whole.

    *sections* names whole branches and nothing finer. Inside ``model`` the
    keys are read by upstream with a default apiece, so dropping one there
    would not raise but would quietly reconstruct against a different model.

    Two keys are not settings and so are added here rather than declared.
    ``training.mode`` follows from which reconstructor was selected, and the
    encoder takes one channel in every configuration ptychodus builds.
    """
    config: dict[str, Any] = options.model_dump(exclude_unset=True, include=set(sections))

    if 'training' in config:
        config['training']['mode'] = 'supervised' if name == 'Supervised' else 'unsupervised'

    config['model']['encoder']['in_channels'] = 1
    return config


def build_reconstruct_config(name: str, options: PtychoFMOptions) -> dict[str, Any]:
    """The config an inference child is given.

    Inference resolves the model, reads all three of its own knobs, and takes a
    scale and a mode count off the data branch. It asks nothing about training,
    so nothing about training is sent -- a reconstruct payload that changed
    when a training value was edited would be indistinguishable from one that
    mattered.
    """
    return _build_config(name, options, sections=frozenset({'data', 'inference', 'model'}))


def build_train_config(name: str, options: PtychoFMOptions) -> dict[str, Any]:
    """The config a training child is given, or a complaint about the objective.

    Training reads the whole data branch and the whole training branch, and
    never looks at the inference knobs.
    """
    _verify_loss_supports_mode(name, str(options.training.loss_function))
    return _build_config(name, options, sections=frozenset({'data', 'model', 'training'}))


def _verify_spatial_size(shape: tuple[int, ...], img_size: int, what: str) -> None:
    """Raise unless a trailing ``(height, width)`` is exactly ``img_size`` square."""
    if tuple(shape) != (img_size, img_size):
        height, width = shape
        raise ValueError(
            f'{what} are {height}x{width}, but PtychoFM requires {img_size}x{img_size} '
            'to match the encoder input. Adjust the pattern crop or the model image size.'
        )


def _verify_loss_supports_mode(name: str, loss_function: str) -> None:
    """Raise unless ``loss_function`` is meaningful for the training objective.

    The diffraction-space losses do not fail on object patches, they quietly
    compute something else: a log of a signed phase clamps half the patch to the
    stability floor, and a probe envelope the caller never supplies degrades to
    a plain masked error.
    """
    if name == 'Supervised' and loss_function in _DIFFRACTION_SPACE_LOSSES:
        raise ValueError(
            f'The {loss_function!r} loss compares diffraction amplitudes, but supervised '
            'training compares the decoded object against the ground truth patch. Select '
            'an object-space loss, or train unsupervised.'
        )


def _declared(options: BaseModel) -> PtychoFMOptions:
    """Narrow the options a call was given to what this backend declares.

    Checked rather than asserted: the dispatching layer validates the type
    against what ``get_default_options`` returns, so a mismatch here means
    those two have drifted apart, and saying so beats an attribute error
    several frames later.
    """
    if not isinstance(options, PtychoFMOptions):
        raise TypeError(f'Expected PtychoFMOptions, got {type(options).__name__}.')

    return options


def build_reconstructor(name: str) -> SubprocessReconstructor:
    """Build a :class:`SubprocessReconstructor` for one PtychoFM mode.

    ``name`` is 'Unsupervised' or 'Supervised' and becomes the reconstructor's
    display name.
    """

    def dump_reconstruct_options(
        parameters: ReconstructInput, options: BaseModel
    ) -> dict[str, Any]:
        return build_reconstruct_config(name, _declared(options))

    def dump_train_options(options: BaseModel) -> dict[str, Any]:
        return build_train_config(name, _declared(options))

    def export_training_data(
        file_path: Path, parameters: ReconstructInput, options: BaseModel
    ) -> None:
        # ptycho_fm.CombinedDataset expects one directory per scan holding
        # two HDF5 files sharing a common stem:
        #     <stem>_dp.hdf5    dataset 'dp'    (N, H, W) float
        #     <stem>_para.hdf5  dataset 'object'              (1, H, W) complex
        #                                       'probe'                (1, N_modes, H, W) complex
        #                                       'probe_position_x_m'   (N,) float64
        #                                       'probe_position_y_m'   (N,) float64
        #                       'object' carries a 'pixel_height_m' attr; the
        #                       loader auto-detects meters vs pixels from range.
        # Patterns and probes are stored at their native spatial size and the
        # object patch is extracted at the pattern size, so both must already
        # match the encoder input; nothing downstream resizes them.
        # file_path is treated as a stem, so the picked filename itself is
        # never created; the two derived files land next to it.
        stem = file_path.stem
        dp_path = file_path.with_name(f'{stem}_dp.hdf5')
        para_path = file_path.with_name(f'{stem}_para.hdf5')

        dp = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)
        obj = parameters.product.object_
        object_layer = obj.get_layer(0)
        pixel_geometry = obj.get_pixel_geometry()
        probe_array = parameters.product.probes.get_probe_no_opr().get_array()

        img_size = int(_declared(options).model.encoder.img_size)
        _verify_spatial_size(dp.shape[-2:], img_size, 'Diffraction patterns')
        _verify_spatial_size(probe_array.shape[-2:], img_size, 'Probe')

        pos_x_m: list[float] = []
        pos_y_m: list[float] = []
        for point in parameters.product.probe_positions:
            pos_x_m.append(point.x_m)
            pos_y_m.append(point.y_m)

        with h5py.File(dp_path, 'w') as h5_dp:
            h5_dp.create_dataset('dp', data=dp)

        with h5py.File(para_path, 'w') as h5_para:
            obj_ds = h5_para.create_dataset('object', data=object_layer[numpy.newaxis, :, :])
            obj_ds.attrs['pixel_height_m'] = pixel_geometry.height_m
            obj_ds.attrs['pixel_width_m'] = pixel_geometry.width_m
            h5_para.create_dataset('probe', data=probe_array[numpy.newaxis, :, :, :])
            h5_para.create_dataset(
                'probe_position_x_m', data=numpy.asarray(pos_x_m, dtype=numpy.float64)
            )
            h5_para.create_dataset(
                'probe_position_y_m', data=numpy.asarray(pos_y_m, dtype=numpy.float64)
            )

    return SubprocessReconstructor(
        name=name,
        default_options=default_options,
        dump_reconstruct_options=dump_reconstruct_options,
        is_trainable=True,
        dump_train_options=dump_train_options,
        model_file_filter='PyTorch Checkpoint (*.pth *.pt)',
        model_file_extension='.pth',
        training_data_file_filter='PtychoFM Training Pair (*.hdf5)',
        export_training_data=export_training_data,
    )
