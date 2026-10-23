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

from pydantic import BaseModel

from ptychodus.api.reconstruct import ReconstructInput, SubprocessReconstructor

from .options import PtychoFMOptions, default_options
from .training_file import PtychoFMProductFileReader, save_training_data

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
        save_training_data(
            file_path, parameters, img_size=int(_declared(options).model.encoder.img_size)
        )

    return SubprocessReconstructor(
        name=name,
        default_options=default_options,
        dump_reconstruct_options=dump_reconstruct_options,
        is_trainable=True,
        dump_train_options=dump_train_options,
        model_file_filter='PyTorch Checkpoint (*.pth *.pt)',
        model_file_extension='.pth',
        # The same filter the reader registers, so the export and import dialogs
        # name one format once.
        training_data_file_filter=PtychoFMProductFileReader().get_file_filter(),
        export_training_data=export_training_data,
    )
