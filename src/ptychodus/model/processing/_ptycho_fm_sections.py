"""Where each PtychoFM option is persisted in ptychodus's flat settings layout.

PtychoFM declares 165 fields and ptychodus persists 15, so this table is written
the other way up from pty-chi's: each policy names the subset it surfaces and
everything else is hidden. An upstream addition then stays out of the settings
file until someone chooses to put it there.

Three branches, not four. The architecture has no section at all: a ``.pth``
holds a bare state dict naming no architecture, so the sizes a checkpoint was
trained with are not a user's to vary, and they are frozen in the declaration
instead. ``options.py`` states them, and the dump carries them to the child
whether or not a settings group exists.

ptychodus persists PtychoFM keys as the literal field names, so the policy is
verbatim.

Upstream carries nothing a reader would need: no title, no description, no
bound, and every choice typed as a bare ``str`` rather than an enum. All of it
is supplied here, in the policy's overlay.

**Transitional**, with the flat layout it serves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .flatten import (
    EnumSpelling,
    FlatteningPolicy,
    NamingConvention,
    Overlay,
    Sentinel,
    fields_of,
)
from .parameter_store import OptionsSection

__all__ = ['sections']


def _hide_all_but(model: type, prefix: str, *exposed: str) -> set[str]:
    """Hide every field of *model* except *exposed*, under a dotted *prefix*."""
    return {f'{prefix}{name}' for name in fields_of(model) if name not in exposed}


_DATA_OVERLAYS: Mapping[str, Overlay] = {
    'scale': Overlay(
        'Diffraction Scale:',
        'Scale factor applied to diffraction intensities before sqrt.',
        minimum=0.0,
    ),
    'default_normalization': Overlay('Default Normalization:', minimum=0.0),
    'max_probe_modes': Overlay(
        'Max Probe Modes:',
        'Probes are zero-padded to this many mixed-state modes.',
        minimum=1,
    ),
    'max_OPR_modes': Overlay(
        'Max OPR Modes:',
        'Probes are zero-padded to this many orthogonal probe relaxation modes.',
        minimum=1,
    ),
    'train_split': Overlay(
        'Train Split:',
        'Fraction of dataset used for training; remainder is validation.',
        minimum=0.0,
        maximum=1.0,
    ),
    'random_seed': Overlay('Random Seed:'),
    'max_files': Overlay('Max Files (0 = no cap):', minimum=0),
}


def _data_policy() -> FlatteningPolicy:
    """The dataset branch.

    ``data_path`` is hidden with the rest: the child fills it from the training
    payload, so it is a path ptychodus passes rather than one a user sets. So
    are the dataloader and array-cache sizings -- ``num_workers``,
    ``prefetch_factor``, ``cache_object`` and ``cache_memory_budget_mb`` --
    which follow from the machine rather than from the experiment.
    """
    from ptycho_fm.config import DataConfig

    return FlatteningPolicy(
        naming=NamingConvention.VERBATIM,
        spelling=EnumSpelling.BY_VALUE,
        collapse_onto_prefix=frozenset(),
        hidden=frozenset(
            _hide_all_but(
                DataConfig,
                '',
                'scale',
                'default_normalization',
                'max_probe_modes',
                'max_OPR_modes',
                'train_split',
                'random_seed',
                'max_files',
            )
        ),
        structural={'max_files': Sentinel('max_files', value_type=int, sentinel=0)},
        overlays=_DATA_OVERLAYS,
    )


# The three losses that are defined on both objectives and take no parameters
# of their own. The rest of upstream's are research knobs: each is configured
# by a sub-branch of its own, and the diffraction-space ones compute something
# other than what they claim when applied to an object patch.
_LOSS_FUNCTIONS: Sequence[str] = ('mse', 'l1', 'smooth_l1')

_TRAINING_OVERLAYS: Mapping[str, Overlay] = {
    'batch_size': Overlay('Batch Size:', minimum=1),
    'learning_rate': Overlay('Learning Rate:', minimum=0.0),
    'epochs': Overlay('Epochs:', minimum=1),
    'loss_function': Overlay('Loss Function:', choices=_LOSS_FUNCTIONS),
    'save_epoch_models': Overlay('Save Per-Epoch Models'),
}


def _training_policy() -> FlatteningPolicy:
    """The optimization branch.

    ``mode`` is hidden: supervised and unsupervised are two reconstructors, so
    the objective follows from which one was selected rather than from a key.
    So are the parameterized loss branches, with the losses that read them.
    """
    from ptycho_fm.config import TrainingConfig

    return FlatteningPolicy(
        naming=NamingConvention.VERBATIM,
        spelling=EnumSpelling.BY_VALUE,
        collapse_onto_prefix=frozenset(),
        hidden=frozenset(
            _hide_all_but(
                TrainingConfig,
                '',
                'batch_size',
                'learning_rate',
                'epochs',
                'loss_function',
                'save_epoch_models',
            )
        ),
        overlays=_TRAINING_OVERLAYS,
    )


def _inference_policy() -> FlatteningPolicy:
    """The stitching branch, which is ptychodus's own and states its own text."""
    return FlatteningPolicy(
        naming=NamingConvention.VERBATIM,
        spelling=EnumSpelling.BY_VALUE,
        collapse_onto_prefix=frozenset(),
    )


def sections() -> Sequence[OptionsSection]:
    """Return the three settings groups backing PtychoFM."""
    return [
        OptionsSection('PtychoFMData', root_path='data', policy=_data_policy()),
        OptionsSection('PtychoFMTraining', root_path='training', policy=_training_policy()),
        OptionsSection('PtychoFMInference', root_path='inference', policy=_inference_policy()),
    ]
