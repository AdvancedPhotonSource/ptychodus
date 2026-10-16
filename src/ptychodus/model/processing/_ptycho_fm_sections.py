"""Where each PtychoFM option is persisted in ptychodus's flat settings layout.

PtychoFM declares 165 fields and ptychodus persists 52, so this table is written
the other way up from pty-chi's: each policy names the subset it surfaces and
everything else is hidden. An upstream addition then stays out of the settings
file until someone chooses to put it there.

ptychodus persists PtychoFM keys as the literal field names, so the policy is
verbatim; the three nested branches contribute an underscore-suffixed prefix,
except the encoder branch, which is transparent -- which is how ``dropout`` and
``decoder_dropout`` coexist.

Upstream types every choice as a bare ``str``, so the accepted values come from
the policy's ``choices`` table rather than from an enum in the declaration. The
display text is ported from the Qt panel for the same reason it is for pty-chi:
an upstream declaration carries no title or description.

**Transitional**, with the flat layout it serves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .flatten import FlatteningPolicy, Label, Sentinel, fields_of
from .parameter_store import OptionsSection

__all__ = ['sections']


def _hide_all_but(model: type, prefix: str, *exposed: str) -> set[str]:
    """Hide every field of *model* except *exposed*, under a dotted *prefix*."""
    return {f'{prefix}{name}' for name in fields_of(model) if name not in exposed}


_DATA_LABELS: Mapping[str, Label] = {
    'scale': Label(
        'Diffraction Scale:', 'Scale factor applied to diffraction intensities before sqrt.'
    ),
    'default_normalization': Label('Default Normalization:'),
    'cache_object': Label('Cache Object In Memory'),
    'max_probe_modes': Label(
        'Max Probe Modes:', 'Probes are zero-padded to this many mixed-state modes.'
    ),
    'max_OPR_modes': Label(
        'Max OPR Modes:',
        'Probes are zero-padded to this many orthogonal probe relaxation modes.',
    ),
    'cache_memory_budget_mb': Label(
        'Array Cache Budget (MiB):',
        'Per-worker budget for cached object and probe arrays; 0 disables caching.',
    ),
    'train_split': Label(
        'Train Split:', 'Fraction of dataset used for training; remainder is validation.'
    ),
    'random_seed': Label('Random Seed:'),
    'max_files': Label('Max Files (0 = no cap):'),
    'num_workers': Label('Dataloader Workers:'),
    'prefetch_factor': Label('Prefetch Factor:'),
}


def _data_policy() -> FlatteningPolicy:
    """The dataset and dataloader branch.

    ``data_path`` is hidden with the rest: the child fills it from the training
    payload, so it is a path ptychodus passes rather than one a user sets.
    """
    from ptycho_fm.config import DataConfig

    return FlatteningPolicy(
        verbatim=True,
        enum_by_name=False,
        collapse_onto_prefix=frozenset(),
        hidden=frozenset(
            _hide_all_but(
                DataConfig,
                '',
                'scale',
                'default_normalization',
                'cache_object',
                'max_probe_modes',
                'max_OPR_modes',
                'cache_memory_budget_mb',
                'train_split',
                'random_seed',
                'max_files',
                'num_workers',
                'prefetch_factor',
            )
        ),
        structural={'max_files': Sentinel('max_files', value_type=int, sentinel=0)},
        labels=_DATA_LABELS,
    )


_MODEL_LABELS: Mapping[str, Label] = {
    'encoder_type': Label('Encoder Type:'),
    'coupled_decoder_mode': Label(
        'Coupled Decoder:',
        "'none' decodes amplitude and phase separately; the coupled modes share one "
        'decoder and run the forward physics. A checkpoint loads only into the mode '
        'it was trained with.',
    ),
    'encoder.img_size': Label('Image Size:'),
    'encoder.patch_size': Label('Patch Size:'),
    'encoder.embed_dim': Label('Embed Dimension:'),
    'encoder.depth': Label('Depth:'),
    'encoder.num_heads': Label('Attention Heads:'),
    'encoder.mlp_ratio': Label('MLP Ratio:'),
    'encoder.use_cls_token': Label('Use CLS Token'),
    'encoder.dropout': Label('Dropout:'),
    'encoder.attn_dropout': Label('Attention Dropout:'),
    'encoder.timm_model_name': Label(
        'TIMM Model Name:', "Only used when encoder_type is 'pretrained'."
    ),
    'decoder.base_channels': Label('Decoder Base Channels:'),
    'decoder.latent_dim': Label('Decoder Latent Dim:'),
    'decoder.num_stages': Label('Decoder Upsample Stages:'),
    'decoder.use_batchnorm': Label('Decoder Batch Norm'),
    'decoder.dropout': Label('Decoder Dropout:'),
    'init.enabled': Label('Custom Weight Init'),
    'init.method': Label('Init Method:'),
}

# 'none' is how ptychodus spells the absent mode, which builds separate
# amplitude and phase decoders; the Sentinel below turns it back into None.
_COUPLED_DECODER_MODES: Sequence[str] = ('none', 'real_imag', 'amp_phase')

_MODEL_CHOICES: Mapping[str, Sequence[str]] = {
    'encoder_type': ('custom', 'pretrained'),
    'coupled_decoder_mode': _COUPLED_DECODER_MODES,
    'init.method': ('trunc_normal', 'kaiming'),
}


def _model_policy() -> FlatteningPolicy:
    """The architecture branch, flattened across encoder, decoder and init."""
    from ptycho_fm.config import DecoderConfig, EncoderConfig, InitConfig, ModelConfig

    return FlatteningPolicy(
        verbatim=True,
        enum_by_name=False,
        # 'enabled' would otherwise collapse onto the bare 'init_' prefix,
        # where ptychodus persists it as 'init_enabled'.
        collapse_onto_prefix=frozenset(),
        hidden=frozenset(
            _hide_all_but(
                ModelConfig,
                '',
                'encoder_type',
                'coupled_decoder_mode',
                'encoder',
                'decoder',
                'init',
            )
            | _hide_all_but(
                EncoderConfig,
                'encoder.',
                'img_size',
                'patch_size',
                'embed_dim',
                'depth',
                'num_heads',
                'mlp_ratio',
                'use_cls_token',
                'dropout',
                'attn_dropout',
                'timm_model_name',
            )
            | _hide_all_but(
                DecoderConfig,
                'decoder.',
                'base_channels',
                'latent_dim',
                'num_stages',
                'use_batchnorm',
                'dropout',
            )
            | _hide_all_but(InitConfig, 'init.', 'enabled', 'method')
        ),
        prefixes={'encoder': '', 'decoder': 'decoder_', 'init': 'init_'},
        structural={
            'coupled_decoder_mode': Sentinel(
                'coupled_decoder_mode', value_type=str, sentinel='none'
            )
        },
        labels=_MODEL_LABELS,
        choices=_MODEL_CHOICES,
    )


_TRAINING_LABELS: Mapping[str, Label] = {
    'batch_size': Label('Batch Size:'),
    'learning_rate': Label('Learning Rate:'),
    'epochs': Label('Epochs:'),
    'loss_function': Label('Loss Function:'),
    'weighted_loss.loss_type': Label(
        'Weighted Loss Type:', "Only used when loss_function is 'weighted'."
    ),
    'weighted_loss.threshold': Label('Weighted Loss Threshold:'),
    'weighted_loss.alpha': Label('Weighted Loss Alpha:'),
    'probe_aware_loss.loss_type': Label(
        'Probe-Aware Loss Type:', "Only used when loss_function is 'probe_aware'."
    ),
    'probe_aware_loss.threshold': Label('Probe-Aware Loss Threshold:'),
    'probe_aware_loss.alpha': Label(
        'Probe-Aware Loss Alpha:',
        'Power applied to the probe far-field envelope in the denominator.',
    ),
    'probe_aware_loss.envelope_floor': Label(
        'Probe-Aware Envelope Floor:', 'Per-sample floor as a fraction of the mean probe envelope.'
    ),
    'q_dependent_loss.loss_type': Label(
        'Q-Dependent Loss Type:', "Only used when loss_function is 'q_dependent'."
    ),
    'q_dependent_loss.alpha': Label(
        'Q-Dependent Loss Alpha:',
        'Focal exponent emphasizing poorly matched pixels; 0 disables it.',
    ),
    'q_dependent_loss.q_beta': Label(
        'Q-Dependent Q Exponent:',
        'Radial exponent emphasizing high-Q pixels; 0 disables Q weighting.',
    ),
    'q_dependent_loss.q_floor': Label(
        'Q-Dependent Q Floor:', 'Minimum normalized Q, so low-Q pixels are not zero-weighted.'
    ),
    'q_dependent_loss.threshold': Label('Q-Dependent Loss Threshold:'),
    'q_dependent_loss.log_error': Label(
        'Q-Dependent Log Error', 'Compute the data error on log10 amplitudes.'
    ),
    'q_dependent_loss.normalize_q': Label(
        'Q-Dependent Normalize Q', 'Normalize the Q weight map to unit mean.'
    ),
    'save_epoch_models': Label('Save Per-Epoch Models'),
}

# 'combined' is absent on purpose: it is configured by a list of weighted loss
# terms, which a settings group of scalars cannot hold.
_LOSS_FUNCTIONS: Sequence[str] = (
    'smooth_l1',
    'mse',
    'l1',
    'poisson_nll',
    'total_flux',
    'weighted',
    'probe_aware',
    'q_dependent',
)

# The weighted, probe-aware and q-dependent losses validate against the same pair.
_LOSS_TYPES: Sequence[str] = ('mse', 'mae')

_TRAINING_CHOICES: Mapping[str, Sequence[str]] = {
    'loss_function': _LOSS_FUNCTIONS,
    'weighted_loss.loss_type': _LOSS_TYPES,
    'probe_aware_loss.loss_type': _LOSS_TYPES,
    'q_dependent_loss.loss_type': _LOSS_TYPES,
}


def _training_policy() -> FlatteningPolicy:
    """The optimization branch, flattened across the three parameterized losses.

    ``mode`` is hidden: supervised and unsupervised are two reconstructors, so
    the objective follows from which one was selected rather than from a key.
    """
    from ptycho_fm.config import (
        LossConfig,
        ProbeAwareLossConfig,
        QDependentLossConfig,
        TrainingConfig,
    )

    return FlatteningPolicy(
        verbatim=True,
        enum_by_name=False,
        collapse_onto_prefix=frozenset(),
        hidden=frozenset(
            _hide_all_but(
                TrainingConfig,
                '',
                'batch_size',
                'learning_rate',
                'epochs',
                'loss_function',
                'weighted_loss',
                'probe_aware_loss',
                'q_dependent_loss',
                'save_epoch_models',
            )
            | _hide_all_but(LossConfig, 'weighted_loss.', 'loss_type', 'threshold', 'alpha')
            | _hide_all_but(
                ProbeAwareLossConfig,
                'probe_aware_loss.',
                'loss_type',
                'threshold',
                'alpha',
                'envelope_floor',
            )
            | _hide_all_but(
                QDependentLossConfig,
                'q_dependent_loss.',
                'loss_type',
                'threshold',
                'alpha',
                'q_beta',
                'q_floor',
                'log_error',
                'normalize_q',
            )
        ),
        prefixes={
            'weighted_loss': 'weighted_loss_',
            'probe_aware_loss': 'probe_aware_loss_',
            'q_dependent_loss': 'q_dependent_loss_',
        },
        # Each loss branch already says 'loss' in its prefix, so its own
        # loss_type is persisted as '<branch>_type' rather than doubling it.
        renames={
            'weighted_loss.loss_type': 'type',
            'probe_aware_loss.loss_type': 'type',
            'q_dependent_loss.loss_type': 'type',
        },
        labels=_TRAINING_LABELS,
        choices=_TRAINING_CHOICES,
    )


def _inference_policy() -> FlatteningPolicy:
    """The stitching branch, which is ptychodus's own and states its own text."""
    return FlatteningPolicy(verbatim=True, enum_by_name=False, collapse_onto_prefix=frozenset())


def sections() -> Sequence[OptionsSection]:
    """Return the four settings groups backing PtychoFM, one per declared branch."""
    return [
        OptionsSection('PtychoFMData', root_path='data', policy=_data_policy()),
        OptionsSection('PtychoFMModel', root_path='model', policy=_model_policy()),
        OptionsSection('PtychoFMTraining', root_path='training', policy=_training_policy()),
        OptionsSection('PtychoFMInference', root_path='inference', policy=_inference_policy()),
    ]
