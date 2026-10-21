"""Where each PtychoFM option is persisted in ptychodus's flat settings layout.

PtychoFM declares 165 fields and ptychodus persists 52, so this table is written
the other way up from pty-chi's: each policy names the subset it surfaces and
everything else is hidden. An upstream addition then stays out of the settings
file until someone chooses to put it there.

ptychodus persists PtychoFM keys as the literal field names, so the policy is
verbatim; the three nested branches contribute an underscore-suffixed prefix,
except the encoder branch, which is transparent -- which is how ``dropout`` and
``decoder_dropout`` coexist.

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
    'cache_object': Overlay('Cache Object In Memory'),
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
    'cache_memory_budget_mb': Overlay(
        'Array Cache Budget (MiB):',
        'Per-worker budget for cached object and probe arrays; 0 disables caching.',
        minimum=0,
    ),
    'train_split': Overlay(
        'Train Split:',
        'Fraction of dataset used for training; remainder is validation.',
        minimum=0.0,
        maximum=1.0,
    ),
    'random_seed': Overlay('Random Seed:'),
    'max_files': Overlay('Max Files (0 = no cap):', minimum=0),
    'num_workers': Overlay('Dataloader Workers:', minimum=0),
    'prefetch_factor': Overlay('Prefetch Factor:', minimum=0),
}


def _data_policy() -> FlatteningPolicy:
    """The dataset and dataloader branch.

    ``data_path`` is hidden with the rest: the child fills it from the training
    payload, so it is a path ptychodus passes rather than one a user sets.
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
        overlays=_DATA_OVERLAYS,
    )


# 'none' is how ptychodus spells the absent mode, which builds separate
# amplitude and phase decoders; a Sentinel turns it back into None.
_COUPLED_DECODER_MODES: Sequence[str] = ('none', 'real_imag', 'amp_phase')

_MODEL_OVERLAYS: Mapping[str, Overlay] = {
    'encoder_type': Overlay('Encoder Type:', choices=('custom', 'pretrained')),
    'coupled_decoder_mode': Overlay(
        'Coupled Decoder:',
        "'none' decodes amplitude and phase separately; the coupled modes share one "
        'decoder and run the forward physics. A checkpoint loads only into the mode '
        'it was trained with.',
        choices=_COUPLED_DECODER_MODES,
    ),
    'encoder.img_size': Overlay('Image Size:', minimum=32),
    'encoder.patch_size': Overlay('Patch Size:', minimum=1),
    'encoder.embed_dim': Overlay('Embed Dimension:', minimum=1),
    'encoder.depth': Overlay('Depth:', minimum=1),
    'encoder.num_heads': Overlay('Attention Heads:', minimum=1),
    'encoder.mlp_ratio': Overlay('MLP Ratio:', minimum=0.0),
    'encoder.use_cls_token': Overlay('Use CLS Token'),
    'encoder.dropout': Overlay('Dropout:', minimum=0.0, maximum=1.0),
    'encoder.attn_dropout': Overlay('Attention Dropout:', minimum=0.0, maximum=1.0),
    'encoder.timm_model_name': Overlay(
        'TIMM Model Name:', "Only used when encoder_type is 'pretrained'."
    ),
    'decoder.base_channels': Overlay('Decoder Base Channels:', minimum=1),
    'decoder.latent_dim': Overlay('Decoder Latent Dim:', minimum=1),
    'decoder.num_stages': Overlay('Decoder Upsample Stages:', minimum=1),
    'decoder.use_batchnorm': Overlay('Decoder Batch Norm'),
    'decoder.dropout': Overlay('Decoder Dropout:', minimum=0.0, maximum=1.0),
    'init.enabled': Overlay('Custom Weight Init'),
    'init.method': Overlay('Init Method:', choices=('trunc_normal', 'kaiming')),
}


def _model_policy() -> FlatteningPolicy:
    """The architecture branch, flattened across encoder, decoder and init."""
    from ptycho_fm.config import DecoderConfig, EncoderConfig, InitConfig, ModelConfig

    return FlatteningPolicy(
        naming=NamingConvention.VERBATIM,
        spelling=EnumSpelling.BY_VALUE,
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
        overlays=_MODEL_OVERLAYS,
    )


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

_TRAINING_OVERLAYS: Mapping[str, Overlay] = {
    'batch_size': Overlay('Batch Size:', minimum=1),
    'learning_rate': Overlay('Learning Rate:', minimum=0.0),
    'epochs': Overlay('Epochs:', minimum=1),
    'loss_function': Overlay('Loss Function:', choices=_LOSS_FUNCTIONS),
    'weighted_loss.loss_type': Overlay(
        'Weighted Loss Type:', "Only used when loss_function is 'weighted'.", choices=_LOSS_TYPES
    ),
    'weighted_loss.threshold': Overlay('Weighted Loss Threshold:'),
    'weighted_loss.alpha': Overlay('Weighted Loss Alpha:'),
    'probe_aware_loss.loss_type': Overlay(
        'Probe-Aware Loss Type:',
        "Only used when loss_function is 'probe_aware'.",
        choices=_LOSS_TYPES,
    ),
    'probe_aware_loss.threshold': Overlay('Probe-Aware Loss Threshold:'),
    'probe_aware_loss.alpha': Overlay(
        'Probe-Aware Loss Alpha:',
        'Power applied to the probe far-field envelope in the denominator.',
        minimum=0.0,
    ),
    'probe_aware_loss.envelope_floor': Overlay(
        'Probe-Aware Envelope Floor:',
        'Per-sample floor as a fraction of the mean probe envelope.',
        minimum=0.0,
    ),
    'q_dependent_loss.loss_type': Overlay(
        'Q-Dependent Loss Type:',
        "Only used when loss_function is 'q_dependent'.",
        choices=_LOSS_TYPES,
    ),
    'q_dependent_loss.alpha': Overlay(
        'Q-Dependent Loss Alpha:',
        'Focal exponent emphasizing poorly matched pixels; 0 disables it.',
    ),
    'q_dependent_loss.q_beta': Overlay(
        'Q-Dependent Q Exponent:',
        'Radial exponent emphasizing high-Q pixels; 0 disables Q weighting.',
    ),
    'q_dependent_loss.q_floor': Overlay(
        'Q-Dependent Q Floor:',
        'Minimum normalized Q, so low-Q pixels are not zero-weighted.',
        minimum=0.0,
    ),
    'q_dependent_loss.threshold': Overlay('Q-Dependent Loss Threshold:'),
    'q_dependent_loss.log_error': Overlay(
        'Q-Dependent Log Error', 'Compute the data error on log10 amplitudes.'
    ),
    'q_dependent_loss.normalize_q': Overlay(
        'Q-Dependent Normalize Q', 'Normalize the Q weight map to unit mean.'
    ),
    'save_epoch_models': Overlay('Save Per-Epoch Models'),
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
    """Return the four settings groups backing PtychoFM, one per declared branch."""
    return [
        OptionsSection('PtychoFMData', root_path='data', policy=_data_policy()),
        OptionsSection('PtychoFMModel', root_path='model', policy=_model_policy()),
        OptionsSection('PtychoFMTraining', root_path='training', policy=_training_policy()),
        OptionsSection('PtychoFMInference', root_path='inference', policy=_inference_policy()),
    ]
