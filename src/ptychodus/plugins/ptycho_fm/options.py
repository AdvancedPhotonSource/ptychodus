"""What PtychoFM accepts, declared for ptychodus to generate settings from.

PtychoFM declares its own configuration as pydantic, so this is an overlay
rather than a restatement: :func:`default_options` returns upstream's own
``PtychoFMConfig`` carrying the defaults ptychodus has chosen, and the section
table beside the flattener says which of its 165 fields are surfaced and what
each is called.

Importing ``ptycho_fm.config`` costs exactly two modules -- the package root is
empty and the config module imports only pydantic -- so the parent can read the
declaration without pulling torch in. That is what makes the overlay possible
here and not for PtychoPINN, whose configuration is inseparable from its
backend.

**Absent means absent.** Upstream resolves pretrained encoder sizes, validation
batch size and dataset normalization from a field having no value at all, so the
config travels as ``model_dump(exclude_unset=True)`` and only the fields
ptychodus surfaces or freezes may be set. Every default below is therefore passed
explicitly, including the ones that agree with upstream's: a field left to its
declared default would be dropped from the dump and resolved at runtime instead.

The architecture is **frozen here rather than surfaced**. A ``.pth`` holds a bare
state dict naming no architecture, so a checkpoint loads only into the model it
was trained with; offering the sizes as settings invited a mismatch that only
shows up as a failed load. The ``model`` branch below is therefore the single
statement of what every run builds, and nothing in the settings file can reach
it. Changing a value here changes which checkpoints this build can load.
"""

from __future__ import annotations

from typing import TypeVar

from ptycho_fm.config import (
    DataConfig,
    DecoderConfig,
    EncoderConfig,
    InitConfig,
    ModelConfig,
    PtychoFMConfig,
    TrainingConfig,
)
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    'InferenceOptions',
    'PtychoFMOptions',
    'default_options',
]


class InferenceOptions(BaseModel):
    """How a predicted patch is cropped, shifted and batched back into an object.

    ptychodus's own: PtychoFM has no inference section, because stitching is
    what the caller does with the model's output rather than part of the model.
    Being ptychodus's, it states its bounds and display text inline.
    """

    model_config = ConfigDict(validate_assignment=True)

    central_crop: int = Field(
        64,
        ge=1,
        title='Central Crop:',
        description='Pixels cropped from each border of every patch before stitching.',
    )
    pad: int = Field(32, ge=0, title='Fourier Shift Pad:')
    batch_size: int = Field(256, ge=1, title='Batch Size:')


class PtychoFMOptions(PtychoFMConfig):
    """Upstream's configuration plus the inference branch ptychodus adds.

    A subclass rather than a wrapper, so the object stays a ``PtychoFMConfig``
    and dumps to the same nested shape the child already reads. Upstream allows
    extra keys, so the added branch rides along as a top-level section.
    """

    inference: InferenceOptions = Field(default_factory=InferenceOptions)


_ModelT = TypeVar('_ModelT', bound=BaseModel)


def _fully_set(options: _ModelT) -> _ModelT:
    """Return *options* with every field marked as explicitly set.

    For a branch ptychodus owns, which has no absent-means-resolve-at-runtime
    semantics to preserve: rebuilding it from its own dump marks each field set
    without restating any default a second time.
    """
    return type(options)(**options.model_dump())


def default_options() -> PtychoFMOptions:
    """Return the configuration ptychodus starts a PtychoFM run from."""
    return PtychoFMOptions(
        data=DataConfig(
            scale=10000.0,
            default_normalization=10000.0,
            cache_object=False,
            max_probe_modes=10,
            max_OPR_modes=1,
            cache_memory_budget_mb=512,
            train_split=0.95,
            random_seed=8,
            # None is upstream's "no cap"; ptychodus persists it as 0.
            max_files=None,
            num_workers=4,
            prefetch_factor=2,
        ),
        model=ModelConfig(
            encoder_type='custom',
            # Selects the model class, so a checkpoint loads only into the mode
            # that produced it. None builds separate amplitude and phase
            # decoders instead, sharing no decoder and running no forward
            # physics.
            coupled_decoder_mode='amp_phase',
            encoder=EncoderConfig(
                img_size=256,
                patch_size=16,
                embed_dim=512,
                depth=12,
                num_heads=8,
                mlp_ratio=4.0,
                use_cls_token=False,
                dropout=0.1,
                attn_dropout=0.0,
                timm_model_name='vit_large_patch32_224',
            ),
            decoder=DecoderConfig(
                base_channels=64,
                latent_dim=512,
                num_stages=4,
                use_batchnorm=True,
                dropout=0.1,
            ),
            init=InitConfig(enabled=False, method='trunc_normal'),
        ),
        training=TrainingConfig(
            batch_size=128,
            learning_rate=1.0e-5,
            epochs=2,
            loss_function='mse',
            save_epoch_models=True,
        ),
        inference=_fully_set(InferenceOptions()),
    )
