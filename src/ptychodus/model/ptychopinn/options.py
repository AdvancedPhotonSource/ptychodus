"""What PtychoPINN accepts, declared for ptychodus to generate settings from.

Unlike pty-chi and PtychoFM, PtychoPINN's upstream configuration cannot be
declared directly. ``ptycho_torch.config_params`` is plain dataclasses, and
importing it pulls a thousand modules including torch and the TensorFlow-side
``ptycho`` package -- so reading a declaration from it would drag both into the
parent process at startup. This states the same parameters in a form that costs
nothing to import.

Being ptychodus's own, it carries its bounds, choices and display text inline
through ``pydantic.Field``, rather than needing them supplied alongside.

It describes only what ptychodus surfaces: 65 of the roughly 130 fields the
backend accepts. The rest keep whatever the backend defaults them to.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    'Accelerator',
    'AmplitudeActivation',
    'AuxiliaryLoss',
    'DataOptions',
    'DataScaling',
    'InferenceOptions',
    'LearningRateScheduler',
    'LossFunction',
    'ModelOptions',
    'NormalizationMode',
    'PatchWeighting',
    'PhysicsWeightSchedule',
    'PtychoPINNOptions',
    'TrainingOptions',
]


class NormalizationMode(StrEnum):
    GROUP = 'Group'
    BATCH = 'Batch'
    NONE = 'None'


class DataScaling(StrEnum):
    PARSEVAL = 'Parseval'
    MAX = 'Max'


class LossFunction(StrEnum):
    POISSON = 'Poisson'
    MAE = 'MAE'


class AmplitudeActivation(StrEnum):
    SILU = 'silu'
    SIGMOID = 'sigmoid'


class AuxiliaryLoss(StrEnum):
    NONE = 'None'
    TOTAL_VARIATION = 'Total_Variation'
    MEAN_DEVIATION = 'Mean_Deviation'


class LearningRateScheduler(StrEnum):
    DEFAULT = 'Default'
    EXPONENTIAL = 'Exponential'
    MULTI_STAGE = 'MultiStage'
    ADAPTIVE = 'Adaptive'
    WARMUP_COSINE = 'WarmupCosine'
    REDUCE_LR_ON_PLATEAU = 'ReduceLROnPlateau'


class PhysicsWeightSchedule(StrEnum):
    LINEAR = 'linear'
    COSINE = 'cosine'
    EXPONENTIAL = 'exponential'


class PatchWeighting(StrEnum):
    PROBE = 'probe'
    UNIFORM = 'uniform'


class Accelerator(StrEnum):
    AUTO = 'auto'
    CPU = 'cpu'
    GPU = 'gpu'
    CUDA = 'cuda'
    MPS = 'mps'


class _Options(BaseModel):
    # Several fields are named model_*, which pydantic otherwise reserves.
    model_config = ConfigDict(protected_namespaces=(), validate_assignment=True)


class DataOptions(_Options):
    model_size: int = Field(128, ge=1, title='Model Size:')
    gridsize: int = Field(
        2,
        ge=1,
        title='Grid Size:',
        description='Square group side length; each group holds Grid Size squared patterns.',
    )
    neighbor_count: int = Field(
        6,
        ge=1,
        title='Nearest Neighbors for Lookup:',
        description='Neighbor search width used to build each group. Raised automatically '
        'if it is smaller than the group size.',
    )
    group_padding_step: float = Field(
        3.0,
        ge=0.0,
        title='Group Padding Step [px]:',
        description='Sizes the grouped object canvas; does not affect neighbor selection.',
    )
    data_normalization_mode: NormalizationMode = Field(NormalizationMode.BATCH, title='Normalize:')
    normalize_probe: bool = Field(True, title='Normalize Probe')
    x_lower_bound: float = Field(0.1, ge=0.0, le=1.0, title='X Lower Bound:')
    x_upper_bound: float = Field(0.9, ge=0.0, le=1.0, title='X Upper Bound:')
    y_lower_bound: float = Field(0.1, ge=0.0, le=1.0, title='Y Lower Bound:')
    y_upper_bound: float = Field(0.9, ge=0.0, le=1.0, title='Y Upper Bound:')
    probe_scale: float = Field(4.0, ge=0.0, title='Probe Scale:')
    data_scaling_method: DataScaling = Field(DataScaling.PARSEVAL, title='Data Scaling:')
    subtract_mean_phase: bool = Field(
        True, title='Subtract Phase', description='Only useful for supervised training dataset.'
    )


class ModelOptions(_Options):
    object_big: bool = Field(True, title='Object Big')
    probe_big: bool = Field(True, title='Probe Big')
    loss_function: LossFunction = Field(LossFunction.POISSON, title='Loss Function:')
    amplitude_activation_function: AmplitudeActivation = Field(
        AmplitudeActivation.SILU, title='Amplitude Activation:'
    )
    cbam_encoder: bool = Field(
        True,
        title='CBAM Encoder',
        description='Whether Convolutional Block Attention Module (CBAM) is turned on for encoder.',
    )
    use_shared_decoder: bool = Field(False, title='Shared Decoder')
    intensity_scale_trainable: bool = Field(False, title='Intensity Scale Trainable')
    intensity_scale: float = Field(10000.0, ge=0.0, title='Intensity Scale:')
    max_position_jitter: int = Field(10, ge=0, title='Max Position Jitter:')
    num_datasets: int = Field(1, ge=1, title='Datasets:')
    auxiliary_amplitude_loss: AuxiliaryLoss = Field(
        AuxiliaryLoss.NONE, title='Auxiliary Amplitude Loss Function:'
    )
    auxiliary_phase_loss: AuxiliaryLoss = Field(
        AuxiliaryLoss.NONE, title='Auxiliary Phase Loss Function:'
    )
    auxiliary_amplitude_loss_coeff: float = Field(
        1.0, ge=0.0, title='Auxiliary Amplitude Loss Coefficient:'
    )
    auxiliary_phase_loss_coeff: float = Field(
        1.0, ge=0.0, title='Auxiliary Phase Loss Coefficient:'
    )
    num_filters_scale: int = Field(
        2,
        ge=1,
        title='Num Filters Scale:',
        description='Shrinking factor for channels in network layers.',
    )
    eca_decoder: bool = Field(
        False,
        title='ECA Decoder',
        description='Whether Efficient Channel Attention (ECA) is turned on for decoder.',
    )
    use_batch_normalization: bool = Field(
        False, title='Use Batch Normalization', description='Whether to use batch normalization'
    )
    edge_pad: int = Field(
        10,
        ge=0,
        title='Edge Padding:',
        description='For padding the decoder_last reconstruction.',
    )
    decoder_last_c_outer_fraction: float = Field(
        0.125,
        ge=0.0,
        le=1.0,
        title='Decoder Last C Outer Fraction:',
        description='Amount of channels going to higher frequency components in decoder_last',
    )
    cbam_bottleneck: bool = Field(False, title='CBAM Bottleneck')
    cbam_decoder: bool = Field(
        False,
        title='CBAM Decoder',
        description='Whether Convolutional Block Attention Module (CBAM) is turned on for decoder.',
    )
    spatial_decoder: bool = Field(
        False,
        title='Decoder Spatial Attention',
        description='Whether Spatial Attention is turned on for decoder.',
    )
    decoder_spatial_kernel: int = Field(
        7,
        ge=1,
        title='Decoder Spatial Atttention Kernel:',
        description='Spatial attention kernel for decoder.',
    )
    eca_encoder: bool = Field(
        False,
        title='ECA Encoder',
        description='Whether Efficient Channel Attention (ECA) is turned on for encoder.',
    )
    offset: int = Field(
        6, ge=0, title='Offset:', description='Offset parameter for nearest neighbor patches.'
    )


class TrainingOptions(_Options):
    epochs: int = Field(50, ge=1, title='Default Epochs:')
    batch_size: int = Field(16, ge=1, title='Batch Size:')
    learning_rate: float = Field(1e-3, ge=0.0, title='Learning Rate:')
    training_groups: int = Field(
        512,
        ge=1,
        title='Training Groups:',
        description='Number of grouped samples drawn from the training data. Grouping never '
        'reuses a center, so the achievable count is capped by the dataset.',
    )
    num_dataloader_workers: int = Field(4, ge=0, title='Dataloader Workers:')
    gradient_accumulation_steps: int = Field(1, ge=1, title='Gradient Accumulation Steps:')
    epochs_finetune: int = Field(0, ge=0, title='Fine Tune Epochs:')
    finetune_gamma: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        title='Fine Tune Gamma:',
        description='Scales base learning rate for fine-tuning.',
    )
    gradient_clip_val: float = Field(0.0, ge=0.0, title='Gradient Clip Value:')
    accelerator: Accelerator = Field(
        Accelerator.AUTO,
        title='Accelerator:',
        description='Hardware to train on. "auto" prefers CUDA when a device is visible and '
        'otherwise falls back to the CPU.',
    )
    # Removed once device selection moves to the spawn layer.
    visible_gpu_indices: str = Field(
        '',
        title='Visible GPU Indices:',
        description='Comma-separated CUDA device indices exposed to the training subprocess '
        '(via CUDA_VISIBLE_DEVICES).',
    )
    learning_rate_scheduler: LearningRateScheduler = Field(
        LearningRateScheduler.DEFAULT, title='Learning Rate Scheduler:'
    )
    learning_rate_warmup_epochs: int = Field(
        0,
        ge=0,
        title='Warmup Epochs:',
        description='warmup epochs for WarmupCosine scheduler',
    )
    minimum_learning_rate_ratio: float = Field(
        0.1,
        ge=0.0,
        title='Min Learning Rate Ratio:',
        description='Minimum learning rate ratio for WarmupCosine scheduler',
    )
    notes: str = Field('', title='MLflow Notes:')
    model_name: str = Field('PtychoPINNv2', title='MLflow Model Name:')
    stage_1_epochs: int = Field(
        0,
        ge=0,
        title='Stage 1 Epochs:',
        description='Multi-stage training. Zero disables the stage.',
    )
    stage_2_epochs: int = Field(
        0,
        ge=0,
        title='Stage 2 Epochs:',
        description='Weighted transition stage. Zero disables the stage.',
    )
    stage_3_epochs: int = Field(
        0, ge=0, title='Stage 3 Epochs:', description='Physics-only stage. Zero disables the stage.'
    )
    stage_3_lr_factor: float = Field(
        0.1,
        ge=0.0,
        title='Stage 3 LR Factor:',
        description='Learning-rate reduction applied during the physics-only stage.',
    )
    physics_weight_schedule: PhysicsWeightSchedule = Field(
        PhysicsWeightSchedule.COSINE, title='Physics Weight Schedule:'
    )


class InferenceOptions(_Options):
    batch_size: int = Field(
        1000, ge=1, title='Batch Size:', description='Batch size for reconstruction.'
    )
    middle_trim: int = Field(32, ge=0, title='Middle Trim:')
    experiment_number: int = Field(0, ge=0, title='Experiment Number:')
    patch_weighting_method: PatchWeighting = Field(
        PatchWeighting.PROBE, title='Patch Weighting Method:'
    )
    pad_eval: bool = Field(
        True,
        title='Pad Evaluation Edges',
        description='Pads the evaluation edges, enforced during training for Nyquist frequency.',
    )
    window: int = Field(
        20,
        ge=0,
        title='Window Padding:',
        description='Window padding around reconstruction due to edge errors.',
    )


class PtychoPINNOptions(_Options):
    """Everything ptychodus exposes for PtychoPINN, in the shape its panels use."""

    data: DataOptions = Field(default_factory=DataOptions)
    model: ModelOptions = Field(default_factory=ModelOptions)
    training: TrainingOptions = Field(default_factory=TrainingOptions)
    inference: InferenceOptions = Field(default_factory=InferenceOptions)
