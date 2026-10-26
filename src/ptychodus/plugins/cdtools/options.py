"""What cdtools accepts, declared for ptychodus to generate settings from.

The declaration is ptychodus's own -- cdtools configures a reconstruction through
constructor and ``optimize`` keyword arguments rather than a schema -- so it carries its
bounds, choices and display text inline through ``pydantic.Field``.

A run is one optimizer over up to three stages, each with its own epoch count, learning
rate and batch size, so a coarse-then-fine recipe needs no second run. A learning rate
suited to one optimizer is far from right for another: Adam's are of order 1e-3, LBFGS's
of order 1e-1, and SGD's depend on the photon count.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    'CDToolsDevice',
    'CDToolsLineSearch',
    'CDToolsLoss',
    'CDToolsOptimizer',
    'CDToolsOptions',
    'FancyPtychoModelOptions',
    'OptimizerOptions',
    'RefinementOptions',
    'ScheduleOptions',
    'StageOptions',
    'default_options',
]


class CDToolsOptimizer(StrEnum):
    ADAM = 'Adam'
    LBFGS = 'LBFGS'
    SGD = 'SGD'


class CDToolsLineSearch(StrEnum):
    NONE = 'none'
    STRONG_WOLFE = 'strong_wolfe'


class CDToolsLoss(StrEnum):
    AMPLITUDE_MSE = 'amplitude_mse'
    POISSON_NLL = 'poisson_nll'
    INTENSITY_MSE = 'intensity_mse'


class CDToolsDevice(StrEnum):
    AUTO = 'auto'
    CPU = 'cpu'
    CUDA = 'cuda'


class OptimizerOptions(BaseModel):
    """Which optimizer runs every stage, and the hyperparameters a stage does not vary."""

    model_config = ConfigDict(validate_assignment=True)

    optimizer: CDToolsOptimizer = Field(
        CDToolsOptimizer.ADAM,
        title='Optimizer:',
        description='The optimizer every stage runs. Adam suits most data; LBFGS can be '
        'much faster near convergence but is less stable; SGD is mainly for comparison.',
    )
    adam_beta1: float = Field(
        0.9,
        ge=0.0,
        le=0.999999,
        title='Adam Beta 1:',
        description='Decay rate of the running mean of the gradient.',
    )
    adam_beta2: float = Field(
        0.999,
        ge=0.0,
        le=0.999999,
        title='Adam Beta 2:',
        description='Decay rate of the running mean of the squared gradient.',
    )
    adam_amsgrad: bool = Field(False, title='Adam AMSGrad', description='Use the AMSGrad variant.')
    adam_schedule: bool = Field(
        False,
        title='Adam Reduce on Plateau',
        description='Cut the learning rate fivefold whenever the loss stops improving.',
    )
    sgd_momentum: float = Field(0.0, ge=0.0, le=1.0, title='SGD Momentum:')
    sgd_dampening: float = Field(0.0, ge=0.0, le=1.0, title='SGD Dampening:')
    sgd_weight_decay: float = Field(0.0, ge=0.0, le=1.0, title='SGD Weight Decay:')
    sgd_nesterov: bool = Field(False, title='SGD Nesterov Momentum')
    lbfgs_history_size: int = Field(
        2,
        ge=1,
        le=100,
        title='LBFGS History Size:',
        description='Update steps the inverse-Hessian estimate remembers.',
    )
    lbfgs_line_search: CDToolsLineSearch = Field(
        CDToolsLineSearch.NONE,
        title='LBFGS Line Search:',
        description='Strong Wolfe line search costs extra evaluations per step but is more '
        'robust to a learning rate that is too large.',
    )
    shuffle: bool = Field(
        True,
        title='Shuffle Patterns',
        description='Visit the patterns in a new random order every epoch. LBFGS always '
        'takes the whole dataset at once.',
    )
    calculation_width: int = Field(
        10,
        ge=1,
        le=100000,
        title='Calculation Width:',
        description='Patterns simulated together while accumulating a batch gradient. '
        'Affects speed and memory only, not the result.',
    )
    num_sync_epochs: int = Field(
        5,
        ge=1,
        le=100000,
        title='Epochs Between Updates:',
        description='How often the current probe, object and positions are sent back to '
        'ptychodus. The end of every stage is always sent.',
    )
    device: CDToolsDevice = Field(
        CDToolsDevice.AUTO,
        title='Device:',
        description='Where the reconstruction runs. Auto takes a CUDA device when one is '
        'visible and the CPU otherwise.',
    )


class StageOptions(BaseModel):
    """One stage of the schedule."""

    model_config = ConfigDict(validate_assignment=True)

    epochs: int = Field(
        0,
        ge=0,
        le=1000000,
        title='Epochs:',
        description='Passes over the dataset in this stage. Zero skips the stage.',
    )
    learning_rate: float = Field(
        0.005,
        ge=0.0,
        le=100.0,
        title='Learning Rate:',
    )
    batch_size: int = Field(
        10,
        ge=1,
        le=1000000,
        title='Batch Size:',
        description='Patterns per optimizer step. LBFGS ignores it and takes every pattern.',
    )


class ScheduleOptions(BaseModel):
    """Up to three stages run in order by the same optimizer.

    The defaults are a coarse-to-fine Adam recipe: small batches at a high learning rate
    to find the structure, then larger batches at lower rates to refine it.
    """

    model_config = ConfigDict(validate_assignment=True)

    stage1: StageOptions = Field(
        default_factory=lambda: StageOptions(epochs=20, learning_rate=0.005, batch_size=10),
        title='Stage 1',
    )
    stage2: StageOptions = Field(
        default_factory=lambda: StageOptions(epochs=50, learning_rate=0.002, batch_size=50),
        title='Stage 2',
    )
    stage3: StageOptions = Field(
        default_factory=lambda: StageOptions(epochs=50, learning_rate=0.001, batch_size=100),
        title='Stage 3',
    )

    def stages(self) -> list[StageOptions]:
        """The stages that run, in order: those with at least one epoch."""
        return [stage for stage in (self.stage1, self.stage2, self.stage3) if stage.epochs > 0]


class RefinementOptions(BaseModel):
    """What the optimizer may change, and which measured detector maps the model uses."""

    model_config = ConfigDict(validate_assignment=True)

    refine_probe: bool = Field(True, title='Refine Probe')
    refine_object: bool = Field(True, title='Refine Object')
    refine_positions: bool = Field(
        False,
        title='Refine Positions',
        description='Optimize a per-position offset. A position may only move within the '
        "object canvas, so leave room with the object builder's extra padding.",
    )
    translation_scale: float = Field(
        1.0,
        ge=1.0e-6,
        le=1000.0,
        title='Translation Scale:',
        description='Pixels a unit position offset moves, which scales how fast positions '
        'learn relative to the probe and object.',
    )
    use_background: bool = Field(
        True,
        title='Model the Measured Background',
        description='Start the additive background term from the background the data '
        'carries, when it carries one, rather than from almost nothing.',
    )
    refine_background: bool = Field(
        True,
        title='Refine Background',
        description='Optimize the additive detector background.',
    )
    use_flat_field: bool = Field(
        True,
        title='Model the Measured Flat Field',
        description='Multiply the model by the flat field the data carries, when it '
        'carries one. The flat field is never refined.',
    )


class FancyPtychoModelOptions(BaseModel):
    """How the FancyPtycho forward model is built."""

    model_config = ConfigDict(validate_assignment=True)

    loss: CDToolsLoss = Field(
        CDToolsLoss.AMPLITUDE_MSE,
        title='Loss:',
        description='Amplitude MSE is robust for most data; Poisson NLL is the maximum '
        'likelihood estimate for photon-counting data.',
    )
    fourier_probe: bool = Field(
        False,
        title='Fourier-Space Probe',
        description='Represent and refine the probe in the detector plane.',
    )
    exponentiate_obj: bool = Field(
        False,
        title='Exponentiate Object',
        description='Refine the complex phase delay and take the transmission as its '
        'exponential, which keeps strongly absorbing regions well conditioned.',
    )
    phase_only: bool = Field(
        False,
        title='Phase-Only Object',
        description='Hold the object at unit transmission and refine only its phase. '
        'Implies an exponentiated object, the representation it is defined in.',
    )
    probe_support_radius_px: float = Field(
        0.0,
        ge=0.0,
        le=100000.0,
        title='Probe Support Radius [px]:',
        description='Zero the probe outside a centered disk of this radius. Zero applies '
        'no support.',
    )


class CDToolsOptions(BaseModel):
    """Everything ptychodus exposes for a cdtools FancyPtycho reconstruction."""

    model_config = ConfigDict(validate_assignment=True)

    optimizer: OptimizerOptions = Field(default_factory=OptimizerOptions)
    schedule: ScheduleOptions = Field(default_factory=ScheduleOptions)
    refinement: RefinementOptions = Field(default_factory=RefinementOptions)
    model: FancyPtychoModelOptions = Field(default_factory=FancyPtychoModelOptions)


def default_options() -> CDToolsOptions:
    """Return the configuration ptychodus starts a cdtools run from."""
    return CDToolsOptions()
