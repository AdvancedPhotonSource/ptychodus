"""Which pty-chi algorithms exist, and what each one declares.

An algorithm is the six ``*Options`` classes pty-chi subclasses per engine,
named together in a :class:`_Spec`. That is the whole of the per-algorithm
difference: the options object a reconstructor declares is an instance of its
own ``task_options_cls``, and the settings behind it are generated from what
that object states. Adding an engine is one subclass plus one entry in
:data:`_ALGORITHM_CLASSES`.

The module imports ``ptychi.api`` (for the Options dataclasses), which pulls
torch in for type annotations only -- no CUDA runtime is acquired. That
happens when ``PtychographyTask`` is instantiated in :mod:`.task`, inside
:meth:`PtyChiReconstructor.reconstruct`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Final
import logging

from ptychi.api import (
    AutodiffPtychographyOPRModeWeightsOptions,
    AutodiffPtychographyObjectOptions,
    AutodiffPtychographyOptions,
    AutodiffPtychographyProbeOptions,
    AutodiffPtychographyProbePositionOptions,
    AutodiffPtychographyReconstructorOptions,
    BHOPRModeWeightsOptions,
    BHObjectOptions,
    BHOptions,
    BHProbeOptions,
    BHProbePositionOptions,
    BHReconstructorOptions,
    DMOPRModeWeightsOptions,
    DMObjectOptions,
    DMOptions,
    DMProbeOptions,
    DMProbePositionOptions,
    DMReconstructorOptions,
    EPIEOptions,
    EPIEReconstructorOptions,
    LSQMLOPRModeWeightsOptions,
    LSQMLObjectOptions,
    LSQMLOptions,
    LSQMLProbeOptions,
    LSQMLProbePositionOptions,
    LSQMLReconstructorOptions,
    PIEOPRModeWeightsOptions,
    PIEObjectOptions,
    PIEOptions,
    PIEProbeOptions,
    PIEProbePositionOptions,
    PIEReconstructorOptions,
    RAAROPRModeWeightsOptions,
    RAARObjectOptions,
    RAAROptions,
    RAARProbeOptions,
    RAARProbePositionOptions,
    RAARReconstructorOptions,
    RPIEOptions,
    RPIEReconstructorOptions,
)
from ptychi.api.options.task import PtychographyTaskOptions
from pydantic import BaseModel

from ptychodus.api.reconstruct import ReconstructInput, ReconstructOutput, Reconstructor

from .options import PtyChiOptions, apply_to_task, default_options
from .task import (
    align_task_options_with_product,
    dump_task_options_yaml,
    reconstruct_with_ptychi,
)

__all__ = [
    'PtyChiAlgorithm',
    'PtyChiReconstructor',
    'build_reconstructors',
    'task_options_classes',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Spec:
    """Names the six pty-chi ``*Options`` classes plus the algorithm's display name."""

    display_name: str
    task_options_cls: type[PtychographyTaskOptions]
    reconstructor_options_cls: type
    object_options_cls: type
    probe_options_cls: type
    probe_position_options_cls: type
    opr_options_cls: type


class PtyChiAlgorithm:
    """One pty-chi engine, identified by the options classes it is built from.

    Carries no values. An options object arrives per call, already holding
    whatever the settings layer has in it, and the only thing left to do with
    it is reconcile it with the product being reconstructed.
    """

    spec: ClassVar[_Spec]


class DMAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        'DM',
        DMOptions,
        DMReconstructorOptions,
        DMObjectOptions,
        DMProbeOptions,
        DMProbePositionOptions,
        DMOPRModeWeightsOptions,
    )


class RAARAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        'RAAR',
        RAAROptions,
        RAARReconstructorOptions,
        RAARObjectOptions,
        RAARProbeOptions,
        RAARProbePositionOptions,
        RAAROPRModeWeightsOptions,
    )


class PIEAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        'PIE',
        PIEOptions,
        PIEReconstructorOptions,
        PIEObjectOptions,
        PIEProbeOptions,
        PIEProbePositionOptions,
        PIEOPRModeWeightsOptions,
    )


class EPIEAlgorithm(PIEAlgorithm):
    spec = _Spec(
        'ePIE',
        EPIEOptions,
        EPIEReconstructorOptions,
        PIEObjectOptions,
        PIEProbeOptions,
        PIEProbePositionOptions,
        PIEOPRModeWeightsOptions,
    )


class RPIEAlgorithm(PIEAlgorithm):
    spec = _Spec(
        'rPIE',
        RPIEOptions,
        RPIEReconstructorOptions,
        PIEObjectOptions,
        PIEProbeOptions,
        PIEProbePositionOptions,
        PIEOPRModeWeightsOptions,
    )


class LSQMLAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        'LSQML',
        LSQMLOptions,
        LSQMLReconstructorOptions,
        LSQMLObjectOptions,
        LSQMLProbeOptions,
        LSQMLProbePositionOptions,
        LSQMLOPRModeWeightsOptions,
    )


class AutodiffAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        'Autodiff',
        AutodiffPtychographyOptions,
        AutodiffPtychographyReconstructorOptions,
        AutodiffPtychographyObjectOptions,
        AutodiffPtychographyProbeOptions,
        AutodiffPtychographyProbePositionOptions,
        AutodiffPtychographyOPRModeWeightsOptions,
    )


class BHAlgorithm(PtyChiAlgorithm):
    spec = _Spec(
        'BH',
        BHOptions,
        BHReconstructorOptions,
        BHObjectOptions,
        BHProbeOptions,
        BHProbePositionOptions,
        BHOPRModeWeightsOptions,
    )


# Every algorithm, in the order the GUI lists them.
_ALGORITHM_CLASSES: Final[tuple[type[PtyChiAlgorithm], ...]] = (
    DMAlgorithm,
    RAARAlgorithm,
    PIEAlgorithm,
    EPIEAlgorithm,
    RPIEAlgorithm,
    LSQMLAlgorithm,
    AutodiffAlgorithm,
    BHAlgorithm,
)


def task_options_classes() -> dict[str, type[PtychographyTaskOptions]]:
    """Each algorithm's pty-chi task-options class, keyed by display name.

    The options class is what distinguishes one algorithm from another on the
    wire: it stamps ``options_class_name`` into the serialized payload, which is
    what :func:`~ptychodus.plugins.ptychi.task.load_task_options` reads back to
    choose the subclass.
    """
    return {cls.spec.display_name: cls.spec.task_options_cls for cls in _ALGORITHM_CLASSES}


def _declared(options: BaseModel) -> PtyChiOptions:
    """Narrow the options a call was given to what this backend declares.

    Checked rather than asserted: the dispatching layer validates the type
    against what ``get_default_options`` returns, so a mismatch here means
    those two have drifted apart, and saying so beats an attribute error
    several frames later.
    """
    if not isinstance(options, PtyChiOptions):
        raise TypeError(f'Expected PtyChiOptions, got {type(options).__name__}.')

    return options


def _resolve_default_device(task_options: PtychographyTaskOptions) -> None:
    """Keep the device the options name, unless they name a GPU and none is visible here.

    Settled where the reconstruction runs, because that is the process whose
    devices count. ptychodus restricts that process to the devices a run
    selected, so a selection naming only the CPU leaves no GPU visible and the
    options fall back to the CPU. An explicit CPU is never promoted to a GPU.
    """
    import torch

    from ptychi.api.enums import Devices

    device = task_options.reconstructor_options.default_device

    if device != Devices.CPU:
        visible = torch.cuda.is_available() or (
            getattr(torch, 'xpu', None) is not None and torch.xpu.is_available()
        )

        if not visible:
            logger.info('No GPU is visible; falling back to the CPU.')
            device = Devices.CPU

    task_options.reconstructor_options.default_device = device
    logger.info(f'Reconstructing on the {device.name}.')


class PtyChiReconstructor(Reconstructor):
    """One pty-chi algorithm, holding nothing but the class that names it."""

    def __init__(self, algorithm: type[PtyChiAlgorithm]) -> None:
        super().__init__()
        self._algorithm = algorithm

    @property
    def name(self) -> str:
        return self._algorithm.spec.display_name

    def get_default_options(self) -> BaseModel:
        return default_options(self._algorithm.spec.task_options_cls)

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        declared = _declared(options)
        # Every field that must agree with the product rather than with a setting
        # is applied here, in one place, rather than being read off the product
        # wherever it happens to be needed.
        task_options = align_task_options_with_product(apply_to_task(declared), input_.product)
        _resolve_default_device(task_options)
        task_options.check()
        logger.info('Aligned pty-chi task options:\n%s', dump_task_options_yaml(task_options))
        yield from reconstruct_with_ptychi(input_, task_options, declared.num_sync_epochs)


def build_reconstructors() -> list[PtyChiReconstructor]:
    """One reconstructor per algorithm, in the order the GUI lists them."""
    return [PtyChiReconstructor(cls) for cls in _ALGORITHM_CLASSES]
