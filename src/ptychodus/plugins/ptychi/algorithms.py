"""Which pty-chi algorithms exist, and what each one declares.

An algorithm is the six ``*Options`` classes pty-chi subclasses per engine,
named together in a :class:`_Spec`. That is the whole of the per-algorithm
difference: the options object a reconstructor declares is an instance of its
own ``task_options_cls``, and the settings behind it are generated from what
that object states. Adding an engine is one subclass plus one entry in
:data:`_ALGORITHM_CLASSES`.

The module imports ``ptychi.api`` (for the Options dataclasses), which pulls
torch in for type annotations only -- no CUDA runtime is acquired. That
happens child-side when ``PtychographyTask`` is instantiated in :mod:`.task`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

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

from ptychodus.api.reconstruct import ReconstructInput, SubprocessReconstructor

from ._payload import PtyChiPayload
from .options import PtyChiOptions, apply_to_task, default_options
from .task import align_task_options_with_product

__all__ = [
    'PtyChiAlgorithm',
    'build_reconstructors',
    'task_options_classes',
]


_RECONSTRUCT_ENTRY = 'ptychodus.plugins.ptychi._subprocess:run_reconstruct'


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


def _wrap_as_subprocess_reconstructor(
    algorithm: type[PtyChiAlgorithm],
) -> SubprocessReconstructor:
    task_options_cls = algorithm.spec.task_options_cls

    def build_payload(
        parameters: ReconstructInput,
        _loaded_model_path: Path | None,
        options: BaseModel,
    ) -> PtyChiPayload:
        declared = _declared(options)
        return PtyChiPayload(
            reconstruct_input=parameters,
            # Every field that must agree with the product rather than with a
            # setting is applied here, in one place, rather than being read
            # off the product wherever it happens to be needed.
            task_options=align_task_options_with_product(
                apply_to_task(declared), parameters.product
            ),
            num_sync_epochs=declared.num_sync_epochs,
        )

    def build_default_options() -> BaseModel:
        return default_options(task_options_cls)

    return SubprocessReconstructor(
        name=algorithm.spec.display_name,
        reconstruct_entry_point=_RECONSTRUCT_ENTRY,
        default_options=build_default_options,
        build_reconstruct_payload=build_payload,
    )


def build_reconstructors() -> list[SubprocessReconstructor]:
    """One reconstructor per algorithm, in the order the GUI lists them."""
    return [_wrap_as_subprocess_reconstructor(cls) for cls in _ALGORITHM_CLASSES]
