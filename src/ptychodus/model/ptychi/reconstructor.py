"""Parent-side factory that builds :class:`SubprocessReconstructor`s for PtyChi.

This module imports ``ptychi.api`` (via the per-algorithm modules and
:mod:`.helper`) to construct the ``*Options`` dataclasses that make up a
``PtychographyTaskOptions``. That import chain pulls torch in — but torch's
CUDA runtime is lazy, so no GPU context is acquired here. The context is
acquired only in the spawned child when it instantiates ``PtychographyTask``
(see :mod:`._subprocess`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptychodus.api.reconstructor import ReconstructInput

from ..diffraction import PatternSizer
from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import PtyChiPayload
from .autodiff import AutodiffReconstructor
from .bh import BHReconstructor
from .dm import DMReconstructor
from .epie import EPIEReconstructor
from .helper import PtyChiOptionsHelper
from .lsqml import LSQMLReconstructor
from .pie import PIEReconstructor
from .rpie import RPIEReconstructor
from .settings import (
    PtyChiAutodiffSettings,
    PtyChiBHSettings,
    PtyChiDMSettings,
    PtyChiLSQMLSettings,
    PtyChiOPRSettings,
    PtyChiObjectSettings,
    PtyChiPIESettings,
    PtyChiProbePositionSettings,
    PtyChiProbeSettings,
    PtyChiSettings,
)

__all__ = ['build_reconstructor_list']


_RECONSTRUCT_ENTRY = 'ptychodus.model.ptychi._subprocess:run_reconstruct'


# Callable that turns a ReconstructInput into a fully-built pty-chi
# ``PtychographyTaskOptions`` (algorithm-specific subclass). ``Any`` because the
# concrete return type lives in ``ptychi.api.options.task`` and typing it here
# would drag that module into non-ptychi environments.
_TaskOptionsBuilder = Callable[[ReconstructInput], Any]


@dataclass(frozen=True)
class _AlgorithmSpec:
    """One row of the algorithm dispatch table used by the factory."""

    display_name: str
    make_option_factory: Callable[
        [PtyChiOptionsHelper, 'PtyChiSettingsBundle'], _TaskOptionsBuilder
    ]


@dataclass(frozen=True)
class PtyChiSettingsBundle:
    """The algorithm-specific settings groups held by :class:`PtyChiReconstructorLibrary`."""

    dm: PtyChiDMSettings
    pie: PtyChiPIESettings
    lsqml: PtyChiLSQMLSettings
    autodiff: PtyChiAutodiffSettings
    bh: PtyChiBHSettings


_ALGORITHMS: tuple[_AlgorithmSpec, ...] = (
    _AlgorithmSpec(
        'DM', lambda helper, bundle: DMReconstructor(helper, bundle.dm)._create_task_options
    ),
    _AlgorithmSpec(
        'PIE', lambda helper, bundle: PIEReconstructor(helper, bundle.pie)._create_task_options
    ),
    _AlgorithmSpec(
        'ePIE', lambda helper, bundle: EPIEReconstructor(helper, bundle.pie)._create_task_options
    ),
    _AlgorithmSpec(
        'rPIE', lambda helper, bundle: RPIEReconstructor(helper, bundle.pie)._create_task_options
    ),
    _AlgorithmSpec(
        'LSQML',
        lambda helper, bundle: LSQMLReconstructor(helper, bundle.lsqml)._create_task_options,
    ),
    _AlgorithmSpec(
        'Autodiff',
        lambda helper, bundle: AutodiffReconstructor(helper, bundle.autodiff)._create_task_options,
    ),
    _AlgorithmSpec(
        'BH', lambda helper, bundle: BHReconstructor(helper, bundle.bh)._create_task_options
    ),
)


def build_reconstructor_list(
    reconstructor_settings: PtyChiSettings,
    object_settings: PtyChiObjectSettings,
    probe_settings: PtyChiProbeSettings,
    probe_position_settings: PtyChiProbePositionSettings,
    opr_settings: PtyChiOPRSettings,
    bundle: PtyChiSettingsBundle,
    pattern_sizer: PatternSizer,
) -> list[SubprocessReconstructor]:
    """Build one :class:`SubprocessReconstructor` per pty-chi algorithm."""
    options_helper = PtyChiOptionsHelper(
        reconstructor_settings,
        object_settings,
        probe_settings,
        probe_position_settings,
        opr_settings,
        pattern_sizer,
    )

    def num_epochs() -> int:
        return reconstructor_settings.num_epochs.get_value()

    def num_sync_epochs() -> int:
        return options_helper.num_sync_epochs

    reconstructors: list[SubprocessReconstructor] = []

    for spec in _ALGORITHMS:
        build_task_options = spec.make_option_factory(options_helper, bundle)

        def build_payload(
            parameters: ReconstructInput,
            _loaded_model_path: Path | None,
            _build: _TaskOptionsBuilder = build_task_options,
        ) -> PtyChiPayload:
            return PtyChiPayload(
                task_options=_build(parameters),
                num_sync_epochs=num_sync_epochs(),
                reconstruct_input=parameters,
            )

        reconstructors.append(
            SubprocessReconstructor(
                name=spec.display_name,
                reconstruct_entry_point=_RECONSTRUCT_ENTRY,
                progress_goal_fn=num_epochs,
                build_reconstruct_payload=build_payload,
            )
        )

    return reconstructors
