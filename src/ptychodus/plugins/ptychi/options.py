"""What pty-chi accepts, declared for ptychodus to generate settings from.

pty-chi declares its own options as pydantic dataclasses, so there is nothing
to restate: :func:`default_options` hands back upstream's own
``PtychographyTaskOptions`` subclass, and the section table beside the
flattener says which of its fields are surfaced, what each is called and what
ptychodus knows about it that upstream does not.

The one addition is :attr:`PtyChiOptions.num_sync_epochs`, which chunks the
epoch loop so a run reports progress. It is ptychodus's own -- pty-chi has no
equivalent -- so it rides beside the task options rather than inside them. A
wrapper, not a subclass: pty-chi stamps ``options_class_name`` from the class
that wrote a dict and validates it on the way back in, so a ptychodus subclass
would rename every algorithm on the wire and break the files already written.

Each algorithm has its own wrapper, declared here at module scope, whose
``task`` is that algorithm's concrete options class. That is what lets the
options survive a round trip -- validated against the base class, a task would
lose the fields that distinguish its algorithm -- and what makes each
algorithm's schema complete. Declared rather than generated, because an
instance travels to a child process by pickle, which finds a class by its
qualified name.

``task`` is written as pty-chi's own ``get_dict()`` payload and read back
through :func:`~ptychodus.plugins.ptychi.task.load_task_options_from_mapping`,
so the ``options_class_name`` stamp it carries must name the declared class.
pty-chi's dataclasses track no set-ness, so ``task`` is set or unset as a
whole.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Final

from ptychi.api import (
    AutodiffPtychographyOptions,
    BHOptions,
    DMOptions,
    EPIEOptions,
    LSQMLOptions,
    PIEOptions,
    RAAROptions,
    RPIEOptions,
)
from ptychi.api.options.task import PtychographyTaskOptions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    ValidatorFunctionWrapHandler,
    field_validator,
)

from .task import dump_task_options, load_task_options_from_mapping

__all__ = [
    'AutodiffPtyChiOptions',
    'BHPtyChiOptions',
    'DMPtyChiOptions',
    'EPIEPtyChiOptions',
    'LSQMLPtyChiOptions',
    'PIEPtyChiOptions',
    'PtyChiOptions',
    'RAARPtyChiOptions',
    'RPIEPtyChiOptions',
    'apply_to_task',
    'default_options',
    'options_class_for',
]


# Written from the data by ``align_task_options_with_product`` at run time, whatever a
# document says. Spelled out rather than derived from that function: this exists to warn a
# person away from tuning a value that will be discarded, and a reader of a printed document
# cannot introspect anything. A test holds the two in agreement.
_DATA_DERIVED_TASK_FIELDS: Final = (
    'data_options.wavelength_m',
    'data_options.free_space_propagation_distance_m',
    'data_options.fft_shift',
    'object_options.pixel_size_m',
    'object_options.pixel_size_aspect_ratio',
    'object_options.slice_spacings_m',
    'object_options.determine_position_origin_coords_by',
    'object_options.position_origin_coords',
    'probe_options.pixel_size_m',
    'probe_options.pixel_size_aspect_ratio',
    'probe_options.power_constraint.probe_power',
)

_TASK_DESCRIPTION: Final = (
    "pty-chi's own task options, as pty-chi writes them; options_class_name names the "
    'algorithm, so print another algorithm rather than editing it. Written from the data '
    'when a reconstruction starts, whatever this says: ' + ', '.join(_DATA_DERIVED_TASK_FIELDS)
)


class PtyChiOptions(BaseModel):
    """One algorithm's pty-chi options, plus the two knobs ptychodus adds.

    Both additions are shapes pty-chi does not have rather than values it
    does. ``num_sync_epochs`` has no upstream equivalent at all, and the
    clustering count stands for a flag and a stride that ptychodus has always
    offered as one number. :func:`apply_to_task` is where the second becomes
    the pair pty-chi reads.

    Instantiated only through a per-algorithm subclass, which narrows
    :attr:`task` to that algorithm's options class.
    """

    # 'constants' keeps an infinite distance a float in JSON mode, which YAML writes as
    # .inf; the default would write null, which pty-chi refuses for a float field.
    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra='forbid', ser_json_inf_nan='constants'
    )

    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = PtychographyTaskOptions

    task: PtychographyTaskOptions = Field(description=_TASK_DESCRIPTION)
    num_sync_epochs: int = Field(
        100,
        ge=1,
        title='Update Interval:',
        description='Number of epochs between reports of intermediate results.',
    )
    compact_mode_update_clustering: int = Field(
        0,
        ge=0,
        title='Update Clustering:',
        description=(
            'Number of epochs between cluster updates in COMPACT batching mode (0 disables).'
        ),
    )

    # A wrap validator that never calls the inner handler: pydantic would otherwise
    # revalidate the dataclass field by field, and pty-chi's own defaults do not all
    # satisfy its annotations (a None where a float is declared, for one).
    @field_validator('task', mode='wrap')
    @classmethod
    def _load_task(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        if isinstance(value, PtychographyTaskOptions):
            task = value
        else:
            task = load_task_options_from_mapping(value)

        if type(task) is not cls.task_options_cls:
            raise ValueError(
                f'{cls.__name__} holds {cls.task_options_cls.__name__}, not {type(task).__name__}.'
            )

        return task

    @field_serializer('task')
    def _dump_task(self, task: PtychographyTaskOptions) -> dict[str, Any]:
        return json.loads(dump_task_options(task))


class DMPtyChiOptions(PtyChiOptions):
    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = DMOptions
    task: DMOptions = Field(description=_TASK_DESCRIPTION)


class RAARPtyChiOptions(PtyChiOptions):
    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = RAAROptions
    task: RAAROptions = Field(description=_TASK_DESCRIPTION)


class PIEPtyChiOptions(PtyChiOptions):
    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = PIEOptions
    task: PIEOptions = Field(description=_TASK_DESCRIPTION)


class EPIEPtyChiOptions(PtyChiOptions):
    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = EPIEOptions
    task: EPIEOptions = Field(description=_TASK_DESCRIPTION)


class RPIEPtyChiOptions(PtyChiOptions):
    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = RPIEOptions
    task: RPIEOptions = Field(description=_TASK_DESCRIPTION)


class LSQMLPtyChiOptions(PtyChiOptions):
    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = LSQMLOptions
    task: LSQMLOptions = Field(description=_TASK_DESCRIPTION)


class AutodiffPtyChiOptions(PtyChiOptions):
    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = AutodiffPtychographyOptions
    task: AutodiffPtychographyOptions = Field(description=_TASK_DESCRIPTION)


class BHPtyChiOptions(PtyChiOptions):
    task_options_cls: ClassVar[type[PtychographyTaskOptions]] = BHOptions
    task: BHOptions = Field(description=_TASK_DESCRIPTION)


def options_class_for(task_options_cls: type[PtychographyTaskOptions]) -> type[PtyChiOptions]:
    """The wrapper declared for the algorithm *task_options_cls* names.

    Raises:
        LookupError: if no wrapper holds that class.
    """
    for options_cls in PtyChiOptions.__subclasses__():
        if options_cls.task_options_cls is task_options_cls:
            return options_cls

    raise LookupError(f'No pty-chi options declared for {task_options_cls.__name__}!')


def apply_to_task(options: PtyChiOptions) -> PtychographyTaskOptions:
    """Write the options ptychodus shapes differently into the task they configure.

    pty-chi requires a clustering stride of at least one even with clustering
    off, so the count ptychodus holds becomes a flag and a stride rather than
    being passed through.
    """
    task = options.task
    clustering = options.compact_mode_update_clustering
    task.reconstructor_options.compact_mode_update_clustering = clustering > 0
    task.reconstructor_options.compact_mode_update_clustering_stride = max(1, clustering)
    return task


def default_options(task_options_cls: type[PtychographyTaskOptions]) -> PtyChiOptions:
    """Return the declared defaults for the algorithm *task_options_cls* names.

    Where ptychodus has long chosen a different starting point from pty-chi's,
    it is stated here rather than left to the settings file: an affine
    constraint starts with no degrees of freedom selected, so that enabling
    the constraint does not silently also enable four corrections.

    LSQML's first-epoch probe rescale is turned off for a related reason. That epoch
    performs no object or probe update at all -- it only measures intensity, to correct
    a probe whose power does not match the data -- and ptychodus already hands over a
    probe scaled to match. Leaving it on would spend an epoch confirming a correction of
    one. It stays available for anyone who wants pty-chi's own estimate instead, and the
    two agree where both run.
    """
    task = task_options_cls()
    task.probe_position_options.affine_transform_constraint.degrees_of_freedom = []

    # Only LSQML declares it; every other algorithm's reconstructor options lack the
    # field, and a library must not fail to offer itself over a default.
    if hasattr(task.reconstructor_options, 'rescale_probe_intensity_in_first_epoch'):
        task.reconstructor_options.rescale_probe_intensity_in_first_epoch = False

    return options_class_for(task_options_cls)(task=task)
