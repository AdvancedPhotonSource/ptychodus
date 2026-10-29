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
would rename every algorithm on the wire and break the YAML files already
written.

``task`` skips validation for the same reason it cannot be a subclass. Its
declared type is the base, but what it holds is always one of the eight
algorithm-specific subclasses, and validating against the base would coerce
away the fields that distinguish them.
"""

from __future__ import annotations

from ptychi.api.options.task import PtychographyTaskOptions
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

__all__ = [
    'PtyChiOptions',
    'default_options',
]


class PtyChiOptions(BaseModel):
    """One algorithm's pty-chi options, plus the two knobs ptychodus adds.

    Both additions are shapes pty-chi does not have rather than values it
    does. ``num_sync_epochs`` has no upstream equivalent at all, and the
    clustering count stands for a flag and a stride that ptychodus has always
    offered as one number. :func:`apply_to_task` is where the second becomes
    the pair pty-chi reads.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task: SkipValidation[PtychographyTaskOptions]
    num_sync_epochs: int = Field(
        25,
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
    # field, and a backend must not fail to offer itself over a default.
    if hasattr(task.reconstructor_options, 'rescale_probe_intensity_in_first_epoch'):
        task.reconstructor_options.rescale_probe_intensity_in_first_epoch = False

    return PtyChiOptions(task=task)
