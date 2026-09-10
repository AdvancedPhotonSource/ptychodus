from __future__ import annotations

from ..task_monitor import TaskProgressMonitor

__all__ = [
    'ProductTaskMonitor',
]


class ProductTaskMonitor(TaskProgressMonitor):
    """Progress and stop control for queued product construction.

    Products enqueued with ``block=False`` land as pending stubs and are finalized on
    the shared background worker once their diffraction dataset finishes loading. This
    monitor is what the Products panel's status strip observes.
    """
