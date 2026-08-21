from __future__ import annotations

from ptychodus.api.observer import Observable, Observer

from ..model.task_monitor import TaskProgressMonitor
from ..view.widgets import TaskStatusView


class TaskStatusController(Observer):
    """Shows a TaskStatusView while its monitor reports work in progress.

    Shared by the diffraction, product, and fluorescence panels: all three want the
    same show-while-processing behavior over their own *TaskMonitor.
    """

    def __init__(self, monitor: TaskProgressMonitor, view: TaskStatusView) -> None:
        super().__init__()
        self._monitor = monitor
        self._view = view

        view.stop_button.clicked.connect(monitor.stop_processing)

        self._sync_model_to_view()
        monitor.add_observer(self)

    def _sync_model_to_view(self) -> None:
        progress_goal = self._monitor.get_progress_goal()
        progress_bar = self._view.progress_bar

        if self._monitor.is_processing and progress_goal > 0:
            progress_bar.show()
            progress_bar.setRange(0, progress_goal)
            progress_bar.setValue(self._monitor.get_progress())
            self._view.stop_button.show()
        else:
            progress_bar.hide()
            self._view.stop_button.hide()

    def _update(self, observable: Observable) -> None:
        if observable is self._monitor:
            self._sync_model_to_view()
