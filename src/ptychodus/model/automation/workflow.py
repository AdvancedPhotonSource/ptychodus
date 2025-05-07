from collections.abc import Iterator
from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

from .settings import AutomationSettings


class CurrentFileBasedWorkflow(FileBasedWorkflow, Observable, Observer):
    def __init__(
        self,
        settings: AutomationSettings,
        workflow_chooser: PluginChooser[FileBasedWorkflow],
    ) -> None:
        super().__init__()
        self._workflow_chooser = workflow_chooser

        workflow_chooser.synchronize_with_parameter(settings.strategy)
        workflow_chooser.add_observer(self)

    def get_available_workflows(self) -> Iterator[str]:
        for plugin in self._workflow_chooser:
            yield plugin.display_name

    def get_workflow(self) -> str:
        return self._workflow_chooser.get_current_plugin().display_name

    def set_workflow(self, name: str) -> None:
        self._workflow_chooser.set_current_plugin(name)

    @property
    def is_watch_recursive(self) -> bool:
        workflow = self._workflow_chooser.get_current_plugin().strategy
        return workflow.is_watch_recursive

    def get_watch_file_pattern(self) -> str:
        workflow = self._workflow_chooser.get_current_plugin().strategy
        return workflow.get_watch_file_pattern()

    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        workflow = self._workflow_chooser.get_current_plugin().strategy
        workflow.execute(api, file_path)

    def _update(self, observable: Observable) -> None:
        if observable is self._workflow_chooser:
            self.notify_observers()
