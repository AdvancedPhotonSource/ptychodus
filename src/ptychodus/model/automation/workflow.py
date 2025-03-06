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
        workflowChooser: PluginChooser[FileBasedWorkflow],
    ) -> None:
        super().__init__()
        self._workflowChooser = workflowChooser

        workflowChooser.synchronize_with_parameter(settings.strategy)
        workflowChooser.addObserver(self)

    def getAvailableWorkflows(self) -> Iterator[str]:
        for plugin in self._workflowChooser:
            yield plugin.display_name

    def getWorkflow(self) -> str:
        return self._workflowChooser.get_current_plugin().display_name

    def setWorkflow(self, name: str) -> None:
        self._workflowChooser.set_current_plugin(name)

    @property
    def isWatchRecursive(self) -> bool:
        workflow = self._workflowChooser.get_current_plugin().strategy
        return workflow.isWatchRecursive

    def getWatchFilePattern(self) -> str:
        workflow = self._workflowChooser.get_current_plugin().strategy
        return workflow.getWatchFilePattern()

    def execute(self, api: WorkflowAPI, filePath: Path) -> None:
        workflow = self._workflowChooser.get_current_plugin().strategy
        workflow.execute(api, filePath)

    def update(self, observable: Observable) -> None:
        if observable is self._workflowChooser:
            self.notifyObservers()
