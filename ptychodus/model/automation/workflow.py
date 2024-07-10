from collections.abc import Sequence
from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

from .settings import AutomationSettings


class CurrentFileBasedWorkflow(FileBasedWorkflow, Observable, Observer):

    def __init__(self, settings: AutomationSettings,
                 workflowChooser: PluginChooser[FileBasedWorkflow]) -> None:
        super().__init__()
        self._settings = settings
        self._workflowChooser = workflowChooser

        settings.addObserver(self)
        workflowChooser.addObserver(self)

    def getAvailableWorkflows(self) -> Sequence[str]:
        return self._workflowChooser.getDisplayNameList()

    def getWorkflow(self) -> str:
        return self._workflowChooser.currentPlugin.displayName

    def setWorkflow(self, name: str) -> None:
        self._workflowChooser.setCurrentPluginByName(name)
        self._settings.strategy.value = self._workflowChooser.currentPlugin.simpleName

    def getFilePattern(self) -> str:
        workflow = self._workflowChooser.currentPlugin.strategy
        return workflow.getFilePattern()

    def execute(self, api: WorkflowAPI, filePath: Path) -> None:
        workflow = self._workflowChooser.currentPlugin.strategy
        workflow.execute(api, filePath)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.setWorkflow(self._settings.strategy.value)
        if observable is self._workflowChooser:
            self.notifyObservers()
