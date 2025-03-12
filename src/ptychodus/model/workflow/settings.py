from __future__ import annotations
from uuid import UUID

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class WorkflowSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Workflow')
        self._settingsGroup.add_observer(self)

        self.computeEndpointID = self._settingsGroup.create_uuid_parameter(
            'ComputeEndpointID', UUID(int=0)
        )
        self.statusRefreshIntervalInSeconds = self._settingsGroup.create_integer_parameter(
            'StatusRefreshIntervalInSeconds', 10
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()
