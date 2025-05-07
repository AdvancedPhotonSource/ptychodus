from __future__ import annotations
from uuid import UUID

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class WorkflowSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Workflow')
        self._group.add_observer(self)

        self.compute_endpoint_id = self._group.create_uuid_parameter(
            'ComputeEndpointID', UUID(int=0)
        )
        self.status_refresh_interval_s = self._group.create_integer_parameter(
            'StatusRefreshIntervalInSeconds', 10
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
