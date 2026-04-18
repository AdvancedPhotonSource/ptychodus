from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class GenesisSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Genesis')
        self._group.add_observer(self)

        self.facility = self._group.create_string_parameter('Facility', 'NERSC')
        self.compute_resource_id = self._group.create_string_parameter('ComputeResourceID', '')
        self.globus_transfer_provider = self._group.create_string_parameter(
            'GlobusTransferProvider', 'AmSC'
        )

        self.local_collection_id = self._group.create_uuid_parameter(
            'LocalCollectionID', UUID(int=0)
        )
        self.local_collection_globus_path = self._group.create_string_parameter(
            'LocalCollectionGlobusPath', '/~/path/to/local/data'
        )
        self.local_collection_posix_path = self._group.create_path_parameter(
            'LocalCollectionPosixPath', Path('/path/to/local/data')
        )

        self.remote_collection_id = self._group.create_uuid_parameter(
            'RemoteCollectionID', UUID(int=0)
        )
        self.remote_collection_globus_path = self._group.create_string_parameter(
            'RemoteCollectionGlobusPath', '/~/path/to/remote/data'
        )
        self.remote_collection_posix_path = self._group.create_path_parameter(
            'RemoteCollectionPosixPath', Path('/path/to/remote/data')
        )

        self.status_auto_refresh = self._group.create_boolean_parameter('StatusAutoRefresh', False)
        self.status_refresh_interval_s = self._group.create_integer_parameter(
            'StatusRefreshIntervalInSeconds', 30, minimum=10, maximum=86400
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
