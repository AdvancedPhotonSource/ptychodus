from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class GlobusSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Globus')
        self._group.add_observer(self)

        # Transfer copies only those files where the source file differs from
        # the destination file. The sync level is an integer in the 0-3 range,
        # and it controls what checks are used to determine if a file is
        # different. Higher levels include the checks from lower levels. Delete
        # tasks are always null:
        #
        # 0 Copy files that do not exist at the destination.
        # 1 Copy files if the size of the destination does not match the size of the source.
        # 2 Copy files if the timestamp of the destination is older than the timestamp of the source.
        # 3 Copy files if checksums of the source and destination do not match.
        self.transfer_sync_level = self._group.create_integer_parameter(
            'TransferSyncLevel', 3, minimum=0, maximum=3
        )

        self.input_collection_id = self._group.create_uuid_parameter(
            'InputCollectionID', UUID(int=0)
        )
        self.input_collection_globus_path = self._group.create_string_parameter(
            'InputCollectionGlobusPath', '/~/path/to/input/data'
        )
        self.input_collection_posix_path = self._group.create_path_parameter(
            'InputCollectionPosixPath', Path('/path/to/input/data')
        )

        self.compute_endpoint_id = self._group.create_uuid_parameter(
            'ComputeEndpointID', UUID(int=0)
        )
        self.compute_function_id = self._group.create_uuid_parameter(
            'ComputeFunctionID', UUID(int=0)
        )
        self.compute_flow_id = self._group.create_uuid_parameter('ComputeFlowID', UUID(int=0))
        self.compute_collection_id = self._group.create_uuid_parameter(
            'ComputeCollectionID', UUID(int=0)
        )
        self.compute_collection_globus_path = self._group.create_string_parameter(
            'ComputeCollectionGlobusPath', '/~/path/to/compute/data'
        )
        self.compute_collection_posix_path = self._group.create_path_parameter(
            'ComputeCollectionPosixPath', Path('/path/to/compute/data')
        )

        self.status_auto_refresh = self._group.create_boolean_parameter('StatusAutoRefresh', False)
        self.status_refresh_interval_s = self._group.create_integer_parameter(
            'StatusRefreshIntervalInSeconds', 30, minimum=10, maximum=86400
        )

        self.output_collection_id = self._group.create_uuid_parameter(
            'OutputCollectionID', UUID(int=0)
        )
        self.output_collection_globus_path = self._group.create_string_parameter(
            'OutputCollectionGlobusPath', '/~/path/to/output/data'
        )
        self.output_collection_posix_path = self._group.create_path_parameter(
            'OutputCollectionPosixPath', Path('/path/to/output/data')
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
