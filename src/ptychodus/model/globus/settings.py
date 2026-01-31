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

        self.input_data_endpoint_id = self._group.create_uuid_parameter(
            'InputDataEndpointID', UUID(int=0)
        )
        self.input_data_globus_path = self._group.create_string_parameter(
            'InputDataGlobusPath', '/~/path/to/input/data'
        )
        self.input_data_posix_path = self._group.create_path_parameter(
            'InputDataPosixPath', Path('/path/to/input/data')
        )

        self.compute_endpoint_id = self._group.create_uuid_parameter(
            'ComputeEndpointID', UUID(int=0)
        )
        self.compute_data_endpoint_id = self._group.create_uuid_parameter(
            'ComputeDataEndpointID', UUID(int=0)
        )
        self.compute_data_globus_path = self._group.create_string_parameter(
            'ComputeDataGlobusPath', '/~/path/to/compute/data'
        )
        self.compute_data_posix_path = self._group.create_path_parameter(
            'ComputeDataPosixPath', Path('/path/to/compute/data')
        )

        self.status_auto_refresh = self._group.create_boolean_parameter('StatusAutoRefresh', False)
        self.status_refresh_interval_s = self._group.create_integer_parameter(
            'StatusRefreshIntervalInSeconds', 10, minimum=10, maximum=86400
        )

        self.output_data_endpoint_id = self._group.create_uuid_parameter(
            'OutputDataEndpointID', UUID(int=0)
        )
        self.output_data_globus_path = self._group.create_string_parameter(
            'OutputDataGlobusPath', '/~/path/to/output/data'
        )
        self.output_data_posix_path = self._group.create_path_parameter(
            'OutputDataPosixPath', Path('/path/to/output/data')
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
