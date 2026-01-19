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
