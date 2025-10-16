from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import ParameterGroup


class DataLocator(ABC, Observable):
    @abstractmethod
    def set_endpoint_id(self, endpoint_id: UUID) -> None:
        pass

    @abstractmethod
    def get_endpoint_id(self) -> UUID:
        pass

    @abstractmethod
    def set_globus_path(self, globus_path: str) -> None:
        pass

    @abstractmethod
    def get_globus_path(self) -> str:
        pass

    @abstractmethod
    def set_posix_path(self, posix_path: Path) -> None:
        pass

    @abstractmethod
    def get_posix_path(self) -> Path:
        pass


class SimpleDataLocator(DataLocator, Observer):
    def __init__(self, group: ParameterGroup, entry_prefix: str) -> None:
        super().__init__()
        self._endpoint_id = group.create_uuid_parameter(
            f'{entry_prefix}DataEndpointID', UUID(int=0)
        )
        self._globus_path = group.create_string_parameter(
            f'{entry_prefix}DataGlobusPath',
            f'/~/path/to/{entry_prefix.lower()}/data',
        )
        self._posix_path = group.create_path_parameter(
            f'{entry_prefix}DataPosixPath',
            Path(f'/path/to/{entry_prefix.lower()}/data'),
        )

        self._endpoint_id.add_observer(self)
        self._globus_path.add_observer(self)
        self._posix_path.add_observer(self)

    def set_endpoint_id(self, endpoint_id: UUID) -> None:
        self._endpoint_id.set_value(endpoint_id)

    def get_endpoint_id(self) -> UUID:
        return self._endpoint_id.get_value()

    def set_globus_path(self, globus_path: str) -> None:
        self._globus_path.set_value(globus_path)

    def get_globus_path(self) -> str:
        return self._globus_path.get_value()

    def set_posix_path(self, posix_path: Path) -> None:
        self._posix_path.set_value(posix_path)

    def get_posix_path(self) -> Path:
        return self._posix_path.get_value()

    def _update(self, observable: Observable) -> None:
        if observable is self._endpoint_id:
            self.notify_observers()
        elif observable is self._globus_path:
            self.notify_observers()
        elif observable is self._posix_path:
            self.notify_observers()


class OutputDataLocator(DataLocator, Observer):
    def __init__(
        self, group: ParameterGroup, entry_prefix: str, input_data_locator: DataLocator
    ) -> None:
        super().__init__()
        self._use_round_trip = group.create_boolean_parameter('UseRoundTrip', True)
        self._output_data_locator = SimpleDataLocator(group, entry_prefix)
        self._input_data_locator = input_data_locator

        self._use_round_trip.add_observer(self)
        self._input_data_locator.add_observer(self)
        self._output_data_locator.add_observer(self)

    def set_round_trip_enabled(self, enable: bool) -> None:
        self._use_round_trip.set_value(enable)

    def is_round_trip_enabled(self) -> bool:
        return self._use_round_trip.get_value()

    def set_endpoint_id(self, endpoint_id: UUID) -> None:
        self._output_data_locator.set_endpoint_id(endpoint_id)

    def get_endpoint_id(self) -> UUID:
        return (
            self._input_data_locator.get_endpoint_id()
            if self._use_round_trip.get_value()
            else self._output_data_locator.get_endpoint_id()
        )

    def set_globus_path(self, globus_path: str) -> None:
        self._output_data_locator.set_globus_path(globus_path)

    def get_globus_path(self) -> str:
        return (
            self._input_data_locator.get_globus_path()
            if self._use_round_trip.get_value()
            else self._output_data_locator.get_globus_path()
        )

    def set_posix_path(self, posix_path: Path) -> None:
        self._output_data_locator.set_posix_path(posix_path)

    def get_posix_path(self) -> Path:
        return (
            self._input_data_locator.get_posix_path()
            if self._use_round_trip.get_value()
            else self._output_data_locator.get_posix_path()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._use_round_trip:
            self.notify_observers()
        elif observable is self._input_data_locator:
            self.notify_observers()
        elif observable is self._output_data_locator:
            self.notify_observers()
