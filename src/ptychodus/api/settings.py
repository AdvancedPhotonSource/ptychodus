from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
import configparser
import logging

from .observer import Observable
from .parametric import (
    ParameterGroup,
    PathParameter,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathPrefixChange:
    find_path_prefix: Path
    replacement_path_prefix: Path


class SettingsRegistry(Observable):
    def __init__(self) -> None:
        super().__init__()
        self._parameter_group = ParameterGroup()
        self._file_filter_list: list[str] = ['Initialization Files (*.ini)']

    def create_group(self, name: str) -> ParameterGroup:
        return self._parameter_group.create_group(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameter_group.groups())

    def __getitem__(self, name: str) -> ParameterGroup:
        return self._parameter_group.get_group(name)

    def __len__(self) -> int:
        return len(self._parameter_group.groups())

    def get_open_file_filters(self) -> Sequence[str]:
        return self._file_filter_list

    def get_open_file_filter(self) -> str:
        return self._file_filter_list[0]

    def open_settings(self, file_path: Path) -> None:
        config = configparser.ConfigParser(interpolation=None)
        logger.debug(f'Reading settings from "{file_path}"')

        try:
            config.read(file_path)
        except Exception as exc:
            logger.exception(exc)
            return

        # TODO generalize to support nested parameter groups
        for group_name, group in self._parameter_group.groups().items():
            try:
                group_config = config[group_name]
            except KeyError:
                pass
            else:
                for parameter_name, parameter in group.parameters().items():
                    try:
                        value_string = group_config[parameter_name]
                    except KeyError:
                        pass
                    else:
                        parameter.set_value_from_string(value_string)

        self.notify_observers()

    def get_save_file_filters(self) -> Sequence[str]:
        return self._file_filter_list

    def get_save_file_filter(self) -> str:
        return self._file_filter_list[0]

    def save_settings(
        self, file_path: Path, change_path_prefix: PathPrefixChange | None = None
    ) -> None:
        config = configparser.ConfigParser(interpolation=None)
        setattr(config, 'optionxform', lambda option: option)

        for group_name, group in self._parameter_group.groups().items():
            config.add_section(group_name)

            for parameter_name, parameter in group.parameters().items():
                value_string = parameter.get_value_as_string()

                if change_path_prefix and isinstance(parameter, PathParameter):
                    modified_path = parameter.change_path_prefix(
                        change_path_prefix.find_path_prefix,
                        change_path_prefix.replacement_path_prefix,
                    )
                    value_string = str(modified_path)

                config.set(group_name, parameter_name, value_string)

        logger.debug(f'Writing settings to "{file_path}"')

        try:
            with file_path.open(mode='w') as config_file:
                config.write(config_file)
        except Exception as exc:
            logger.exception(exc)
            return
